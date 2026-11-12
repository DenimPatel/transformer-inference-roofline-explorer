/**
 * A kernel-by-kernel trace of one forward pass of a Qwen model on one H100.
 *
 * Pure data and arithmetic, no React. `buildTrace` turns a model architecture
 * plus a serving setting (phase, batch, context) into an ordered list of
 * steps. Each step is roughly one GPU kernel and says where its bytes come
 * from, how many FLOPs it does, which unit runs it, and how long it takes
 * under a simple roofline: time = max(bytes / bandwidth, flops / peak) plus a
 * small fixed per-kernel gap.
 *
 * Architecture numbers come from the models' Hugging Face config.json files:
 *   https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
 *   https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
 * H100 numbers come from src/data/hardware/nvidia-h100-sxm5.json plus NVIDIA's
 * Hopper architecture whitepaper (L2, SMEM, register file).
 */

import { HARDWARE_PROFILES, effectiveMfu, type HardwareProfile } from './hardware';

// ---------------------------------------------------------------------------
// Model architectures
// ---------------------------------------------------------------------------

export type LayerKind = 'full' | 'linear';
export type ModelId = 'qwen3.8-27b' | 'qwen3-8b';

export interface ModelArch {
  id: ModelId;
  name: string;
  short: string;
  hidden: number;
  layers: number;
  vocab: number;
  ffn: number;
  layerTypes: LayerKind[];
  attn: {
    qHeads: number;
    kvHeads: number;
    headDim: number;
    /** Qwen3.8 gates the attention output, which doubles q_proj's width. */
    outputGate: boolean;
  };
  /** Gated DeltaNet (linear attention) layers, when the model has them. */
  linear?: {
    qkHeads: number;
    vHeads: number;
    headDim: number;
    convKernel: number;
  };
  bytesPerParam: number;
  blurb: string;
  sourceUrl: string;
}

export const QWEN38_27B: ModelArch = {
  id: 'qwen3.8-27b',
  name: 'Qwen3.8-27B',
  short: '27B hybrid',
  hidden: 5120,
  layers: 64,
  vocab: 248320,
  ffn: 17408,
  // 16 x [3 x Gated DeltaNet, 1 x gated full attention]
  layerTypes: Array.from({ length: 64 }, (_, i) => ((i + 1) % 4 === 0 ? 'full' : 'linear')),
  attn: { qHeads: 24, kvHeads: 4, headDim: 256, outputGate: true },
  linear: { qkHeads: 16, vHeads: 48, headDim: 128, convKernel: 4 },
  bytesPerParam: 2,
  blurb:
    'Hybrid: three Gated DeltaNet (linear-attention) layers for every full-attention layer. Only the 16 full layers keep a KV cache; the other 48 keep a fixed-size state.',
  sourceUrl: 'https://huggingface.co/Qwen/Qwen3.8-27B',
};

export const QWEN3_8B: ModelArch = {
  id: 'qwen3-8b',
  name: 'Qwen3-8B',
  short: '8B classic',
  hidden: 4096,
  layers: 36,
  vocab: 151936,
  ffn: 12288,
  layerTypes: Array.from({ length: 36 }, () => 'full' as const),
  attn: { qHeads: 32, kvHeads: 8, headDim: 128, outputGate: false },
  bytesPerParam: 2,
  blurb:
    'A classic decoder: 36 identical layers, each with grouped-query attention (32 query heads sharing 8 KV heads) and a SwiGLU feed-forward block.',
  sourceUrl: 'https://huggingface.co/Qwen/Qwen3-8B',
};

export const MODEL_ARCHS: Record<ModelId, ModelArch> = {
  'qwen3.8-27b': QWEN38_27B,
  'qwen3-8b': QWEN3_8B,
};

// Parameter counts per block --------------------------------------------------

export function attnParams(m: ModelArch) {
  const { qHeads, kvHeads, headDim, outputGate } = m.attn;
  const q = m.hidden * qHeads * headDim * (outputGate ? 2 : 1);
  const kv = 2 * m.hidden * kvHeads * headDim;
  const o = qHeads * headDim * m.hidden;
  return { qkv: q + kv, o, total: q + kv + o };
}

export function linearParams(m: ModelArch) {
  if (!m.linear) return { inProj: 0, out: 0, conv: 0, total: 0 };
  const { qkHeads, vHeads, headDim, convKernel } = m.linear;
  const qkvzWidth = 2 * qkHeads * headDim + 2 * vHeads * headDim; // q, k, v, z (gate)
  const inProj = m.hidden * (qkvzWidth + 2 * vHeads); // + beta and decay (a) per head
  const conv = (2 * qkHeads * headDim + vHeads * headDim) * convKernel;
  const out = vHeads * headDim * m.hidden;
  return { inProj, out, conv, total: inProj + conv + out };
}

export const ffnParams = (m: ModelArch) => 3 * m.hidden * m.ffn;

export function countLayers(m: ModelArch) {
  const full = m.layerTypes.filter((t) => t === 'full').length;
  return { full, linear: m.layerTypes.length - full };
}

export function totalParams(m: ModelArch) {
  const { full, linear } = countLayers(m);
  const ffn = ffnParams(m);
  const embed = m.vocab * m.hidden;
  const head = m.vocab * m.hidden; // untied in both models
  return {
    embed,
    head,
    layers: full * (attnParams(m).total + ffn) + linear * (linearParams(m).total + ffn),
    get total() { return this.embed + this.head + this.layers; },
  };
}

/** Weight bytes that must stream from HBM on every forward pass (the embedding is a lookup, not a stream). */
export function streamedWeightBytes(m: ModelArch) {
  const p = totalParams(m);
  return (p.layers + p.head) * m.bytesPerParam;
}

/** KV-cache bytes per token of context: only full-attention layers keep one. */
export function kvBytesPerToken(m: ModelArch) {
  return countLayers(m).full * 2 * m.attn.kvHeads * m.attn.headDim * m.bytesPerParam;
}

/** DeltaNet recurrent state per sequence (fp32), independent of context length. */
export function stateBytesPerSeq(m: ModelArch) {
  if (!m.linear) return 0;
  const { linear } = countLayers(m);
  const { qkHeads, vHeads, headDim, convKernel } = m.linear;
  const recurrent = vHeads * headDim * headDim * 4;
  const conv = (2 * qkHeads * headDim + vHeads * headDim) * (convKernel - 1) * m.bytesPerParam;
  return linear * (recurrent + conv);
}

// ---------------------------------------------------------------------------
// The H100
// ---------------------------------------------------------------------------

export interface GpuSpec {
  name: string;
  hbmGB: number;
  hbmBw: number;        // bytes/s
  tensorFlops: number;  // dense BF16 FLOP/s
  cudaFlops: number;    // FP32 CUDA-core FLOP/s
  mfu: number;
  pcieBw: number;       // bytes/s
  sms: number;
  tensorCoresPerSm: number;
  l2MB: number;
  smemKB: number;
  regKB: number;
  hbmStacks: number;
  hbmStacksActive: number;
}

const H100_PROFILE: HardwareProfile | undefined = HARDWARE_PROFILES.find((h) => h.id === 'H100 SXM5');

export const H100: GpuSpec = {
  name: 'H100 SXM5',
  hbmGB: H100_PROFILE?.capacity ?? 80,
  hbmBw: (H100_PROFILE?.memBw ?? 3.35) * 1e12,
  tensorFlops: (H100_PROFILE?.tflops ?? 989) * 1e12,
  cudaFlops: (H100_PROFILE?.vectorTflops ?? 67) * 1e12,
  mfu: H100_PROFILE ? effectiveMfu(H100_PROFILE) : 0.82,
  pcieBw: (H100_PROFILE?.hostBwGBs ?? 64) * 1e9,
  sms: 132,
  tensorCoresPerSm: 4,
  l2MB: 50,
  smemKB: 228,
  regKB: 256,
  hbmStacks: 6,
  hbmStacksActive: 5,
};

/** Well-tuned kernels reach roughly this fraction of peak HBM bandwidth. */
export const MEM_EFF = 0.85;
/** Gap between back-to-back kernels inside a CUDA graph, in microseconds (estimate). */
export const KERNEL_GAP_US = 1.5;
/** CPU-side scheduling plus one CUDA-graph launch per forward pass, in microseconds (estimate). */
export const LAUNCH_US = 15;
/** Round-trip latency of a tiny PCIe copy, in microseconds (estimate). */
export const PCIE_LAT_US = 8;
/** CUDA context, activations, and workspace kept resident beside the weights (estimate). */
export const RUNTIME_RESERVE_BYTES = 2e9;

export const tensorRidge = (g: GpuSpec) => g.tensorFlops / g.hbmBw;
export const cudaRidge = (g: GpuSpec) => g.cudaFlops / g.hbmBw;

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------

export type Phase = 'decode' | 'prefill';
export type StepKind = 'host' | 'pcie' | 'embed' | 'gemm' | 'attn' | 'state' | 'vector' | 'sample';
export type Unit = 'cpu' | 'pcie' | 'copy' | 'tensor' | 'cuda';
export type HbmRegion = 'weights' | 'kv' | 'state' | 'act' | null;
export type Bound = 'memory' | 'compute' | 'latency' | 'overlapped';

export interface TraceStep {
  id: string;
  /** Group the step belongs to, e.g. "DeltaNet layer". */
  group: string;
  title: string;
  kind: StepKind;
  unit: Unit;
  region: HbmRegion;
  /** Layer type this step repeats over, if any. */
  layerKind?: LayerKind;
  repeat: number;
  /** HBM (or PCIe) bytes per instance. */
  bytes: number;
  /** Bytes per instance that are weights, the rest being KV, state or activations. */
  weightBytes: number;
  flops: number;
  memUs: number;
  computeUs: number;
  overheadUs: number;
  /** Time of one instance. */
  timeUs: number;
  /** Time of all instances (timeUs * repeat), the step's share of the pass. */
  totalUs: number;
  bound: Bound;
  intensity: number;
  ridge: number;
  /** Fraction of peak HBM bandwidth in use while this step runs. */
  hbmBusy: number;
  /** Fraction of peak FLOPs of the step's own unit in use. */
  unitBusy: number;
  explain: string;
}

export interface TraceOptions {
  phase: Phase;
  /** Concurrent sequences in decode. Prefill is modelled for one prompt. */
  batch: number;
  /** Tokens already in each sequence's KV cache (decode). */
  context: number;
  /** Prompt length (prefill). */
  prompt: number;
}

export interface MemoryUse {
  weights: number;
  kv: number;
  state: number;
  reserve: number;
  total: number;
  capacity: number;
  fits: boolean;
}

export interface Trace {
  arch: ModelArch;
  gpu: GpuSpec;
  opts: TraceOptions;
  steps: TraceStep[];
  totalUs: number;
  memory: MemoryUse;
  /** Lower bound from streaming every weight once at peak bandwidth. */
  weightFloorUs: number;
  /** Lower bound from the matmul FLOPs at peak. */
  computeFloorUs: number;
  memBoundShare: number;
  /** Rows per matmul: batch in decode, prompt length in prefill. */
  rows: number;
}

// Formatting -------------------------------------------------------------------

export function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(b >= 1e10 ? 1 : 2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(b >= 1e7 ? 0 : 1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(b >= 1e4 ? 0 : 1)} KB`;
  return `${Math.round(b)} B`;
}

export function fmtFlops(f: number): string {
  if (f >= 1e12) return `${(f / 1e12).toFixed(2)} TFLOP`;
  if (f >= 1e9) return `${(f / 1e9).toFixed(f >= 1e10 ? 0 : 1)} GFLOP`;
  if (f >= 1e6) return `${(f / 1e6).toFixed(f >= 1e7 ? 0 : 1)} MFLOP`;
  if (f >= 1e3) return `${(f / 1e3).toFixed(0)} KFLOP`;
  return `${Math.round(f)} FLOP`;
}

export function fmtUs(us: number): string {
  if (us >= 1000) return `${(us / 1000).toFixed(us >= 10000 ? 1 : 2)} ms`;
  if (us >= 10) return `${us.toFixed(0)} µs`;
  return `${us.toFixed(1)} µs`;
}

export const fmtCount = (n: number) => n.toLocaleString('en-US');

// Builder ----------------------------------------------------------------------

type StepInput = Omit<TraceStep,
  'memUs' | 'computeUs' | 'overheadUs' | 'timeUs' | 'totalUs' | 'bound' | 'intensity' | 'ridge' | 'hbmBusy' | 'unitBusy' | 'weightBytes'
> & { weightBytes?: number; fixedUs?: number; overlapped?: boolean };

function finish(g: GpuSpec, s: StepInput): TraceStep {
  const peak = s.unit === 'tensor' ? g.tensorFlops : g.cudaFlops;
  const isGpu = s.unit === 'tensor' || s.unit === 'cuda' || s.unit === 'copy';
  const memUs = isGpu ? (s.bytes / (g.hbmBw * MEM_EFF)) * 1e6 : s.unit === 'pcie' ? (s.bytes / g.pcieBw) * 1e6 : 0;
  const computeUs = isGpu && s.flops > 0 ? (s.flops / (peak * g.mfu)) * 1e6 : 0;
  const overheadUs = s.fixedUs ?? (isGpu ? KERNEL_GAP_US : 0);
  const timeUs = s.overlapped ? 0 : Math.max(memUs, computeUs) + overheadUs;
  const bound: Bound = s.overlapped
    ? 'overlapped'
    : !isGpu || Math.max(memUs, computeUs) < overheadUs
      ? 'latency'
      : computeUs > memUs ? 'compute' : 'memory';
  const busyUs = timeUs > 0 ? timeUs : 1;
  return {
    ...s,
    weightBytes: s.weightBytes ?? 0,
    memUs,
    computeUs,
    overheadUs,
    timeUs,
    totalUs: timeUs * s.repeat,
    bound,
    intensity: s.bytes > 0 ? s.flops / s.bytes : 0,
    ridge: s.unit === 'tensor' ? tensorRidge(g) : cudaRidge(g),
    hbmBusy: isGpu ? Math.min(1, s.bytes / (busyUs * 1e-6) / g.hbmBw) : 0,
    unitBusy: isGpu ? Math.min(1, s.flops / (busyUs * 1e-6) / peak) : 0,
  };
}

export function buildTrace(arch: ModelArch, gpu: GpuSpec, opts: TraceOptions): Trace {
  const m = arch;
  const B = m.bytesPerParam;
  const decode = opts.phase === 'decode';
  const R = decode ? Math.max(1, opts.batch) : Math.max(1, opts.prompt); // rows through every matmul
  const seqs = decode ? Math.max(1, opts.batch) : 1;
  const L = decode ? opts.context : opts.prompt;
  const h = m.hidden;
  const { full: nFull, linear: nLinear } = countLayers(m);
  const { qHeads: qH, kvHeads: kvH, headDim: d, outputGate } = m.attn;
  const ap = attnParams(m);
  const lp = linearParams(m);
  const ridgeT = tensorRidge(gpu);
  const rowsWord = decode
    ? (R === 1 ? 'the single token' : `all ${fmtCount(R)} tokens in the batch (one per sequence)`)
    : `all ${fmtCount(R)} prompt tokens at once`;

  const steps: TraceStep[] = [];
  const add = (s: StepInput) => steps.push(finish(gpu, s));

  // Shared builders ----------------------------------------------------------

  const gemm = (id: string, group: string, title: string, params: number, din: number, dout: number,
    repeat: number, layerKind: LayerKind | undefined, explainFn: (t: { w: number; i: number }) => string, rows = R) => {
    const w = params * B;
    const bytes = w + rows * (din + dout) * B;
    const flops = 2 * rows * params;
    add({
      id, group, title, kind: 'gemm', unit: 'tensor', region: 'weights', layerKind, repeat,
      bytes, weightBytes: w, flops, explain: explainFn({ w, i: flops / bytes }),
    });
  };

  const norm = (id: string, group: string, title: string, repeat: number, layerKind: LayerKind | undefined, what: string) => {
    add({
      id, group, title, kind: 'vector', unit: 'cuda', region: 'act', layerKind, repeat,
      bytes: 2 * R * h * B + h * B, weightBytes: h * B, flops: 5 * R * h,
      explain: `${what} The ${fmtCount(h)}-wide activation vector for ${rowsWord} is read, scaled by its root-mean-square and a learned weight, and written back. This runs on the CUDA cores, not the tensor cores: a handful of FLOPs per element, so it can never come close to either ridge. It is cheap only because the vectors are small.`,
    });
  };

  const gemmTail = (i: number) =>
    i < ridgeT
      ? `Arithmetic intensity is about ${i.toFixed(i < 10 ? 1 : 0)} FLOPs per byte against a ridge of ${ridgeT.toFixed(0)}, so the tensor cores finish each tile long before the next one arrives: this step is memory-bound.`
      : `Arithmetic intensity is about ${i.toFixed(0)} FLOPs per byte, above the ${ridgeT.toFixed(0)} ridge, so the tensor cores, not HBM, set the pace: this step is compute-bound.`;

  const tileStory = decode
    ? (R === 1
      ? 'Each weight tile is copied HBM → L2 → shared memory by the TMA engine, multiplied against one row, and thrown away.'
      : `Each weight tile is copied HBM → L2 → shared memory by the TMA engine once, then multiplied against all ${fmtCount(R)} rows before the next tile streams in. That reuse is the whole point of batching.`)
    : `Each weight tile is copied HBM → L2 → shared memory once and reused across all ${fmtCount(R)} prompt rows.`;

  const ffnSteps = (group: string, repeat: number, layerKind: LayerKind, prefix: string) => {
    norm(`${prefix}-norm2`, group, 'Post-attention RMSNorm', repeat, layerKind, 'Normalize again before the feed-forward block.');
    gemm(`${prefix}-gateup`, group, 'FFN gate + up projection (SwiGLU)', 2 * h * m.ffn, h, 2 * m.ffn, repeat, layerKind,
      ({ w, i }) => `The feed-forward block widens each token from ${fmtCount(h)} to ${fmtCount(m.ffn)} twice (gate and up), then applies SiLU(gate) × up in the kernel's epilogue. That is ${fmtBytes(w)} of weights, the largest read in the layer. ${tileStory} ${gemmTail(i)}`);
    gemm(`${prefix}-down`, group, 'FFN down projection + residual', h * m.ffn, m.ffn, h, repeat, layerKind,
      ({ w, i }) => `Project ${fmtCount(m.ffn)} back down to ${fmtCount(h)} and add the result into the residual stream: another ${fmtBytes(w)} of weights. ${gemmTail(i)} After this, the token's vector leaves the layer and the next layer begins.`);
  };

  // 1. Host side -------------------------------------------------------------

  if (decode) {
    add({
      id: 'launch', group: 'Host', title: 'CPU schedules the batch and launches the forward pass',
      kind: 'host', unit: 'cpu', region: null, repeat: 1, bytes: 0, flops: 0, fixedUs: LAUNCH_US,
      explain: `The serving engine's scheduler on the CPU decides which ${seqs === 1 ? 'sequence runs' : `${fmtCount(seqs)} sequences run`} this step, then launches one pre-recorded CUDA graph that enqueues every kernel of the forward pass in a single call. Nothing crosses PCIe on the way in: the token ${seqs === 1 ? 'id' : 'ids'} sampled last step ${seqs === 1 ? 'is' : 'are'} already sitting in GPU memory.`,
    });
  } else {
    const idBytes = R * 8;
    add({
      id: 'tokenize', group: 'Host', title: 'CPU tokenizes the prompt',
      kind: 'host', unit: 'cpu', region: null, repeat: 1, bytes: 0, flops: 0, fixedUs: 20 + R * 0.5,
      explain: `The prompt arrives as text. The tokenizer on the CPU splits it into ${fmtCount(R)} token ids from a vocabulary of ${fmtCount(m.vocab)}. The GPU never sees characters, only these integers.`,
    });
    add({
      id: 'h2d', group: 'Host', title: 'Token ids cross PCIe into GPU memory',
      kind: 'pcie', unit: 'pcie', region: null, repeat: 1, bytes: idBytes, flops: 0, fixedUs: PCIE_LAT_US,
      explain: `${fmtCount(R)} ids × 8 bytes = ${fmtBytes(idBytes)} is copied over PCIe Gen5 (${(gpu.pcieBw / 1e9).toFixed(0)} GB/s). The link is slow next to HBM, but the payload is tiny: the model's weights never cross it during serving, they were loaded once at startup.`,
    });
    add({
      id: 'launch', group: 'Host', title: 'CPU launches the forward pass',
      kind: 'host', unit: 'cpu', region: null, repeat: 1, bytes: 0, flops: 0, fixedUs: LAUNCH_US,
      explain: 'The serving engine enqueues every kernel of the forward pass. From here until the logits come back, the CPU only waits.',
    });
  }

  // 2. Embedding -------------------------------------------------------------

  add({
    id: 'embed', group: 'Embedding', title: 'Embedding lookup',
    kind: 'embed', unit: 'copy', region: 'weights', repeat: 1,
    bytes: 2 * R * h * B, weightBytes: R * h * B, flops: 0,
    explain: `Each token id selects one row of the ${fmtCount(m.vocab)} × ${fmtCount(h)} embedding table (${fmtBytes(m.vocab * h * B)} in HBM). Only ${fmtBytes(R * h * B)} is actually read, one ${fmtCount(h)}-wide row per token. This is a gather, not a matmul, and it is the only weight that does not have to stream in full.`,
  });

  // 3. Layers ----------------------------------------------------------------

  if (m.linear && nLinear > 0) {
    const { qkHeads, vHeads, headDim: dk } = m.linear;
    const g = `Gated DeltaNet layer`;
    const statePerLayer = vHeads * dk * dk * 4;
    norm('dn-norm1', g, 'Input RMSNorm', nLinear, 'linear',
      `This is one of ${nLinear} linear-attention layers. The same seven kernels run in every one of them, one layer after another.`);
    gemm('dn-inproj', g, 'Input projection (q, k, v, gate, β, decay)', lp.inProj, h,
      2 * qkHeads * dk + 2 * vHeads * dk + 2 * vHeads, nLinear, 'linear',
      ({ w, i }) => `One matmul produces everything the DeltaNet needs: ${qkHeads} query and key heads, ${vHeads} value heads (each ${dk} wide), an output gate, and per-head write strength β and decay α. ${fmtBytes(w)} of weights. ${tileStory} ${gemmTail(i)}`);
    const stateBytes = decode
      ? seqs * 2 * statePerLayer + R * (2 * qkHeads * dk + 2 * vHeads * dk) * B
      : statePerLayer + R * (2 * qkHeads * dk + 2 * vHeads * dk) * B;
    const stateFlops = decode
      ? seqs * vHeads * 8 * dk * dk
      : R * vHeads * (4 * dk * dk + 2 * 64 * 2 * dk);
    add({
      id: 'dn-delta', group: g, title: decode ? 'Short conv + gated delta rule (update the state)' : 'Short conv + chunked delta rule (build the state)',
      kind: 'state', unit: decode ? 'cuda' : 'tensor', region: 'state', layerKind: 'linear', repeat: nLinear,
      bytes: stateBytes, flops: stateFlops,
      explain: decode
        ? `Instead of a KV cache, each DeltaNet layer keeps a fixed ${vHeads} × ${dk} × ${dk} matrix per sequence (${fmtBytes(statePerLayer)} in fp32). This kernel reads it from HBM, decays it, writes the new key/value association in with the delta rule, reads out the answer for the query, and writes the state back. The cost depends on the state size, not on context length: at ${fmtCount(L)} tokens of context it is the same ${fmtBytes(seqs * 2 * statePerLayer)} round trip as at 10. It is memory-bound, like all of decode.`
        : `For a prompt, the delta rule is evaluated in chunks of 64 tokens with small matmuls on the tensor cores, leaving one ${vHeads} × ${dk} × ${dk} state (${fmtBytes(statePerLayer)}) written to HBM for decode to pick up.`,
    });
    gemm('dn-out', g, 'Gated RMSNorm + output projection + residual', lp.out, vHeads * dk, h, nLinear, 'linear',
      ({ w, i }) => `The ${vHeads * dk}-wide DeltaNet output is normalized, multiplied by the SiLU of its gate, projected back to ${fmtCount(h)} (${fmtBytes(w)} of weights), and added to the residual stream. ${gemmTail(i)}`);
    ffnSteps(g, nLinear, 'linear', 'dn');
  }

  {
    const g = m.linear ? 'Full-attention layer' : 'Transformer layer';
    const qWidth = qH * d * (outputGate ? 2 : 1);
    norm('fa-norm1', g, 'Input RMSNorm', nFull, 'full',
      m.linear
        ? `Every fourth layer is a full softmax-attention layer: ${nFull} of them. This is the only kind of layer that keeps a KV cache.`
        : `Each of the ${nFull} layers runs the same sequence of kernels.`);
    gemm('fa-qkv', g, 'Q, K, V projection', ap.qkv, h, qWidth + 2 * kvH * d, nFull, 'full',
      ({ w, i }) => `One fused matmul produces ${qH} query heads and ${kvH} key/value heads, each ${d} wide${outputGate ? ', plus a per-head output gate (which is why the query projection is twice as wide)' : ''}. ${fmtBytes(w)} of weights. ${tileStory} ${gemmTail(i)}`);
    const kvTok = 2 * kvH * d * B;
    add({
      id: 'fa-rope', group: g, title: 'QK-norm, RoPE, append K and V to the cache',
      kind: 'vector', unit: 'cuda', region: 'kv', layerKind: 'full', repeat: nFull,
      bytes: 2 * R * (qH + 2 * kvH) * d * B + R * kvTok, flops: 12 * R * (qH + kvH) * d,
      explain: `Each query and key head is RMS-normalized and rotated by its position (RoPE${m.linear ? ', on the first 25% of each head' : ''}). The new key and value vectors, ${fmtBytes(kvTok)} per token for this layer, are then written into the KV cache in HBM so that every later token can attend to this one. ${decode ? `This is the moment the cache grows by one token per sequence.` : `For the prompt, that is ${fmtBytes(R * kvTok)} written per layer.`}`,
    });
    if (decode) {
      const kvRead = seqs * L * kvTok;
      add({
        id: 'fa-attn', group: g, title: 'Attention over the KV cache (flash-decoding)',
        kind: 'attn', unit: 'tensor', region: 'kv', layerKind: 'full', repeat: nFull,
        bytes: kvRead + 2 * seqs * qH * d * B, flops: 4 * seqs * qH * L * d,
        explain: `The new query must be compared with every past key and blend every past value: ${fmtCount(L)} tokens × ${fmtBytes(kvTok)} = ${fmtBytes(L * kvTok)} read per sequence${seqs > 1 ? `, ${fmtBytes(kvRead)} for the batch` : ''}. Unlike the weights, this cache is different for every sequence, so batching does not share it: intensity stays at about ${qH / kvH} FLOPs per byte (the ${qH}:${kvH} grouped-query ratio) at any batch size. Raise the context slider to watch this step take over.`,
      });
    } else {
      const blocks = Math.ceil(R / 128);
      add({
        id: 'fa-attn', group: g, title: 'Causal attention over the prompt (FlashAttention)',
        kind: 'attn', unit: 'tensor', region: 'kv', layerKind: 'full', repeat: nFull,
        bytes: 2 * R * qH * d * B + (blocks * R * kvTok) / 2, flops: 2 * qH * d * R * R,
        explain: `Every prompt token attends to all earlier ones: ${fmtCount(R)}² / 2 query-key pairs per head. FlashAttention tiles Q, K and V through shared memory so the ${fmtCount(R)} × ${fmtCount(R)} score matrix is never written to HBM. The FLOPs grow with the square of the prompt and the bytes only linearly, so long prompts are firmly compute-bound.`,
      });
    }
    gemm('fa-o', g, outputGate ? 'Output gate + O projection + residual' : 'O projection + residual', ap.o, qH * d, h, nFull, 'full',
      ({ w, i }) => `${outputGate ? 'The attention output is multiplied by the sigmoid of its gate, then ' : 'The '}${qH} heads × ${d} are projected back to ${fmtCount(h)} (${fmtBytes(w)} of weights) and added to the residual stream. ${gemmTail(i)}`);
    ffnSteps(g, nFull, 'full', 'fa');
  }

  // 4. Head ------------------------------------------------------------------

  const headRows = decode ? R : 1;
  norm('final-norm', 'Output', 'Final RMSNorm', 1, undefined, 'After the last layer, one more normalization.');
  gemm('lm-head', 'Output', 'LM head: hidden → vocabulary logits', m.vocab * h, h, m.vocab, 1, undefined,
    ({ w, i }) => `The single biggest matmul in the pass: ${fmtCount(h)} → ${fmtCount(m.vocab)} logits, ${fmtBytes(w)} of weights. ${decode ? '' : `In prefill only the last prompt position needs a next-token prediction, so this runs on 1 row while still paying for all ${fmtBytes(w)}. `}${gemmTail(i)}`,
    headRows);
  add({
    id: 'sample', group: 'Output', title: 'Softmax + top-p sampling on the GPU',
    kind: 'sample', unit: 'cuda', region: 'act', repeat: 1,
    bytes: 2 * headRows * m.vocab * 4, flops: 10 * headRows * m.vocab,
    explain: `Temperature, softmax over ${fmtCount(m.vocab)} logits, a top-p cut and a random draw, all on the CUDA cores. The chosen token id is written to GPU memory, where the next forward pass will read it.`,
  });

  // 5. Back to the host -----------------------------------------------------

  add({
    id: 'd2h', group: 'Host', title: 'Token id copied back over PCIe',
    kind: 'pcie', unit: 'pcie', region: null, repeat: 1, bytes: headRows * 8, flops: 0, fixedUs: PCIE_LAT_US,
    explain: `${headRows === 1 ? 'One 8-byte id goes' : `${fmtCount(headRows)} ids (${fmtBytes(headRows * 8)}) go`} back to the CPU. Latency matters here, not bandwidth.`,
  });
  add({
    id: 'stream', group: 'Host', title: 'CPU detokenizes and streams text to the user',
    kind: 'host', unit: 'cpu', region: null, repeat: 1, bytes: 0, flops: 0, fixedUs: 0, overlapped: true,
    explain: `The CPU turns the id back into text and sends it down the connection${decode ? '' : ': this is the first token, and the time so far is the time-to-first-token'}. Engines overlap this with the next forward pass, so it costs no GPU time. Then the loop repeats: the new token becomes the input, the KV cache is one longer, and every weight streams through the chip again.`,
  });

  // Totals -------------------------------------------------------------------

  const totalUs = steps.reduce((s, x) => s + x.totalUs, 0);
  const tp = totalParams(m);
  const weights = tp.total * B;
  const kv = (decode ? seqs * L : R) * kvBytesPerToken(m);
  const state = seqs * stateBytesPerSeq(m);
  const memTotal = weights + kv + state + RUNTIME_RESERVE_BYTES;
  const capacity = gpu.hbmGB * 1e9;
  const memBoundUs = steps.filter((s) => s.bound === 'memory').reduce((a, s) => a + s.totalUs, 0);

  return {
    arch: m,
    gpu,
    opts,
    steps,
    totalUs,
    memory: { weights, kv, state, reserve: RUNTIME_RESERVE_BYTES, total: memTotal, capacity, fits: memTotal <= capacity },
    weightFloorUs: (streamedWeightBytes(m) / gpu.hbmBw) * 1e6,
    computeFloorUs: ((2 * (tp.layers + tp.head) * R) / gpu.tensorFlops) * 1e6,
    memBoundShare: totalUs > 0 ? memBoundUs / totalUs : 0,
    rows: R,
  };
}
