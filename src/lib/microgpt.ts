/**
 * TypeScript port of the inference path of reference/microgpt.py (Andrej Karpathy).
 *
 * The reference file is the "most atomic" GPT: a char-level tokenizer, embedding tables,
 * a forward pass that builds a per-layer KV cache, and autoregressive sampling.
 * This module reproduces that algorithm with plain-number tensors (no autograd is needed
 * for generation) so a UI can step through real prefill + decode.
 */

export const DEFAULT_ALPHABET =
  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?':;\"-()";

export interface MicroGPTConfig {
  nLayer: number;
  nEmbd: number;
  blockSize: number;
  nHead: number;
  std: number;
  rmsEps: number;
  seed: number;
}

export const DEFAULT_CONFIG: MicroGPTConfig = {
  nLayer: 1,
  nEmbd: 16,
  blockSize: 16,
  nHead: 4,
  std: 0.08,
  rmsEps: 1e-5,
  seed: 42,
};

// KV cache: [layer][position][hiddenDim] for keys and values.
export interface KV {
  keys: number[][][];
  values: number[][][];
}

export interface ModelWeights {
  wte: number[][];
  wpe: number[][];
  lmHead: number[][];
  layers: Array<{
    attnWq: number[][];
    attnWk: number[][];
    attnWv: number[][];
    attnWo: number[][];
    mlpFc1: number[][];
    mlpFc2: number[][];
  }>;
}

export interface AttentionHead {
  layer: number;
  head: number;
  // softmax weights over the cached (key) positions, for the current query position
  weights: number[];
}

export interface ForwardResult {
  logits: number[];
  attention: AttentionHead[];
}

/* ------------------------- deterministic RNG ------------------------- */

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller standard normal, drawing two uniforms per call.
function gauss(rng: () => number): number {
  let u = rng();
  if (u === 0) u = rng();
  let v = rng();
  if (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ------------------------- math helpers ------------------------- */

function rmsnorm(x: number[], eps: number): number[] {
  const ms = x.reduce((s, v) => s + v * v, 0) / x.length;
  const scale = Math.pow(ms + eps, -0.5);
  return x.map((xi) => xi * scale);
}

// w: [nout][nin] · x: [nin]  ->  [nout]   (row dot x, like microgpt's `linear`)
function linear(x: number[], w: number[][]): number[] {
  return w.map((row) => row.reduce((s, wi, i) => s + wi * x[i], 0));
}

export function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  let sum = 0;
  const exps = logits.map((l) => {
    const e = Math.exp(l - max);
    sum += e;
    return e;
  });
  return exps.map((e) => e / sum);
}

/* ------------------------- autograd ------------------------- */

// A scalar node in a computation graph, mirroring reference/microgpt.py's `Value`
// class: it tracks a value, a gradient, its children, and the local partial
// derivative of this node w.r.t. each child. `backward()` applies the chain rule.
export class Value {
  data: number;
  grad: number;
  private _children: Value[];
  private _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
  }

  private static coerce(other: number | Value): Value {
    return other instanceof Value ? other : new Value(other);
  }

  add(other: number | Value): Value {
    const o = Value.coerce(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }
  mul(other: number | Value): Value {
    const o = Value.coerce(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }
  pow(other: number): Value {
    return new Value(Math.pow(this.data, other), [this], [other * Math.pow(this.data, other - 1)]);
  }
  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }
  exp(): Value {
    return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
  }
  relu(): Value {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }
  neg(): Value {
    return this.mul(-1);
  }
  sub(o: number | Value): Value {
    return this.add(Value.coerce(o).neg());
  }
  rsub(o: number | Value): Value {
    return Value.coerce(o).add(this.neg());
  }
  div(other: number | Value): Value {
    return this.mul(Value.coerce(other).pow(-1));
  }
  rdiv(other: number | Value): Value {
    return Value.coerce(other).mul(this.pow(-1));
  }

  backward() {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const buildTopo = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v._children) buildTopo(child);
      topo.push(v);
    };
    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) {
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}

// Value-based variants of linear() and softmax(), matching microgpt.py's list-comprehension
// `linear` (sum of wi*xi per row) and stable `softmax` (subtract max, exp, normalize).
function linearV(x: Value[], w: Value[][]): Value[] {
  return w.map((row) => {
    let acc = new Value(0);
    for (let i = 0; i < row.length; i++) acc = acc.add(row[i].mul(x[i]));
    return acc;
  });
}

function softmaxV(logits: Value[]): Value[] {
  let maxData = -Infinity;
  for (const l of logits) if (l.data > maxData) maxData = l.data;
  let sum = new Value(0);
  const exps = logits.map((l) => {
    const e = l.sub(maxData).exp();
    sum = sum.add(e);
    return e;
  });
  return exps.map((e) => e.div(sum));
}

/* ------------------------- the model ------------------------- */

export class MicroGPT {
  readonly config: MicroGPTConfig;
  readonly vocab: string[]; // characters -> token id (0..vocab-1)
  readonly charToId: Map<string, number>;
  readonly BOS: number;
  weights: ModelWeights;

  private rng: () => number;
  private sampleRng: () => number;

  constructor(config: Partial<MicroGPTConfig> = {}, alphabet = DEFAULT_ALPHABET) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.vocab = Array.from(new Set(alphabet)).sort();
    this.charToId = new Map(this.vocab.map((c, i) => [c, i]));
    this.BOS = this.vocab.length;
    this.rng = mulberry32(this.config.seed);
    this.sampleRng = mulberry32(this.config.seed);
    this.weights = this.initWeights();
  }

  private makeMatrix(nout: number, nin: number): number[][] {
    const std = this.config.std;
    return Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => std * gauss(this.rng)),
    );
  }

  private initWeights(): ModelWeights {
    const { nEmbd, nLayer } = this.config;
    const layers = Array.from({ length: nLayer }, () => ({
      attnWq: this.makeMatrix(nEmbd, nEmbd),
      attnWk: this.makeMatrix(nEmbd, nEmbd),
      attnWv: this.makeMatrix(nEmbd, nEmbd),
      attnWo: this.makeMatrix(nEmbd, nEmbd),
      mlpFc1: this.makeMatrix(4 * nEmbd, nEmbd),
      mlpFc2: this.makeMatrix(nEmbd, 4 * nEmbd),
    }));
    return {
      wte: this.makeMatrix(this.vocab.length + 1, nEmbd),
      wpe: this.makeMatrix(this.config.blockSize, nEmbd),
      lmHead: this.makeMatrix(this.vocab.length + 1, nEmbd),
      layers,
    };
  }

  /** Re-seed weights + sampler and hand back an empty KV cache (a clean slate). */
  reset(): KV {
    this.rng = mulberry32(this.config.seed);
    this.sampleRng = mulberry32(this.config.seed);
    this.weights = this.initWeights();
    return this.newKV();
  }

  newKV(): KV {
    const keys = Array.from({ length: this.config.nLayer }, () => [] as number[][]);
    const values = Array.from({ length: this.config.nLayer }, () => [] as number[][]);
    return { keys, values };
  }

  tokenize(text: string): number[] {
    const ids: number[] = [];
    for (const ch of text) {
      const id = this.charToId.get(ch);
      if (id !== undefined) ids.push(id);
    }
    return ids;
  }

  labelForToken(id: number): string {
    if (id === this.BOS) return 'BOS';
    return this.vocab[id] ?? '?';
  }

  paramCount(): number {
    const { nEmbd, nLayer, blockSize } = this.config;
    const V = (this.vocab.length + 1); // chars + BOS
    const perLayer =
      4 * nEmbd * nEmbd + // attention q,k,v,o
      2 * 4 * nEmbd * nEmbd; // mlp fc1 + fc2
    return V * nEmbd + blockSize * nEmbd + V * nEmbd + nLayer * perLayer;
  }

  /**
   * One forward pass: embedding -> rmsnorm -> [attention block (appends K,V to the cache)
   * -> residual] -> [MLP block -> residual] -> lm_head logits. Mirrors microgpt's `gpt()`.
   */
  forward(tokenId: number, posId: number, kv: KV): ForwardResult {
    const { nEmbd, nHead, rmsEps } = this.config;
    const headDim = nEmbd / nHead;

    const tok = this.weights.wte[tokenId];
    const pos = this.weights.wpe[posId];
    let x = tok.map((t, i) => t + pos[i]);
    x = rmsnorm(x, rmsEps);

    const attention: AttentionHead[] = [];

    for (let li = 0; li < this.config.nLayer; li++) {
      const l = this.weights.layers[li];

      // --- Attention block ---
      const xResidual = x;
      const xn = rmsnorm(x, rmsEps);
      const q = linear(xn, l.attnWq);
      const k = linear(xn, l.attnWk);
      const v = linear(xn, l.attnWv);
      // >>> KV cache is created here: every token's K/V projection is stored, keyed by layer
      //     and position, so future tokens never recompute the prefix. <<<
      kv.keys[li].push(k);
      kv.values[li].push(v);

      const xAttn: number[] = [];
      for (let h = 0; h < nHead; h++) {
        const hs = h * headDim;
        const qh = q.slice(hs, hs + headDim);
        const kh = kv.keys[li].map((ki) => ki.slice(hs, hs + headDim));
        const vh = kv.values[li].map((vi) => vi.slice(hs, hs + headDim));

        const attnLogits = kh.map((kt) => {
          let s = 0;
          for (let j = 0; j < headDim; j++) s += qh[j] * kt[j];
          return s / Math.sqrt(headDim);
        });
        const attnWeights = softmax(attnLogits);
        attention.push({ layer: li, head: h, weights: attnWeights });

        const headOut: number[] = [];
        for (let j = 0; j < headDim; j++) {
          let s = 0;
          for (let t = 0; t < vh.length; t++) s += attnWeights[t] * vh[t][j];
          headOut.push(s);
        }
        for (let j = 0; j < headDim; j++) xAttn.push(headOut[j]);
      }

      x = linear(xAttn, l.attnWo).map((a, i) => a + xResidual[i]);

      // --- MLP block ---
      const xResidual2 = x;
      const xn2 = rmsnorm(x, rmsEps);
      const m = linear(xn2, l.mlpFc1).map((mi) => Math.max(0, mi)); // relu
      x = linear(m, l.mlpFc2).map((mi, i) => mi + xResidual2[i]);
    }

    const logits = linear(x, this.weights.lmHead);
    return { logits, attention };
  }

  /**
   * Prefill: run the prompt (already BOS-prefixed) one position at a time through `forward`,
   * building the KV cache for every prompt position and returning the final logits (which
   * predict the first generated token).
   */
  prefill(tokenIds: number[], kv: KV): ForwardResult {
    let last: ForwardResult = { logits: [], attention: [] };
    for (let pos = 0; pos < tokenIds.length; pos++) {
      last = this.forward(tokenIds[pos], pos, kv);
    }
    return last;
  }

  /**
   * Turn raw model logits into a sampled token id. Matches microgpt's inference loop:
   * softmax(logits / temperature) then argmax (greedy) or a weighted random pick.
   */
  sample(logits: number[], temperature: number, greedy = false): number {
    if (greedy) {
      let best = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    const t = Math.max(temperature, 1e-6);
    const probs = softmax(logits.map((l) => l / t));
    let r = this.sampleRng();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r < cum) return i;
    }
    return probs.length - 1;
  }

  /** Probability distribution over the vocabulary for the given logits / temperature. */
  probsFor(logits: number[], temperature: number): number[] {
    return softmax(logits.map((l) => l / Math.max(temperature, 1e-6)));
  }

  /* ------------------------- training ------------------------- */

  // Value-based forward pass (a faithful port of microgpt.py's `gpt()`), operating
  // on a parallel set of trainable params while the plain-number `forward()` stays
  // untouched for fast inference.
  private gptValue(
    tokenId: number,
    posId: number,
    keys: Value[][][],
    values: Value[][][],
    p: TrainParams,
  ): Value[] {
    const { nHead, rmsEps } = this.config;
    const headDim = this.config.nEmbd / this.config.nHead;

    // x = wte[token] + wpe[pos], then rmsnorm
    let x = p.wte[tokenId].map((t, i) => t.add(p.wpe[posId][i]));
    x = x.map((xi) => xi.mul((x.reduce((s, v) => s.add(v.mul(v)), new Value(0)).data / x.length + rmsEps) ** -0.5));

    for (let li = 0; li < this.config.nLayer; li++) {
      const l = p.layers[li];
      // --- Attention block ---
      const xResidual = x;
      const xn = x.map((xi) => xi.mul((x.reduce((s, xi) => s.add(xi.mul(xi)), new Value(0)).data / x.length + rmsEps) ** -0.5));
      const q = linearV(xn, l.attnWq);
      const k = linearV(xn, l.attnWk);
      const v = linearV(xn, l.attnWv);
      keys[li].push(k);
      values[li].push(v);
      const xAttn: Value[] = [];
      for (let h = 0; h < this.config.nHead; h++) {
        const hs = h * headDim;
        const qh = q.slice(hs, hs + headDim);
        const kh = keys[li].map((ki) => ki.slice(hs, hs + headDim));
        const vh = values[li].map((vi) => vi.slice(hs, hs + headDim));
        // attn_logits = sum(qh[j] * kh[t][j]) / sqrt(head_dim)
        const attnLogits = kh.map((kt) => {
          let s = new Value(0);
          for (let j = 0; j < headDim; j++) s = s.add(qh[j].mul(kt[j]));
          return s.div(Math.sqrt(headDim));
        });
        const attnWeights = softmaxV(attnLogits);
        const headOut: Value[] = [];
        for (let j = 0; j < headDim; j++) {
          let s = new Value(0);
          for (let t = 0; t < vh.length; t++) s = s.add(attnWeights[t].mul(vh[t][j]));
          headOut.push(s);
        }
        xAttn.push(...headOut);
      }
      x = linearV(xAttn, l.attnWo).map((a, i) => a.add(xResidual[i]));
      // --- MLP block ---
      const xResidual2 = x;
      const xn2 = x.map((xi) => xi.mul((x.reduce((s, xi) => s.add(xi.mul(xi)), new Value(0)).data / x.length + rmsEps) ** -0.5));
      const m = linearV(xn2, l.mlpFc1).map((mi) => mi.relu());
      x = linearV(m, l.mlpFc2).map((mi, i) => mi.add(xResidual2[i]));
    }
    return linearV(x, p.lmHead);
  }

  /** Train the current weights from random or prior state using the reference Adam loop. */
  async train(
    corpus: string[],
    steps: number,
    opts?: {
      learningRate?: number;
      beta1?: number;
      beta2?: number;
      onStep?: (step: number, loss: number) => void;
      signal?: AbortSignal;
    },
  ): Promise<void> {
    if (corpus.length === 0) return;
    const { learningRate = 0.01, beta1 = 0.85, beta2 = 0.99 } = opts ?? {};

    // Build a parallel set of trainable params from the current (plain-number) weights.
    const P = this.trainingParams();
    const paramList = P.all();

    const mBuff = paramList.map(() => 0);
    const vBuff = paramList.map(() => 0);
    const eps = 1e-8;

    const localDoc = (i: number) => corpus[i % corpus.length];

    // Yield to the browser so the loss chart, the progress label, and Stop actually get
    // painted/handled instead of the whole run executing as one uninterrupted chain.
    // A microtask-only yield (`await Promise.resolve()`) never lets the browser repaint —
    // it just chains more microtasks, starving rendering until the loop finishes. And a
    // *pure* wall-clock budget can also miss every checkpoint: this model is tiny enough
    // that a full 1000-step run can complete in well under one frame budget, so the elapsed
    // time never crosses the threshold and no yield ever happens mid-run. So yield on
    // *either* signal — a step-count cap guarantees regular checkpoints even when compute
    // is too fast for the clock to notice, while the time budget still protects against
    // hammering a slow/large run with pointless yields.
    const frameBudgetMs = 16;
    const maxStepsPerChunk = 5;
    let lastYield = performance.now();
    let stepsSinceYield = 0;

    for (let step = 0; step < steps; step++) {
      const doc = localDoc(step);
      const tokens = [this.BOS, ...this.tokenize(doc), this.BOS];
      const n = Math.min(this.config.blockSize, tokens.length - 1);
      if (n < 1) continue;

      const keys: Value[][][] = Array.from({ length: this.config.nLayer }, () => []);
      const values: Value[][][] = Array.from({ length: this.config.nLayer }, () => []);
      const allLogits: Value[][] = [];
      for (let pos = 0; pos < n; pos++) {
        allLogits.push(this.gptValue(tokens[pos], pos, keys, values, P));
      }
      // loss = (1/n) * sum_t -log(softmax(logits)[target]), mirroring microgpt.py
      let loss = new Value(0);
      for (let pos = 0; pos < n; pos++) {
        const probs = softmaxV(allLogits[pos]);
        const targetId = tokens[pos + 1];
        loss = loss.add(probs[targetId].log().neg().div(n));
      }
      loss.backward();

      // Adam update + linear learning-rate decay
      const lrT = learningRate * (1 - (step + 1) / steps);
      for (let i = 0; i < paramList.length; i++) {
        const p = paramList[i];
        mBuff[i] = beta1 * mBuff[i] + (1 - beta1) * p.grad;
        vBuff[i] = beta2 * vBuff[i] + (1 - beta2) * p.grad * p.grad;
        const mHat = mBuff[i] / (1 - beta1 ** (step + 1));
        const vHat = vBuff[i] / (1 - beta2 ** (step + 1));
        p.data -= lrT * mHat / (Math.sqrt(vHat) + eps);
        p.grad = 0;
      }

      opts?.onStep?.(step + 1, loss.data);
      stepsSinceYield++;

      // Hand control back to the browser (a real macrotask yield) once we've spent a
      // frame's worth of time crunching *or* run a handful of steps, whichever comes
      // first, so the page stays interactive and the chart visibly updates while
      // training runs instead of jumping from empty to fully-drawn at the very end.
      const now = performance.now();
      if (now - lastYield >= frameBudgetMs || stepsSinceYield >= maxStepsPerChunk) {
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
        lastYield = performance.now();
        stepsSinceYield = 0;
      }
      if (opts?.signal?.aborted) break;
    }

    // Copy trained numbers back into the plain-number weights used by inference, even on
    // an early stop, so a cancelled run keeps whatever progress it made.
    this.copyTrainingBack(P);
  }

  private trainingParams(): TrainParams {
    const mv = (mat: number[][]) => mat.map((row) => row.map((d) => new Value(d)));
    return new TrainParams({
      wte: mv(this.weights.wte),
      wpe: mv(this.weights.wpe),
      lmHead: mv(this.weights.lmHead),
      layers: this.weights.layers.map((l) => ({
        attnWq: mv(l.attnWq),
        attnWk: mv(l.attnWk),
        attnWv: mv(l.attnWv),
        attnWo: mv(l.attnWo),
        mlpFc1: mv(l.mlpFc1),
        mlpFc2: mv(l.mlpFc2),
      })),
    });
  }

  private copyFrom(mat: Value[][], target: number[][]) {
    for (let i = 0; i < mat.length; i++) for (let j = 0; j < mat[i].length; j++) target[i][j] = mat[i][j].data;
  }
  private copyTrainingBack(P: TrainParams) {
    this.copyFrom(P.wte, this.weights.wte);
    this.copyFrom(P.wpe, this.weights.wpe);
    this.copyFrom(P.lmHead, this.weights.lmHead);
    for (let li = 0; li < this.config.nLayer; li++) {
      const src = P.layers[li];
      const dst = this.weights.layers[li];
      this.copyFrom(src.attnWq, dst.attnWq);
      this.copyFrom(src.attnWk, dst.attnWk);
      this.copyFrom(src.attnWv, dst.attnWv);
      this.copyFrom(src.attnWo, dst.attnWo);
      this.copyFrom(src.mlpFc1, dst.mlpFc1);
      this.copyFrom(src.mlpFc2, dst.mlpFc2);
    }
  }

  /* ------------------------- batched inference ------------------------- */

  /**
   * Sample `count` fresh sequences from the (optionally trained) model, mirroring the
   * reference inference loop: BOS-prefixed prefill + autregression until BOS/block size.
   */
  generate(count: number, temperature = 0.5): string[] {
    const out: string[] = [];
    for (let s = 0; s < count; s++) {
      const kv = this.newKV();
      this.sampleRng = mulberry32(this.config.seed + s + 1);
      const acc: number[] = [];
      let tokenId = this.BOS;
      for (let pos = 0; pos < this.config.blockSize; pos++) {
        const res = this.forward(tokenId, pos, kv);
        tokenId = this.sample(res.logits, temperature, false);
        if (tokenId === this.BOS) break;
        // otherwise tokenId is a real vocab char (< vocab.length by construction)
        acc.push(tokenId);
      }
      out.push(acc.map((id) => this.vocab[id]).join(''));
    }
    return out;
  }
}

// A structural mirror of ModelWeights where every scalar is a trainable `Value`.
class TrainParams {
  wte: Value[][];
  wpe: Value[][];
  lmHead: Value[][];
  layers: Array<{
    attnWq: Value[][];
    attnWk: Value[][];
    attnWv: Value[][];
    attnWo: Value[][];
    mlpFc1: Value[][];
    mlpFc2: Value[][];
  }>;

  constructor(t: {
    wte: Value[][]; wpe: Value[][]; lmHead: Value[][];
    layers: Array<{
      attnWq: Value[][]; attnWk: Value[][]; attnWv: Value[][];
      attnWo: Value[][]; mlpFc1: Value[][]; mlpFc2: Value[][];
    }>;
  }) {
    this.wte = t.wte;
    this.wpe = t.wpe;
    this.lmHead = t.lmHead;
    this.layers = t.layers;
  }

  /** Flatten every trainable scalar into a single list (the optimizer's parameter vector). */
  all(): Value[] {
    const out: Value[] = [];
    const push = (mat: Value[][]) => {
      for (const row of mat) for (const cell of row) out.push(cell);
    };
    push(this.wte);
    push(this.wpe);
    push(this.lmHead);
    for (const l of this.layers) {
      push(l.attnWq);
      push(l.attnWk);
      push(l.attnWv);
      push(l.attnWo);
      push(l.mlpFc1);
      push(l.mlpFc2);
    }
    return out;
  }
}
