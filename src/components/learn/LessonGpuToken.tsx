import { Fragment, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import KpiCard from '../ui/KpiCard';
import SegmentedControl from '../ui/SegmentedControl';
import SliderControl from '../ui/SliderControl';
import { ConceptTag } from '../ui/ConceptTag';
import Figure from '../shell/Figure';
import GoDeeper from '../shell/GoDeeper';
import SectionBody from '../shell/SectionBody';
import Checkpoint from './Checkpoint';
import { cn } from '../../lib/utils';
import { useStill } from '../../lib/prefs';
import { SERIES, NEUTRAL, COMPUTE, MEMORY, INK, SURFACE, tint } from '../../lib/theme';
import {
  H100, MODEL_ARCHS, buildTrace, countLayers, kvBytesPerToken, stateBytesPerSeq, streamedWeightBytes,
  totalParams, tensorRidge, fmtBytes, fmtFlops, fmtUs, fmtCount, MEM_EFF, KERNEL_GAP_US,
  type ModelId, type Phase, type Trace, type TraceStep, type StepKind,
} from '../../lib/gpuTokenTrace';

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

const KIND_COLOR: Record<StepKind, string> = {
  gemm: SERIES.azure,
  attn: MEMORY,
  state: SERIES.violet,
  vector: SERIES.amber,
  sample: SERIES.amber,
  embed: SERIES.cyan,
  host: NEUTRAL[500],
  pcie: NEUTRAL[500],
};

const KIND_LABEL: Record<StepKind, string> = {
  gemm: 'Weight matmul',
  attn: 'Attention / KV cache',
  state: 'DeltaNet state',
  vector: 'Elementwise (norm, RoPE)',
  sample: 'Sampling',
  embed: 'Embedding gather',
  host: 'CPU',
  pcie: 'PCIe copy',
};

const MEM_COLOR = {
  weights: SERIES.azure,
  kv: MEMORY,
  state: SERIES.violet,
  reserve: NEUTRAL[400],
};

const LAYER_COLOR = { linear: SERIES.violet, full: SERIES.azure } as const;

const OUTPUT_TOKENS = ['The', ' H100', ' spends', ' most', ' of', ' each', ' token', ' waiting', ' for', ' HBM', '.'];

// ---------------------------------------------------------------------------
// Die map geometry (viewBox 960 x 600)
// ---------------------------------------------------------------------------

const VB_W = 960;
const VB_H = 600;
const STACK_W = 100;
const STACK_H = 140;
const STACK_YS = [130, 280, 430];
const LEFT_X = 24;
const RIGHT_X = 836;
const DIE = { x: 145, y: 125, w: 670, h: 455 };
const GRID = { x: 160, y: 188, cols: 12, rows: 11, cell: 24, gapX: 2, gapY: 5 };
const ZOOM = { x: 490, y: 185, w: 315, h: 385 };
const TC_XS = [536, 608, 680, 752];
const L2_Y = 152;
const PCIE_X = 210;

interface Stack { x: number; y: number; active: boolean; side: 'left' | 'right' }
const STACKS: Stack[] = [
  ...STACK_YS.map((y) => ({ x: LEFT_X, y, active: true, side: 'left' as const })),
  ...STACK_YS.map((y, i) => ({ x: RIGHT_X, y, active: i < 2, side: 'right' as const })),
];

function smCenter(i: number) {
  const c = i % GRID.cols;
  const r = Math.floor(i / GRID.cols);
  return {
    x: GRID.x + c * (GRID.cell + GRID.gapX) + GRID.cell / 2,
    y: GRID.y + r * (GRID.cell + GRID.gapY) + GRID.cell / 2,
  };
}

/** Path from an HBM stack, through the memory controller and L2, to an SM. */
function stackToSm(s: Stack, sm: number) {
  const cy = s.y + STACK_H / 2;
  const { x, y } = smCenter(sm);
  return s.side === 'left'
    ? `M ${s.x + STACK_W / 2} ${cy} H 152 V ${L2_Y} H ${x} V ${y}`
    : `M ${s.x + STACK_W / 2} ${cy} H 808 V ${L2_Y} H ${x} V ${y}`;
}

const packetRadius = (bytes: number) => {
  const r = 2 + ((Math.log10(Math.max(bytes, 1e3)) - 3) / 6) * 4.5;
  return Math.max(2, Math.min(6.5, r));
};

// ---------------------------------------------------------------------------
// Small pieces
// ---------------------------------------------------------------------------

interface PacketProps { path: string; color: string; r: number; dur: number; begin: number; reverse?: boolean }

/** One dot travelling along an SVG path, restarted whenever its key changes. */
function Packet({ path, color, r, dur, begin, reverse }: PacketProps) {
  return (
    <circle r={r} fill={color} opacity={0.9}>
      <animateMotion
        dur={`${dur}s`}
        begin={`${begin}s`}
        repeatCount="indefinite"
        path={path}
        keyPoints={reverse ? '1;0' : '0;1'}
        keyTimes="0;1"
        calcMode="linear"
        fill="freeze"
      />
    </circle>
  );
}

function Box({ x, y, w, h, label, sub, lit, color = SERIES.azure, rx = 8, dashed }: {
  x: number; y: number; w: number; h: number; label?: string; sub?: string; lit?: boolean; color?: string; rx?: number; dashed?: boolean;
}) {
  return (
    <g>
      <rect
        x={x} y={y} width={w} height={h} rx={rx}
        fill={lit ? tint(color, 16) : SURFACE}
        stroke={lit ? color : NEUTRAL[300]}
        strokeWidth={lit ? 1.8 : 1}
        strokeDasharray={dashed ? '4 3' : undefined}
        style={{ transition: 'fill 300ms, stroke 300ms' }}
      />
      {label && (
        <text x={x + 10} y={y + 18} fontSize={12.5} fontWeight={600} fill={INK}>{label}</text>
      )}
      {sub && (
        <text x={x + 10} y={y + 34} fontSize={10.5} fill={NEUTRAL[500]}>{sub}</text>
      )}
    </g>
  );
}

// ---------------------------------------------------------------------------
// The die map
// ---------------------------------------------------------------------------

function DieMap({ trace, step, tokensOut, still }: { trace: Trace; step: TraceStep; tokensOut: string; still: boolean }) {
  const color = KIND_COLOR[step.kind];
  const onGpu = step.unit === 'tensor' || step.unit === 'cuda' || step.unit === 'copy';
  const usesTensor = step.unit === 'tensor';
  const usesCuda = step.unit === 'cuda';
  const smColor = usesTensor ? COMPUTE : SERIES.amber;
  const busy = step.unitBusy;
  const { memory } = trace;
  const cap = memory.capacity;
  const k = step.id; // restart animations on step change

  // HBM fill fractions (memory is striped evenly over the five active stacks)
  const segs = [
    { key: 'weights', v: memory.weights, c: MEM_COLOR.weights },
    { key: 'kv', v: memory.kv, c: MEM_COLOR.kv },
    { key: 'state', v: memory.state, c: MEM_COLOR.state },
    { key: 'reserve', v: memory.reserve, c: MEM_COLOR.reserve },
  ];
  const scale = Math.min(1, cap / Math.max(memory.total, 1));

  // Packets
  const packets: PacketProps[] = [];
  const dur = 1.5;
  const r = packetRadius(step.bytes / Math.max(1, H100.hbmStacksActive));
  const activeStacks = STACKS.filter((s) => s.active);
  if (!still) {
    if (onGpu) {
      const many = step.kind === 'gemm' || step.kind === 'attn' || step.kind === 'state';
      const targets = step.kind === 'embed' ? [5, 30] : many ? [3, 17, 40, 58, 77, 95, 110, 126] : [8, 50, 100];
      activeStacks.forEach((s, si) => {
        targets.forEach((sm, ti) => {
          if ((si + ti) % (many ? 1 : 2) !== 0) return;
          packets.push({ path: stackToSm(s, sm), color, r, dur, begin: ((si * 0.13 + ti * 0.19) % dur) });
        });
      });
      // Writes back to HBM (KV append, state update)
      if (step.region === 'kv' || step.kind === 'state' || step.id.endsWith('rope')) {
        activeStacks.forEach((s, si) => {
          packets.push({ path: stackToSm(s, 64 + si * 7), color, r: Math.max(2, r - 1), dur, begin: 0.4 + si * 0.21, reverse: true });
        });
      }
      // Inside the zoomed SM
      const smem = { x: ZOOM.x + 155, y: ZOOM.y + 45 };
      if (usesTensor) {
        TC_XS.forEach((tx, i) => packets.push({
          path: `M ${smem.x} ${smem.y} V ${ZOOM.y + 97} H ${tx} V ${ZOOM.y + 161}`,
          color, r: 3, dur: 0.9, begin: i * 0.2,
        }));
      } else {
        [0, 0.3, 0.6].forEach((b) => packets.push({
          path: `M ${smem.x} ${smem.y} V ${ZOOM.y + 97} V ${ZOOM.y + 245}`,
          color, r: 3, dur: 0.9, begin: b,
        }));
      }
    } else if (step.kind === 'pcie') {
      const down = step.id === 'h2d';
      [0, 0.5, 1.0].forEach((b) => packets.push({
        path: `M ${PCIE_X} 72 V ${L2_Y} H 320`, color: SERIES.cyan, r: 4, dur: 1.5, begin: b, reverse: !down,
      }));
    } else if (step.id === 'tokenize') {
      [0, 0.5, 1.0].forEach((b) => packets.push({ path: 'M 95 41 H 290', color: SERIES.cyan, r: 4, dur: 1.5, begin: b }));
    } else if (step.id === 'launch') {
      [0, 0.4, 0.8].forEach((b) => packets.push({ path: `M ${PCIE_X} 60 V ${L2_Y} H 470`, color: NEUTRAL[500], r: 3, dur: 1.2, begin: b }));
    } else if (step.id === 'stream') {
      [0, 0.5, 1.0].forEach((b) => packets.push({ path: 'M 300 41 H 560', color: SERIES.cyan, r: 4, dur: 1.5, begin: b }));
    }
  }

  const hostLit = step.unit === 'cpu';
  const pcieLit = step.unit === 'pcie' || step.id === 'launch';
  const regionLit = (key: string) =>
    (step.region === 'weights' && key === 'weights') ||
    (step.region === 'kv' && key === 'kv') ||
    (step.region === 'state' && key === 'state');

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      role="img"
      aria-label={`H100 die map. Current step: ${step.title}`}
      className="w-full h-auto select-none"
      style={{ minWidth: 640 }}
    >
      {/* ---------- Host row ---------- */}
      <Box x={20} y={10} w={150} h={62} label="Host DRAM" sub="request text, output" lit={step.id === 'tokenize'} color={SERIES.cyan} />
      <Box x={190} y={10} w={210} h={62} label="Host CPU" sub="tokenizer · scheduler · detok" lit={hostLit} color={NEUTRAL[600]} />
      <g>
        <rect x={420} y={10} width={520} height={62} rx={8} fill={SURFACE} stroke={step.id === 'stream' ? SERIES.cyan : NEUTRAL[300]} strokeWidth={step.id === 'stream' ? 1.8 : 1} />
        <text x={432} y={28} fontSize={10.5} fill={NEUTRAL[500]}>Streamed to the user</text>
        <text x={432} y={54} fontSize={15} fontWeight={500} fill={INK} fontFamily="var(--font-mono, monospace)">
          {tokensOut || '…'}
          <tspan fill={SERIES.azure}>▍</tspan>
        </text>
      </g>

      {/* PCIe */}
      <line x1={PCIE_X} y1={72} x2={PCIE_X} y2={DIE.y + 10} stroke={pcieLit ? SERIES.cyan : NEUTRAL[300]} strokeWidth={pcieLit ? 3 : 2} />
      <text x={PCIE_X + 8} y={92} fontSize={10.5} fill={NEUTRAL[500]}>PCIe Gen5 · {(H100.pcieBw / 1e9).toFixed(0)} GB/s</text>

      {/* ---------- Package ---------- */}
      <rect x={10} y={100} width={940} height={492} rx={14} fill="none" stroke={NEUTRAL[300]} strokeDasharray="6 4" />
      <text x={938} y={118} fontSize={11} textAnchor="end" fill={NEUTRAL[500]}>H100 SXM5 package</text>

      {/* HBM stacks */}
      {STACKS.map((s, i) => {
        const inner = { x: s.x + 14, y: s.y + 40, w: STACK_W - 28, h: STACK_H - 52 };
        let acc = 0;
        return (
          <g key={i}>
            <rect x={s.x} y={s.y} width={STACK_W} height={STACK_H} rx={8}
              fill={SURFACE} stroke={s.active ? (onGpu && step.region && step.region !== 'act' ? color : NEUTRAL[400]) : NEUTRAL[300]}
              strokeWidth={s.active && onGpu && step.region && step.region !== 'act' ? 1.8 : 1}
              strokeDasharray={s.active ? undefined : '4 3'} />
            <text x={s.x + 10} y={s.y + 17} fontSize={12} fontWeight={600} fill={s.active ? INK : NEUTRAL[400]}>HBM3</text>
            <text x={s.x + 10} y={s.y + 32} fontSize={10} fill={NEUTRAL[500]}>{s.active ? '16 GB' : 'disabled'}</text>
            {s.active ? (
              <g>
                <rect x={inner.x} y={inner.y} width={inner.w} height={inner.h} rx={3} fill="none" stroke={NEUTRAL[300]} strokeDasharray="2 2" />
                {segs.map((sg) => {
                  const hgt = (sg.v / cap) * scale * inner.h;
                  const y = inner.y + inner.h - acc - hgt;
                  acc += hgt;
                  if (hgt < 0.4) return null;
                  const lit = regionLit(sg.key);
                  return (
                    <rect key={sg.key} x={inner.x} y={y} width={inner.w} height={hgt}
                      fill={sg.c} opacity={lit ? 0.95 : 0.45}
                      stroke={lit ? INK : 'none'} strokeWidth={lit ? 1 : 0}
                      style={{ transition: 'opacity 300ms' }} />
                  );
                })}
                {!memory.fits && (
                  <g>
                    <rect x={inner.x + 8} y={inner.y + 4} width={inner.w - 16} height={15} rx={3} fill={SURFACE} />
                    <text x={inner.x + inner.w / 2} y={inner.y + 15} fontSize={10} fontWeight={700} textAnchor="middle" fill={MEMORY}>FULL</text>
                  </g>
                )}
              </g>
            ) : (
              <text x={s.x + 10} y={s.y + 80} fontSize={9.5} fill={NEUTRAL[400]}>
                <tspan x={s.x + 10} dy={0}>6th stack is</tspan>
                <tspan x={s.x + 10} dy={13}>fused off for</tspan>
                <tspan x={s.x + 10} dy={13}>yield</tspan>
              </text>
            )}
            {s.active && (
              <line
                x1={s.side === 'left' ? s.x + STACK_W : s.x} y1={s.y + STACK_H / 2}
                x2={s.side === 'left' ? DIE.x : DIE.x + DIE.w} y2={s.y + STACK_H / 2}
                stroke={onGpu && step.region ? color : NEUTRAL[300]} strokeWidth={onGpu && step.region ? 3 : 2}
              />
            )}
          </g>
        );
      })}
      <text x={LEFT_X} y={588} fontSize={10.5} fill={NEUTRAL[500]}>
        5 × 16 GB · {(H100.hbmBw / 1e12).toFixed(2)} TB/s total
      </text>

      {/* Die */}
      <rect x={DIE.x} y={DIE.y} width={DIE.w} height={DIE.h} rx={10} fill={tint(NEUTRAL[200], 35)} stroke={NEUTRAL[400]} />
      {/* L2 + memory-controller bus */}
      <path d={`M 152 ${STACK_YS[2] + STACK_H / 2} V ${L2_Y}`} stroke={NEUTRAL[300]} strokeWidth={2} fill="none" />
      <path d={`M 808 ${STACK_YS[1] + STACK_H / 2} V ${L2_Y}`} stroke={NEUTRAL[300]} strokeWidth={2} fill="none" />
      {[0, 1].map((p) => (
        <rect key={p} x={160 + p * 325} y={135} width={315} height={34} rx={6}
          fill={onGpu && step.region ? tint(color, 14) : SURFACE}
          stroke={onGpu && step.region ? color : NEUTRAL[300]} />
      ))}
      <text x={170} y={157} fontSize={12} fontWeight={600} fill={INK}>L2 cache · 50 MB (two 25 MB partitions)</text>
      <text x={795} y={157} fontSize={10.5} textAnchor="end" fill={NEUTRAL[500]}>every HBM byte passes through here</text>

      {/* SM grid */}
      {Array.from({ length: H100.sms }, (_, i) => {
        const c = i % GRID.cols;
        const rr = Math.floor(i / GRID.cols);
        const x = GRID.x + c * (GRID.cell + GRID.gapX);
        const y = GRID.y + rr * (GRID.cell + GRID.gapY);
        const active = onGpu && (step.kind !== 'embed' || i < 24);
        const fillOp = active ? 0.07 + 0.93 * Math.sqrt(busy) : 0;
        return (
          <g key={i}>
            <rect x={x} y={y} width={GRID.cell} height={GRID.cell} rx={3}
              fill={SURFACE} stroke={active ? smColor : NEUTRAL[300]} strokeWidth={active ? 1.2 : 0.8} />
            {active && (
              <rect x={x} y={y + GRID.cell * (1 - Math.max(0.06, Math.sqrt(busy)))} width={GRID.cell}
                height={GRID.cell * Math.max(0.06, Math.sqrt(busy))} rx={2} fill={smColor} opacity={Math.min(1, fillOp + 0.2)} />
            )}
          </g>
        );
      })}
      <text x={GRID.x} y={GRID.y + GRID.rows * (GRID.cell + GRID.gapY) + 14} fontSize={11} fill={NEUTRAL[600]}>
        132 SMs · fill = share of {usesTensor ? 'tensor-core' : 'CUDA-core'} peak in use
      </text>

      {/* Zoom connector */}
      <path d={`M ${smCenter(11).x + 12} ${smCenter(11).y} L ${ZOOM.x} ${ZOOM.y + 10}`} stroke={NEUTRAL[400]} strokeDasharray="3 3" fill="none" />

      {/* Zoomed SM */}
      <rect x={ZOOM.x} y={ZOOM.y} width={ZOOM.w} height={ZOOM.h} rx={10} fill={SURFACE} stroke={NEUTRAL[400]} />
      <text x={ZOOM.x + 12} y={ZOOM.y + 18} fontSize={12} fontWeight={700} fill={INK}>Inside one SM</text>
      <Box x={ZOOM.x + 15} y={ZOOM.y + 28} w={285} h={36} lit={onGpu} color={color} rx={6} />
      <text x={ZOOM.x + 25} y={ZOOM.y + 50} fontSize={11.5} fontWeight={600} fill={INK}>Shared memory / L1 · {H100.smemKB} KB</text>
      <rect x={ZOOM.x + 245} y={ZOOM.y + 35} width={46} height={22} rx={4}
        fill={onGpu && (step.kind === 'gemm' || step.kind === 'attn') ? color : NEUTRAL[200]} opacity={0.9} />
      <text x={ZOOM.x + 268} y={ZOOM.y + 50} fontSize={10.5} fontWeight={700} textAnchor="middle"
        fill={onGpu && (step.kind === 'gemm' || step.kind === 'attn') ? SURFACE : NEUTRAL[600]}>TMA</text>

      <Box x={ZOOM.x + 15} y={ZOOM.y + 78} w={285} h={32} lit={onGpu} color={NEUTRAL[600]} rx={6} />
      <text x={ZOOM.x + 25} y={ZOOM.y + 99} fontSize={11.5} fontWeight={600} fill={INK}>Register file · {H100.regKB} KB</text>

      {TC_XS.map((tx, i) => {
        const x = tx - 31;
        const y = ZOOM.y + 130;
        const lit = onGpu && usesTensor;
        return (
          <g key={i}>
            <rect x={x} y={y} width={62} height={62} rx={6}
              fill={lit ? tint(COMPUTE, 10 + 80 * Math.sqrt(busy)) : SURFACE}
              stroke={lit ? COMPUTE : NEUTRAL[300]} strokeWidth={lit ? 1.6 : 1} />
            {Array.from({ length: 16 }, (_, j) => (
              <rect key={j} x={x + 7 + (j % 4) * 12.5} y={y + 7 + Math.floor(j / 4) * 12.5} width={10} height={10} rx={1.5}
                fill={lit ? COMPUTE : NEUTRAL[200]} opacity={lit ? 0.25 + 0.75 * Math.sqrt(busy) : 1} />
            ))}
          </g>
        );
      })}
      <text x={ZOOM.x + 15} y={ZOOM.y + 210} fontSize={11} fill={NEUTRAL[600]}>4 tensor cores · matrix multiply (BF16)</text>

      <Box x={ZOOM.x + 15} y={ZOOM.y + 222} w={285} h={46} lit={onGpu && usesCuda} color={SERIES.amber} rx={6} />
      {Array.from({ length: 32 }, (_, j) => (
        <rect key={j} x={ZOOM.x + 24 + j * 8.6} y={ZOOM.y + 248} width={6} height={12} rx={1}
          fill={onGpu && usesCuda ? SERIES.amber : NEUTRAL[200]} opacity={onGpu && usesCuda ? 0.35 + 0.65 * Math.sqrt(busy) : 1} />
      ))}
      <text x={ZOOM.x + 25} y={ZOOM.y + 242} fontSize={11.5} fontWeight={600} fill={INK}>128 CUDA cores · norms, softmax, SiLU</text>

      <Box x={ZOOM.x + 15} y={ZOOM.y + 282} w={285} h={28} rx={6} />
      <text x={ZOOM.x + 25} y={ZOOM.y + 301} fontSize={11} fill={NEUTRAL[600]}>4 warp schedulers · issue one instruction / clock each</text>

      <text x={ZOOM.x + 15} y={ZOOM.y + 336} fontSize={11} fill={NEUTRAL[600]}>
        {onGpu
          ? `HBM in use: ${(step.hbmBusy * 100).toFixed(0)}% · ${usesTensor ? 'tensor' : 'CUDA'} cores in use: ${(busy * 100).toFixed(busy < 0.01 ? 2 : 1)}%`
          : 'GPU idle while the host works'}
      </text>
      <text x={ZOOM.x + 15} y={ZOOM.y + 356} fontSize={10.5} fill={NEUTRAL[500]}>× 132 SMs, all running this kernel at once</text>

      {/* Packets drawn last so they sit on top */}
      <g key={k}>
        {packets.map((p, i) => (
          <Fragment key={i}><Packet {...p} /></Fragment>
        ))}
      </g>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Layer strip
// ---------------------------------------------------------------------------

function LayerStrip({ trace, step, sweep }: { trace: Trace; step: TraceStep; sweep: number }) {
  const types = trace.arch.layerTypes;
  let seen = 0;
  const total = step.layerKind ? types.filter((t) => t === step.layerKind).length : 0;
  const done = Math.floor(sweep * total);
  return (
    <div>
      <div className="flex gap-[2px] h-4" aria-hidden>
        {types.map((t, i) => {
          const inGroup = step.layerKind === t;
          const n = inGroup ? seen++ : -1;
          const lit = inGroup && n <= done;
          return (
            <div
              key={i}
              className="flex-1 rounded-[2px] transition-opacity"
              style={{
                background: LAYER_COLOR[t],
                opacity: step.layerKind ? (lit ? 1 : inGroup ? 0.45 : 0.12) : 0.25,
                outline: inGroup && n === done ? `2px solid ${INK}` : undefined,
              }}
            />
          );
        })}
      </div>
      <div className="flex flex-wrap justify-between gap-2 mt-1.5 text-[0.6875rem] text-slate-500">
        <span>
          {trace.arch.linear ? (
            <>
              <span className="inline-block w-2.5 h-2.5 rounded-sm mr-1 align-middle" style={{ background: LAYER_COLOR.linear }} />
              Gated DeltaNet ×{countLayers(trace.arch).linear}
              <span className="inline-block w-2.5 h-2.5 rounded-sm ml-3 mr-1 align-middle" style={{ background: LAYER_COLOR.full }} />
              Full attention ×{countLayers(trace.arch).full}
            </>
          ) : (
            <>
              <span className="inline-block w-2.5 h-2.5 rounded-sm mr-1 align-middle" style={{ background: LAYER_COLOR.full }} />
              Transformer layer ×{trace.arch.layers}
            </>
          )}
        </span>
        <span>
          {step.layerKind
            ? `This kernel runs once in each of the ${total} highlighted layers, in order`
            : 'This step runs once per forward pass'}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

function Timeline({ trace, idx, onPick }: { trace: Trace; idx: number; onPick: (i: number) => void }) {
  const visible = trace.steps.map((s, i) => ({ s, i })).filter(({ s }) => s.totalUs > 0);
  const byKind = new Map<StepKind, number>();
  trace.steps.forEach((s) => byKind.set(s.kind, (byKind.get(s.kind) ?? 0) + s.totalUs));
  const kinds = [...byKind.entries()].filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  return (
    <div>
      <div className="flex h-9 w-full rounded-lg overflow-hidden border border-slate-200" role="list">
        {visible.map(({ s, i }) => (
          <button
            key={s.id}
            type="button"
            role="listitem"
            title={`${s.title}: ${fmtUs(s.totalUs)}${s.repeat > 1 ? ` (${s.repeat} × ${fmtUs(s.timeUs)})` : ''}`}
            onClick={() => onPick(i)}
            className="h-full relative transition-[filter] hover:brightness-110"
            style={{
              width: `${(s.totalUs / trace.totalUs) * 100}%`,
              minWidth: 3,
              background: KIND_COLOR[s.kind],
              opacity: i === idx ? 1 : 0.55,
              boxShadow: i === idx ? `inset 0 0 0 2px ${INK}` : `inset -1px 0 0 0 ${SURFACE}`,
            }}
          />
        ))}
      </div>
      <div className="flex justify-between text-[0.6875rem] text-slate-400 mt-1 font-mono">
        <span>0</span>
        <span>{fmtUs(trace.totalUs)}</span>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-slate-600">
        {kinds.map(([kind, us]) => (
          <span key={kind} className="inline-flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: KIND_COLOR[kind] }} />
            {KIND_LABEL[kind]} <span className="font-mono text-slate-500">{((us / trace.totalUs) * 100).toFixed(0)}%</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step detail panel
// ---------------------------------------------------------------------------

function Gauge({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-[0.6875rem] text-slate-500">
        <span>{label}</span>
        <span className="font-mono">{(value * 100).toFixed(value < 0.01 && value > 0 ? 2 : 0)}%</span>
      </div>
      <div className="h-2 rounded-full bg-slate-100 overflow-hidden mt-1">
        <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${Math.max(value > 0 ? 1 : 0, value * 100)}%`, background: color }} />
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 px-3 py-2">
      <div className="text-[0.625rem] uppercase tracking-wider text-slate-400">{label}</div>
      <div className="font-mono font-semibold text-slate-800 text-sm mt-0.5">{value}</div>
      {sub && <div className="text-[0.6875rem] text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

const BOUND_STYLE: Record<TraceStep['bound'], { label: string; cls: string }> = {
  memory: { label: 'Memory-bound', cls: 'bg-rose-100 text-rose-700' },
  compute: { label: 'Compute-bound', cls: 'bg-emerald-100 text-emerald-700' },
  latency: { label: 'Latency-bound', cls: 'bg-slate-100 text-slate-600' },
  overlapped: { label: 'Overlapped', cls: 'bg-slate-100 text-slate-600' },
};

function StepPanel({ trace, step }: { trace: Trace; step: TraceStep }) {
  const onGpu = step.unit === 'tensor' || step.unit === 'cuda' || step.unit === 'copy';
  const share = trace.totalUs > 0 ? step.totalUs / trace.totalUs : 0;
  const b = BOUND_STYLE[step.bound];
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 text-[0.6875rem] uppercase tracking-wider text-slate-500">
          <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: KIND_COLOR[step.kind] }} />
          {step.group}
        </span>
        {step.repeat > 1 && (
          <span className="glass-chip px-2 py-0.5 text-[0.6875rem] font-semibold text-slate-600">× {step.repeat} layers</span>
        )}
        <span className={cn('px-2 py-0.5 rounded-md text-[0.6875rem] font-semibold', b.cls)}>{b.label}</span>
      </div>
      <h4 className="text-lg font-bold text-slate-800 leading-snug">{step.title}</h4>
      <p className="text-sm leading-relaxed text-slate-600">{step.explain}</p>

      <div className="grid grid-cols-2 gap-2">
        <Stat
          label={step.unit === 'pcie' ? 'Bytes over PCIe' : onGpu ? 'Bytes from HBM' : 'Bytes moved'}
          value={step.bytes > 0 ? fmtBytes(step.bytes) : '—'}
          sub={step.repeat > 1 && step.bytes > 0 ? `${fmtBytes(step.bytes * step.repeat)} over all layers` : undefined}
        />
        <Stat
          label="FLOPs"
          value={step.flops > 0 ? fmtFlops(step.flops) : '—'}
          sub={step.repeat > 1 && step.flops > 0 ? `${fmtFlops(step.flops * step.repeat)} over all layers` : undefined}
        />
        <Stat
          label="Intensity"
          value={onGpu && step.bytes > 0 ? `${step.intensity < 10 ? step.intensity.toFixed(2) : step.intensity.toFixed(0)} FLOP/B` : '—'}
          sub={onGpu && step.flops > 0 ? `${step.unit === 'tensor' ? 'tensor' : 'CUDA-core'} ridge ≈ ${step.ridge.toFixed(0)}` : undefined}
        />
        <Stat
          label="Time"
          value={step.repeat > 1 ? fmtUs(step.totalUs) : fmtUs(step.timeUs)}
          sub={step.repeat > 1
            ? `${fmtUs(step.timeUs)} × ${step.repeat} · ${(share * 100).toFixed(1)}% of pass`
            : `${(share * 100).toFixed(1)}% of pass`}
        />
      </div>
      {onGpu && (
        <div className="space-y-2 pt-1">
          <Gauge label="HBM bandwidth in use" value={step.hbmBusy} color={MEMORY} />
          <Gauge label={step.unit === 'tensor' ? 'Tensor-core FLOPs in use' : 'CUDA-core FLOPs in use'} value={step.unitBusy} color={step.unit === 'tensor' ? COMPUTE : SERIES.amber} />
          <p className="text-[0.6875rem] text-slate-400">
            Time = max(bytes ÷ ({(MEM_EFF * 100).toFixed(0)}% of 3.35 TB/s), FLOPs ÷ ({(H100.mfu * 100).toFixed(0)}% of peak)) + ~{KERNEL_GAP_US} µs kernel gap.
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// The cast
// ---------------------------------------------------------------------------

const CAST: { name: string; spec: string; role: string; color: string }[] = [
  { name: 'Host CPU + PCIe', spec: `PCIe Gen5 · ${(H100.pcieBw / 1e9).toFixed(0)} GB/s`, role: 'Tokenizes text, schedules requests, launches kernels, streams text back. Weights cross PCIe once at startup, never per token.', color: NEUTRAL[500] },
  { name: 'HBM3', spec: `80 GB · ${(H100.hbmBw / 1e12).toFixed(2)} TB/s`, role: 'Off-die DRAM stacks beside the GPU die. Holds every weight, the KV cache and the DeltaNet state. Its bandwidth is the speed limit of decode.', color: SERIES.azure },
  { name: 'L2 cache', spec: '50 MB', role: 'Shared by all SMs. Every byte from HBM passes through it, but 50 MB cannot hold even one 27B layer, so weights do not stay between tokens.', color: SERIES.cyan },
  { name: 'Shared memory + TMA', spec: `${H100.smemKB} KB per SM · ~33 TB/s total`, role: 'Programmer-managed scratchpad. The TMA engine copies weight tiles into it asynchronously while the tensor cores work on the previous tile.', color: SERIES.teal },
  { name: 'Register file', spec: `${H100.regKB} KB per SM`, role: 'Where operands sit while the cores use them. Fastest storage on the chip.', color: NEUTRAL[600] },
  { name: 'Tensor cores', spec: `528 · ${(H100.tensorFlops / 1e12).toFixed(0)} TFLOP/s BF16`, role: 'Matrix-multiply units, four per SM. They do almost all the FLOPs, and at batch 1 they sit idle almost all the time.', color: COMPUTE },
  { name: 'CUDA cores', spec: `16,896 · ${(H100.cudaFlops / 1e12).toFixed(0)} TFLOP/s FP32`, role: 'General-purpose lanes for everything that is not a matmul: RMSNorm, RoPE, SiLU, softmax, sampling.', color: SERIES.amber },
];

// ---------------------------------------------------------------------------
// The lesson
// ---------------------------------------------------------------------------

export default function LessonGpuToken({ onComplete }: { onComplete: () => void }) {
  const still = useStill();
  const [modelId, setModelId] = useState<ModelId>('qwen3.8-27b');
  const [phase, setPhase] = useState<Phase>('decode');
  const [batch, setBatch] = useState(1);
  const [context, setContext] = useState(4096);
  const [prompt, setPrompt] = useState(2048);
  const [rawIdx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<'0.5' | '1' | '2'>('1');
  const [produced, setProduced] = useState(0);
  const [sweep, setSweep] = useState(1);

  const arch = MODEL_ARCHS[modelId];
  const trace = useMemo(
    () => buildTrace(arch, H100, { phase, batch, context, prompt }),
    [arch, phase, batch, context, prompt],
  );
  const steps = trace.steps;
  const idx = Math.min(rawIdx, steps.length - 1);
  const step = steps[idx];
  const dwellMs = 3200 / Number(speed);

  const next = () => {
    if (idx >= steps.length - 1) {
      setProduced((p) => p + 1);
      setIdx(0);
    } else {
      setIdx(idx + 1);
    }
  };
  const prev = () => setIdx(Math.max(0, idx - 1));

  // Auto-advance
  useEffect(() => {
    if (!playing) return;
    const t = window.setTimeout(next, dwellMs);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, idx, dwellMs, steps.length]);

  // Sweep through the layers while a repeated step is on screen
  const sweepStart = useRef(0);
  useEffect(() => {
    if (!step.layerKind || still || !playing) { setSweep(1); return; }
    sweepStart.current = performance.now();
    setSweep(0);
    let raf = 0;
    const tick = (now: number) => {
      const f = Math.min(1, (now - sweepStart.current) / (dwellMs * 0.9));
      setSweep(f);
      if (f < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [step.id, step.layerKind, still, playing, dwellMs]);

  // The sampled token appears in the output once the "stream" step is reached.
  const streamIdx = steps.findIndex((s) => s.id === 'stream');
  const shown = produced + (idx >= streamIdx ? 1 : 0);
  const tokensOut = phase === 'decode'
    ? OUTPUT_TOKENS.slice(0, Math.min(shown, OUTPUT_TOKENS.length)).join('') + (shown > OUTPUT_TOKENS.length ? ' …' : '')
    : (shown > 0 ? OUTPUT_TOKENS[0] : '');

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLButtonElement) return;
    if (e.key === 'ArrowRight') { e.preventDefault(); next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
    else if (e.key === ' ') { e.preventDefault(); setPlaying((p) => !p); }
  };

  const reset = () => { setIdx(0); setProduced(0); setPlaying(false); };

  // Headline numbers
  const tp = totalParams(arch);
  const streamBytes = streamedWeightBytes(arch);
  const msPerPass = trace.totalUs / 1000;
  const perUserTps = 1000 / msPerPass;
  const decode = phase === 'decode';
  const other = MODEL_ARCHS[modelId === 'qwen3.8-27b' ? 'qwen3-8b' : 'qwen3.8-27b'];
  const ridge = tensorRidge(H100);

  return (
    <SectionBody sourceRef="reference/scaling-book/gpus.md — What Is a GPU? · reference/scaling-book/inference.md — What Does Sampling Look Like? · Hugging Face config.json for Qwen3-8B and Qwen3.8-27B">
      {/* ---------- Intro ---------- */}
      <GlassCard className="p-5 space-y-4">
        <p className="text-sm leading-relaxed text-slate-600 max-w-[var(--pref-measure)]">
          The last section followed a token through the model as maths. This one follows the <em>bytes</em> through
          a real chip. The setup is one <strong>NVIDIA H100 SXM5</strong> serving <strong>{arch.name}</strong> in
          BF16. Press play and the animation steps through one forward pass kernel by kernel. Each step shows which
          part of the chip is working, which bytes move where, and how long that takes.
        </p>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
          {CAST.slice(1).concat(CAST.slice(0, 1)).map((c) => (
            <div key={c.name} className="rounded-xl border border-slate-200 p-3">
              <div className="flex items-center gap-1.5">
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: c.color }} />
                <span className="text-sm font-semibold text-slate-800">{c.name}</span>
              </div>
              <div className="font-mono text-[0.6875rem] text-slate-500 mt-0.5">{c.spec}</div>
              <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">{c.role}</p>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* ---------- Controls ---------- */}
      <GlassCard className="p-5 space-y-4">
        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <div className="text-xs font-medium text-slate-600">Model</div>
            <SegmentedControl<ModelId>
              size="sm"
              value={modelId}
              onChange={(v) => { setModelId(v); setIdx(0); setProduced(0); }}
              options={[
                { value: 'qwen3.8-27b', label: 'Qwen3.8-27B (hybrid)' },
                { value: 'qwen3-8b', label: 'Qwen3-8B (classic)' },
              ]}
            />
            <p className="text-[0.6875rem] text-slate-500 leading-relaxed">
              {arch.blurb}{' '}
              <a className="underline hover:text-[var(--color-accent)]" href={arch.sourceUrl} target="_blank" rel="noreferrer">config</a>
            </p>
          </div>
          <div className="space-y-1.5">
            <div className="text-xs font-medium text-slate-600">Phase</div>
            <SegmentedControl<Phase>
              size="sm"
              value={phase}
              onChange={(v) => { setPhase(v); setIdx(0); setProduced(0); }}
              options={[
                { value: 'decode', label: 'Decode (next token)' },
                { value: 'prefill', label: 'Prefill (read the prompt)' },
              ]}
            />
            <p className="text-[0.6875rem] text-slate-500 leading-relaxed">
              {decode
                ? 'Generating one more token for each sequence in the batch, with the prompt already in the cache.'
                : 'Reading the whole prompt in one forward pass. It ends with the first output token.'}
            </p>
          </div>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          {decode ? (
            <>
              <SliderControl label="Batch (sequences decoded together)" value={batch} min={1} max={512} step={1} onChange={setBatch} logScale conceptId="critical-batch" />
              <SliderControl label="Context already in each KV cache" value={context} min={128} max={131072} step={128} onChange={setContext} unit="tokens" logScale conceptId="kv-cache" />
            </>
          ) : (
            <SliderControl label="Prompt length" value={prompt} min={16} max={32768} step={16} onChange={setPrompt} unit="tokens" logScale conceptId="prefill" />
          )}
        </div>
      </GlassCard>

      {/* ---------- KPIs ---------- */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard
          label={decode ? 'Time per token step' : 'Time to first token'}
          value={msPerPass < 10 ? msPerPass.toFixed(2) : msPerPass.toFixed(1)}
          unit="ms"
          subValue={`weight floor ${fmtUs(trace.weightFloorUs)}`}
        />
        <KpiCard
          label={decode ? 'Tokens / s' : 'Prompt tokens / s'}
          value={decode ? fmtCount(Math.round(perUserTps * batch)) : fmtCount(Math.round(prompt / (msPerPass / 1000)))}
          subValue={decode ? (batch > 1 ? `${perUserTps.toFixed(perUserTps < 10 ? 1 : 0)} per user × ${batch}` : 'one user') : `${fmtCount(prompt)} tokens`}
        />
        <KpiCard
          label="HBM used"
          value={(trace.memory.total / 1e9).toFixed(1)}
          unit="/ 80 GB"
          colorStyle={trace.memory.fits ? undefined : { color: MEMORY }}
          subValue={trace.memory.fits ? `weights ${fmtBytes(trace.memory.weights)}` : 'does not fit'}
        />
        <KpiCard
          label="Time memory-bound"
          value={`${(trace.memBoundShare * 100).toFixed(0)}%`}
          subValue={trace.memBoundShare > 0.5 ? 'waiting on HBM' : 'tensor cores busy'}
        />
      </div>
      {!trace.memory.fits && (
        <div className="rounded-xl px-3 py-2 text-sm font-semibold bg-rose-100 text-rose-700">
          This setting needs {fmtBytes(trace.memory.total)} but the H100 has 80 GB. The server would have to shrink the
          batch, shorten the context, quantize, or shard across GPUs. The timings below assume it somehow fit.
        </div>
      )}

      {/* ---------- The animation ---------- */}
      <GlassCard className="p-4 sm:p-5 space-y-4">
        <div
          tabIndex={0}
          onKeyDown={onKey}
          className="space-y-4 outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] rounded-xl"
          aria-label="Forward-pass animation. Left and right arrows step, space plays or pauses."
        >
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => setPlaying((p) => !p)}
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold text-white bg-[var(--color-accent)] hover:opacity-90">
              {playing ? <Pause style={{ width: 14, height: 14 }} /> : <Play style={{ width: 14, height: 14 }} />}
              {playing ? 'Pause' : 'Play'}
            </button>
            <button type="button" onClick={prev} aria-label="Previous step"
              className="nav-icon-btn inline-flex items-center justify-center rounded-lg border border-slate-200 w-8 h-8 text-slate-600">
              <SkipBack style={{ width: 14, height: 14 }} />
            </button>
            <button type="button" onClick={next} aria-label="Next step"
              className="nav-icon-btn inline-flex items-center justify-center rounded-lg border border-slate-200 w-8 h-8 text-slate-600">
              <SkipForward style={{ width: 14, height: 14 }} />
            </button>
            <button type="button" onClick={reset} aria-label="Restart"
              className="nav-icon-btn inline-flex items-center justify-center rounded-lg border border-slate-200 w-8 h-8 text-slate-600">
              <RotateCcw style={{ width: 14, height: 14 }} />
            </button>
            <SegmentedControl<'0.5' | '1' | '2'>
              size="sm"
              className="w-40"
              value={speed}
              onChange={setSpeed}
              options={[{ value: '0.5', label: '0.5×' }, { value: '1', label: '1×' }, { value: '2', label: '2×' }]}
            />
            <span className="text-xs text-slate-500 ml-auto font-mono">
              step {idx + 1} / {steps.length}
              {decode && produced > 0 && ` · token ${produced + 1}`}
            </span>
          </div>
          <p className="text-[0.6875rem] text-slate-400 -mt-2">
            Click inside this panel, then use ← → to step and space to play or pause. Click the timeline below to jump to any step.
          </p>

          <div className="grid xl:grid-cols-[minmax(0,1fr)_340px] gap-5">
            <div className="space-y-3 min-w-0">
              <div className="overflow-x-auto custom-scrollbar">
                <DieMap trace={trace} step={step} tokensOut={tokensOut} still={still} />
              </div>
              <LayerStrip trace={trace} step={step} sweep={sweep} />
            </div>
            <div className="xl:border-l xl:border-slate-200 xl:pl-5">
              <StepPanel trace={trace} step={step} />
            </div>
          </div>
        </div>

        <Figure
          takeaway={decode && batch <= 8
            ? `Read the bar left to right: one forward pass. Almost all of it is blue, the weight matmuls, and each of them is memory-bound. At batch ${batch} the H100 spends its time moving ${fmtBytes(streamBytes)} of weights from HBM, not multiplying. Raise the batch and the blue barely grows while tokens per step multiply. Raise the context and the red attention slice takes over.`
            : decode
              ? `At batch ${batch} each weight tile is reused ${batch} times once it reaches shared memory, so the weight matmuls cost little more than at batch 1 while producing ${batch}× the tokens. What grows is the per-sequence work batching cannot share: KV-cache reads (red)${arch.linear ? ' and the DeltaNet state (violet)' : ''}.`
              : `Prefill pushes ${fmtCount(prompt)} rows through every matmul, so each weight byte is used ${fmtCount(prompt)} times. Past the ridge (~${ridge.toFixed(0)} rows) the same kernels turn compute-bound, which is why reading a prompt is fast per token and writing a reply is not.`}
        >
          <Timeline trace={trace} idx={idx} onPick={(i) => { setIdx(i); setPlaying(false); }} />
        </Figure>
      </GlassCard>

      {/* ---------- The punchline ---------- */}
      <GlassCard className="p-5 space-y-3">
        <h3 className="font-bold text-slate-800">The one number that explains decode</h3>
        <p className="text-sm leading-relaxed text-slate-600 max-w-[var(--pref-measure)]">
          {arch.name} has {(tp.total / 1e9).toFixed(1)} B parameters. Every decode step must stream{' '}
          <strong>{fmtBytes(streamBytes)}</strong> of them (all layers plus the LM head) from HBM into the SMs, and
          HBM delivers at most {(H100.hbmBw / 1e12).toFixed(2)} TB/s. So no single sequence can get a token faster than{' '}
          <strong>{fmtUs(trace.weightFloorUs)}</strong>, about <strong>{(1e6 / trace.weightFloorUs).toFixed(0)} tokens/s</strong>,
          however many tensor cores sit idle. {other.name} streams {fmtBytes(streamedWeightBytes(other))}, so its floor is{' '}
          {fmtUs((streamedWeightBytes(other) / H100.hbmBw) * 1e6)}. The model above adds what that floor ignores: real
          kernels reach about {(MEM_EFF * 100).toFixed(0)}% of peak bandwidth, small kernels pay a fixed gap, and
          attention and norms move bytes of their own.
        </p>
      </GlassCard>

      {/* ---------- Go deeper ---------- */}
      <GoDeeper label="Why the weights don't just stay on the chip" hint="50 MB of L2 against gigabytes of weights">
        <p className="text-sm leading-relaxed text-slate-600 max-w-[var(--pref-measure)]">
          All on-chip storage on an H100 adds up to about 50 MB of L2, 30 MB of shared memory and 33 MB of registers.
          One {arch.name} feed-forward block alone is{' '}
          {fmtBytes(3 * arch.hidden * arch.ffn * arch.bytesPerParam)}. By the time the last layer has run, the first
          layer's weights were evicted long ago, so the next token has to fetch them from HBM again. That is why decode
          time tracks <em>model bytes ÷ HBM bandwidth</em> so closely, and why quantizing to FP8 roughly halves it.
          Chips that do keep weights on-chip (Groq, Cerebras) need hundreds of chips to hold a model this size.
        </p>
      </GoDeeper>

      {arch.linear && (
        <GoDeeper label="Hybrid attention: why Qwen3.8 keeps a fixed-size state" hint="KV cache vs DeltaNet state">
          <p className="text-sm leading-relaxed text-slate-600 max-w-[var(--pref-measure)]">
            A full-attention layer remembers the past by keeping every key and value. For Qwen3.8-27B that is{' '}
            {fmtBytes(kvBytesPerToken(arch))} per token across its {countLayers(arch).full} full layers (Qwen3-8B needs{' '}
            {fmtBytes(kvBytesPerToken(MODEL_ARCHS['qwen3-8b']))} per token across 36). It grows forever, and decode has
            to read all of it every step.
          </p>
          <p className="text-sm leading-relaxed text-slate-600 max-w-[var(--pref-measure)]">
            A Gated DeltaNet layer compresses the past into one fixed matrix per head and edits it with the delta rule:
            decay the old state, subtract what it already predicted for the new key, and write in the new value. The
            state is {fmtBytes(stateBytesPerSeq(arch))} per sequence for all {countLayers(arch).linear} linear layers,
            whether the context is 100 tokens or 100,000. Set the context slider to 131,072 and switch between the two
            models: the 8B model's red attention slice swells to rival its weights, while the 27B model's grows less
            than half as fast per token of context, despite the larger model.
          </p>
        </GoDeeper>
      )}

      <GoDeeper label="What a production server adds" hint="CUDA graphs, continuous batching, paged KV">
        <ul className="text-sm leading-relaxed text-slate-600 list-disc pl-5 space-y-1.5 max-w-[var(--pref-measure)]">
          <li>
            <strong>CUDA graphs.</strong> A forward pass is hundreds of kernels
            ({fmtCount(steps.filter((s) => s.unit === 'tensor' || s.unit === 'cuda' || s.unit === 'copy').reduce((a, s) => a + s.repeat, 0))} here).
            Launching each one from Python would cost more than the kernels themselves at batch 1, so engines record the
            whole pass once and replay it with one launch.
          </li>
          <li>
            <strong>Continuous batching.</strong> Sequences join and leave the batch at every step, so the weight stream
            is always shared by as many users as memory allows. <ConceptTag id="continuous-batching" />
          </li>
          <li>
            <strong>Paged KV cache.</strong> The cache is stored in fixed-size blocks, like virtual memory, so it can grow
            without large contiguous allocations. <ConceptTag id="paged-attention" />
          </li>
          <li>
            <strong>Speculative decoding.</strong> Qwen3.8 ships a multi-token-prediction head that drafts several tokens.
            One forward pass then verifies them all, so each weight stream yields more than one token.{' '}
            <ConceptTag id="speculative-decoding" />
          </li>
          <li>
            <strong>Fusion.</strong> Real kernels fuse the norms, RoPE and activations into their neighbours, so the
            amber slivers above mostly vanish into the matmuls.
          </li>
        </ul>
        <p className="text-xs text-slate-400 max-w-[var(--pref-measure)]">
          Sources: model shapes from the Hugging Face config files for{' '}
          <a className="underline" href="https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json" target="_blank" rel="noreferrer">Qwen3-8B</a> and{' '}
          <a className="underline" href="https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json" target="_blank" rel="noreferrer">Qwen3.8-27B</a>.
          H100 figures from NVIDIA's{' '}
          <a className="underline" href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/" target="_blank" rel="noreferrer">Hopper architecture in depth</a>{' '}
          and{' '}
          <a className="underline" href="https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth" target="_blank" rel="noreferrer">Chips and Cheese</a>.
          Kernel gaps, launch cost and achieved-bandwidth fractions are estimates. The vision encoder and the MTP head
          are left out of the timings.
        </p>
      </GoDeeper>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: `At batch 1, why can't Qwen3.8-27B decode much faster than about ${(1e6 / ((streamedWeightBytes(MODEL_ARCHS['qwen3.8-27b']) / H100.hbmBw) * 1e6)).toFixed(0)} tokens/s on one H100?`,
            options: [
              'The tensor cores are fully busy doing the matmuls',
              'Every token must stream ~51 GB of weights from HBM, and HBM delivers 3.35 TB/s',
              'PCIe is too slow to send the weights for each token',
              'The CPU tokenizer cannot keep up',
            ],
            answer: 1,
            explain: 'The weights do not fit in on-chip memory, so each token re-reads them from HBM: about 51 GB (every layer plus the LM head; the embedding is only a lookup) ÷ 3.35 TB/s ≈ 15 ms. The tensor cores are idle more than 99% of the time at batch 1, and the weights never cross PCIe during serving.',
          },
          {
            q: 'You raise the decode batch from 1 to 64. What happens to the weight matmul steps?',
            options: [
              'They take about 64 times longer',
              'Their time barely changes, because each weight tile is reused for all 64 rows once it is in shared memory',
              'They move from the tensor cores to the CUDA cores',
              'They disappear because the weights are cached in L2',
            ],
            answer: 1,
            explain: 'The bytes are almost the same (the weights) while the FLOPs grow 64×, so intensity rises from about 1 to about 64 FLOPs per byte. That is still below the ~295 ridge, so the steps stay memory-bound and cost about the same, and 64 tokens come out instead of one.',
          },
          {
            q: 'Why does attention over the KV cache not get cheaper per token with a bigger batch?',
            options: [
              'Because attention runs on the CPU',
              'Because each sequence has its own KV cache, so there is no shared data to reuse across the batch',
              'Because the KV cache lives in L2',
              'It does: attention intensity grows with batch like the matmuls',
            ],
            answer: 1,
            explain: 'Weights are shared by every sequence in the batch; KV caches are not. Attention intensity stays at the query-to-KV head ratio (4 for Qwen3-8B, 6 for Qwen3.8-27B) at any batch size, which is why long contexts eat decode throughput, and why Qwen3.8 replaces three of every four attention layers with a fixed-size state.',
          },
        ]}
      />
    </SectionBody>
  );
}
