import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import {
  Cpu, Grid3x3, Server, Layers, Network, Sparkles, Play, Pause, RotateCcw,
  MemoryStick, Gauge, Zap,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { HARDWARE_PROFILES, effectiveMfu, onChipBwRatio, type HardwareProfile } from '../lib/hardware';
import ConceptTag from './ui/ConceptTag';

// ---------------------------------------------------------------------------
// Palette + formatters (mirror DeepDive / Serving)
// ---------------------------------------------------------------------------
const C = {
  compute: '#22c48b',
  memory: '#f25f7d',
  ridge: '#f43f5e',
  accent: '#5b7cfa',
  accentSoft: '#8aa0ff',
  sky: '#0ea5e9',
  amber: '#f59e0b',
  violet: '#8b5cf6',
  slate: '#94a3b8',
  ink: '#0b1220',
};

function fmtNum(v: number, digits = 0) {
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: digits });
}

function fmtFlops(f: number): string {
  if (f >= 1e15) return `${(f / 1e15).toFixed(2)} PFLOP/s`;
  if (f >= 1e12) return `${(f / 1e12).toFixed(1)} TFLOP/s`;
  return `${(f / 1e9).toFixed(1)} GFLOP/s`;
}

function ChartTip({ active, payload, label, unit = '' }: any) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="glass-tooltip px-3 py-2 text-xs space-y-1">
      {label !== undefined && <div className="font-semibold text-slate-800">{label}</div>}
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-slate-500">{p.name}</span>
          <span className="font-mono font-semibold text-slate-800">{fmtNum(p.value, 1)}{unit}</span>
        </div>
      ))}
    </div>
  );
}

function SectionCard({ id, icon: IconCmp, color, number, title, children }: any) {
  return (
    <section id={id} className="glass-card p-6 sm:p-8 mb-8 scroll-mt-24">
      <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
        <span className="inline-flex items-center justify-center w-9 h-9 rounded-xl mr-3 text-white shrink-0"
          style={{ background: color }}>
          <IconCmp className="w-5 h-5" />
        </span>
        <span className="text-3xl font-black mr-3 opacity-20" style={{ color }}>{number}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function HeroKpi({ icon: I, label, value, sub }: any) {
  return (
    <div className="glass rounded-xl p-3 text-left">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 mb-1">
        <I className="w-3.5 h-3.5" /> {label}
      </div>
      <div className="font-mono text-lg font-bold text-slate-900 leading-none">{value}</div>
      <div className="text-[11px] text-slate-400 mt-1 truncate">{sub}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Systolic array animation — hand-rolled SVG, no chart library.
//
// A weight-stationary N x N array computing Z = X @ W. Each cell holds one
// weight forever; activations march in from the left, partial sums march down.
// Cell (r, c) starts working at cycle r + c — the diagonal wavefront — and then
// does one multiply-accumulate per cycle until the input rows run out.
// ---------------------------------------------------------------------------
export function SystolicArray({ n = 4, rows = 6 }: { n?: number; rows?: number }) {
  const totalCycles = 2 * n - 1 + rows;
  const [cycle, setCycle] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setCycle((c) => (c + 1) % (totalCycles + 3)), 620);
    return () => clearInterval(t);
  }, [playing, totalCycles]);

  const cellState = (r: number, c: number) => {
    const start = r + c;
    if (cycle < start) return 'idle';
    if (cycle < start + rows) return 'active';
    return 'done';
  };

  const activeCount = useMemo(() => {
    let k = 0;
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (cellState(r, c) === 'active') k++;
    return k;
  }, [cycle, n, rows]);

  const cellSize = 54;
  const gap = 10;
  const originX = 96;
  const originY = 64;
  const width = originX + n * (cellSize + gap) + 70;
  const height = originY + n * (cellSize + gap) + 76;

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <button onClick={() => setPlaying((p) => !p)}
          className="glass rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-700 inline-flex items-center gap-1.5 hover:border-accent/40">
          {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={() => { setPlaying(false); setCycle((c) => (c + 1) % (totalCycles + 3)); }}
          className="glass rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-700 hover:border-accent/40">
          Step →
        </button>
        <button onClick={() => setCycle(0)}
          className="glass rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-700 inline-flex items-center gap-1.5 hover:border-accent/40">
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </button>
        <span className="text-xs text-slate-400 ml-1">
          cycle <span className="font-mono font-bold text-slate-700">{cycle}</span> ·{' '}
          <span className="font-mono font-bold" style={{ color: activeCount === n * n ? C.compute : C.amber }}>
            {activeCount}/{n * n}
          </span> cells doing a multiply-accumulate
        </span>
      </div>

      <div className="overflow-x-auto custom-scrollbar flex justify-center">
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width, minWidth: 420 }}
          role="img" aria-label="Animated systolic array wavefront">
          {/* activations streaming in from the left */}
          {Array.from({ length: n }).map((_, r) => {
            const y = originY + r * (cellSize + gap) + cellSize / 2;
            const fed = cycle >= r;
            return (
              <g key={`in-${r}`}>
                <line x1={26} y1={y} x2={originX - 6} y2={y} stroke={fed ? C.sky : '#cbd5e1'} strokeWidth={2}
                  strokeDasharray={fed ? undefined : '3 3'} />
                <text x={12} y={y + 4} fontSize={11} fontWeight={700} fill={fed ? C.sky : '#cbd5e1'}>
                  x{r}
                </text>
              </g>
            );
          })}

          {/* the array */}
          {Array.from({ length: n }).map((_, r) =>
            Array.from({ length: n }).map((__, c) => {
              const st = cellState(r, c);
              const x = originX + c * (cellSize + gap);
              const y = originY + r * (cellSize + gap);
              const fill = st === 'active' ? C.compute : st === 'done' ? '#e2e8f0' : '#f1f5f9';
              const stroke = st === 'active' ? C.compute : '#cbd5e1';
              return (
                <g key={`c-${r}-${c}`}>
                  <rect x={x} y={y} width={cellSize} height={cellSize} rx={10}
                    fill={fill} fillOpacity={st === 'active' ? 0.22 : 1} stroke={stroke} strokeWidth={st === 'active' ? 2.5 : 1} />
                  <text x={x + cellSize / 2} y={y + cellSize / 2 - 3} textAnchor="middle" fontSize={11}
                    fontWeight={700} fill={st === 'active' ? '#0f766e' : '#94a3b8'}>
                    w{r}{c}
                  </text>
                  <text x={x + cellSize / 2} y={y + cellSize / 2 + 12} textAnchor="middle" fontSize={9}
                    fill={st === 'active' ? C.compute : '#cbd5e1'}>
                    {st === 'active' ? 'MAC' : st === 'done' ? 'done' : 'idle'}
                  </text>
                </g>
              );
            })
          )}

          {/* partial sums draining downward */}
          {Array.from({ length: n }).map((_, c) => {
            const x = originX + c * (cellSize + gap) + cellSize / 2;
            const yTop = originY + n * (cellSize + gap) - gap + 2;
            const ready = cycle >= n - 1 + c;
            return (
              <g key={`out-${c}`}>
                <line x1={x} y1={yTop} x2={x} y2={yTop + 26} stroke={ready ? C.violet : '#cbd5e1'} strokeWidth={2}
                  strokeDasharray={ready ? undefined : '3 3'} />
                <text x={x} y={yTop + 42} textAnchor="middle" fontSize={11} fontWeight={700}
                  fill={ready ? C.violet : '#cbd5e1'}>
                  z{c}
                </text>
              </g>
            );
          })}

          <text x={12} y={30} fontSize={11} fontWeight={700} fill={C.sky}>activations →</text>
          <text x={width - 150} y={height - 8} fontSize={11} fontWeight={700} fill={C.violet}>↓ partial sums</text>
        </svg>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Memory tier table, driven by the hardware profile
// ---------------------------------------------------------------------------
function MemoryTiers({ hw }: { hw: HardwareProfile }) {
  const peakFlops = hw.tflops * 1e12;
  const hbmBw = hw.memBw * 1e12;
  const ratio = onChipBwRatio(hw);
  const onChipBw = hbmBw * ratio;

  const tiers = [
    {
      name: hw.vendor === 'Google' ? 'VRegs / VMEM tile' : 'Registers',
      size: hw.vendor === 'Google' ? '256 kB' : '32 MB',
      bw: onChipBw * 6,
      note: 'Feeds the ALUs directly. Effectively free bandwidth; you just cannot hold much.',
    },
    {
      name: hw.vendor === 'Google' ? 'VMEM (on-chip scratchpad)' : 'SMEM + L2',
      size: hw.onChipMemMB ? `${hw.onChipMemMB} MB` : '—',
      bw: onChipBw,
      note: 'Where tiles live during a matmul. Weights prefetched here bypass the HBM roofline entirely.',
    },
    {
      name: 'HBM',
      size: `${hw.capacity} GB`,
      bw: hbmBw,
      note: 'Model weights and the KV cache. The bandwidth every inference roofline on this site is about.',
    },
    hw.linkBwGBs ? {
      name: hw.vendor === 'Google' ? 'ICI (chip → chip)' : 'NVLink (GPU → GPU)',
      size: hw.domainSize ? `${fmtNum(hw.domainSize)} chips` : '—',
      bw: hw.linkBwGBs * 1e9,
      note: 'Sharded matmuls. Roughly an order of magnitude below HBM.',
    } : null,
    hw.scaleOutBwGBs ? {
      name: 'DCN / InfiniBand',
      size: 'datacenter',
      bw: hw.scaleOutBwGBs * 1e9,
      note: 'Data parallelism between pods. Two more orders of magnitude down.',
    } : null,
  ].filter(Boolean) as { name: string; size: string; bw: number; note: string }[];

  const chartData = tiers.map((t) => ({ name: t.name, ridge: peakFlops / t.bw }));

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      <div className="overflow-x-auto custom-scrollbar">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-[11px] uppercase tracking-wide text-slate-400 border-b border-slate-200">
              <th className="py-2 pr-3 text-left">Tier</th>
              <th className="py-2 pr-3 text-right">Size</th>
              <th className="py-2 pr-3 text-right">Bandwidth</th>
              <th className="py-2 text-right">Ridge</th>
            </tr>
          </thead>
          <tbody>
            {tiers.map((t) => (
              <tr key={t.name} className="border-b border-slate-100 align-top">
                <td className="py-2 pr-3">
                  <div className="font-semibold text-slate-800">{t.name}</div>
                  <div className="text-[11px] text-slate-400 leading-snug max-w-[22rem]">{t.note}</div>
                </td>
                <td className="py-2 pr-3 text-right font-mono text-slate-600">{t.size}</td>
                <td className="py-2 pr-3 text-right font-mono text-slate-600">
                  {t.bw >= 1e12 ? `${(t.bw / 1e12).toFixed(1)} TB/s` : `${(t.bw / 1e9).toFixed(0)} GB/s`}
                </td>
                <td className="py-2 text-right font-mono font-bold text-slate-900">{fmtNum(peakFlops / t.bw)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-slate-400 mt-3">
          Every tier has its own ridge — the intensity at which <code>{hw.id}</code> stops being starved by <em>that</em>
          link. Move an operand up a tier and its ridge drops by the bandwidth ratio, which is the entire payoff of
          tiling, prefetching, and keeping tensor parallelism inside one domain.
        </p>
      </div>

      <div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 24, left: 10, bottom: 22 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
              <XAxis type="number" scale="log" domain={[0.5, 'dataMax']}
                tickFormatter={(v: any) => fmtNum(v)}
                label={{ value: 'Ridge (FLOPs/byte, log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
              <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 10 }} />
              <Tooltip content={<ChartTip />} />
              <ReferenceLine x={peakFlops / hbmBw} stroke={C.ridge} strokeDasharray="4 4" />
              <Bar dataKey="ridge" name="ridge" radius={[0, 6, 6, 0]} barSize={22}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={[C.compute, C.sky, C.memory, C.violet, C.amber][i % 5]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-slate-400 mt-2">
          Log scale — the span from registers to the datacenter network is four or five orders of magnitude. That spread
          is why &ldquo;is this op compute-bound?&rdquo; is meaningless until you say <em>bound by which link</em>.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main tab
// ---------------------------------------------------------------------------
export default function HardwareTab() {
  const withInternals = HARDWARE_PROFILES.filter((h) => h.linkBwGBs !== undefined);
  const [hwId, setHwId] = useState(withInternals.find((h) => h.id === 'TPU v5p')?.id ?? withInternals[0].id);
  const hw = HARDWARE_PROFILES.find((h) => h.id === hwId) || withInternals[0];

  const peakFlops = hw.tflops * 1e12;
  const hbmBw = hw.memBw * 1e12;
  const ridge = peakFlops / hbmBw;
  const mfu = effectiveMfu(hw);

  const sections = [
    { id: 'tpu', label: 'What Is a TPU', icon: Cpu },
    { id: 'systolic', label: 'Systolic Array', icon: Grid3x3 },
    { id: 'gpu', label: 'What Is a GPU', icon: Server },
    { id: 'hierarchy', label: 'Memory Hierarchy', icon: Layers },
    { id: 'network', label: 'Networking', icon: Network },
    { id: 'rules', label: 'Scaling Rules', icon: Sparkles },
  ];

  return (
    <div className="pb-16 max-w-6xl mx-auto mt-6 px-4">
      {/* ---- Hero ---- */}
      <section className="text-center mb-10">
        <div className="inline-flex items-center gap-2 glass-chip px-3 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4">
          <Cpu className="w-3.5 h-3.5 text-accent" /> Inside the accelerator
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
          Where the roofline <span className="text-accent">comes from</span>
        </h1>
        <p className="text-slate-500 max-w-3xl mx-auto leading-relaxed">
          Every ridge point on this site is a ratio of two hardware numbers. This page opens the chip up and shows you
          where those numbers live: the <strong>matrix unit</strong> that produces the FLOPs, the <strong>memory
          hierarchy</strong> that produces the bytes, and the <strong>network</strong> that decides how far you can shard
          before the fabric becomes the roof.
        </p>

        {/* hardware picker */}
        <div className="mt-6 max-w-xl mx-auto">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 mb-2">
            Numbers on this page follow your selection
          </div>
          <div className="flex flex-wrap justify-center gap-1.5">
            {withInternals.map((h) => (
              <button key={h.id} onClick={() => setHwId(h.id)}
                className={cn('glass rounded-lg px-2.5 py-1.5 text-[11px] font-semibold transition-colors',
                  hwId === h.id ? 'bg-accent text-white border-accent' : 'text-slate-600 hover:border-accent/40')}>
                {h.id}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-8 max-w-3xl mx-auto">
          <HeroKpi icon={Cpu} label="Matrix peak" value={fmtFlops(peakFlops)} sub={hw.computeUnits ?? hw.arch} />
          <HeroKpi icon={MemoryStick} label="HBM" value={`${hw.memBw} TB/s`} sub={`${hw.capacity} GB`} />
          <HeroKpi icon={Gauge} label="HBM ridge" value={fmtNum(ridge)} sub="FLOPs / byte" />
          <HeroKpi icon={Zap} label="Realistic MFU" value={`${Math.round(mfu * 100)}%`} sub={fmtFlops(peakFlops * mfu)} />
        </div>

        <div className="flex flex-wrap justify-center gap-2 mt-6">
          <ConceptTag id="tpu-architecture" />
          <ConceptTag id="systolic-array" />
          <ConceptTag id="sm-streaming-multiprocessor" />
          <ConceptTag id="vector-unit-ridge" />
          <ConceptTag id="memory-hierarchy" />
          <ConceptTag id="two-bandwidth-roofline" />
          <ConceptTag id="ici-topology" />
          <ConceptTag id="nvlink-domain" />
          <ConceptTag id="mfu" />
        </div>
      </section>

      {/* ---- Section nav ---- */}
      <nav className="sticky top-0 z-30 -mx-2 px-2 py-3 mb-8 blur-[1px] backdrop-blur-md bg-[#eef1fb]/70 rounded-2xl">
        <div className="flex gap-1.5 overflow-x-auto custom-scrollbar py-1">
          {sections.map((s) => (
            <a key={s.id} href={`#${s.id}`}
              className="shrink-0 inline-flex items-center gap-1.5 glass-chip px-3 py-1.5 text-[11px] font-semibold text-slate-600 hover:text-accent hover:border-accent/40 transition-colors">
              <s.icon className="w-3.5 h-3.5" /> {s.label}
            </a>
          ))}
        </div>
      </nav>

      {/* ---- 01 What is a TPU ---- */}
      <SectionCard id="tpu" icon={Cpu} color={C.accent} number="01" title="What Is a TPU?">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            A TPU is, to a first approximation, <strong>one very good matrix-multiplication machine bolted to a stack of
            fast memory</strong>. The compute half is called a <strong>TensorCore</strong>; the memory half is HBM. That
            is the whole chip, and it is deliberately much simpler than a GPU.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mb-6">
          {[
            { unit: 'MXU', full: 'Matrix Multiply Unit', color: C.compute,
              body: 'A systolic array that performs bf16[8,128] @ bf16[128,128] → f32[8,128] every 8 cycles (256×256 on Trillium and later). About 5e13 FLOP/s each; most TensorCores carry 2 or 4 of them. This is where essentially all of your FLOPs come from — and the only unit the famous ridge of ~240 describes.' },
            { unit: 'VPU', full: 'Vector Processing Unit', color: C.amber,
              body: 'Elementwise work: ReLU, pointwise add and multiply, and reductions. Four independently programmable 8×128 units, 4096 ALUs in total. Its peak is roughly two orders of magnitude below the MXU — so it has its own, far smaller ridge, and that is the number to use when you reason about softmax or layernorm.' },
            { unit: 'VMEM', full: 'Vector Memory', color: C.sky,
              body: 'The on-chip scratchpad that feeds both units. Small but enormously faster than HBM. Anything you can keep resident here escapes the HBM roofline completely, which is why TPUs can be so strong at inference when the weights fit.' },
          ].map((u) => (
            <div key={u.unit} className="glass rounded-xl p-5">
              <div className="inline-flex items-center gap-2 mb-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: u.color }} />
                <span className="font-bold text-slate-900">{u.unit}</span>
              </div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-2">{u.full}</div>
              <p className="text-sm text-slate-600 leading-relaxed">{u.body}</p>
            </div>
          ))}
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-2">Two units, two ridges</h3>
          <p className="text-sm text-slate-600">
            This split is the reason a single chip does not have <em>a</em> ridge point. On{' '}
            <code>{hw.id}</code> the matrix unit runs at {fmtFlops(peakFlops)} against {hw.memBw} TB/s of HBM, giving a
            ridge of <strong>{fmtNum(ridge)}</strong> FLOPs/byte. The vector unit runs at roughly{' '}
            {fmtFlops((hw.vectorTflops ?? hw.tflops / 60) * 1e12)} against the <em>same</em> HBM, giving a ridge of about{' '}
            <strong>{fmtNum(((hw.vectorTflops ?? hw.tflops / 60) * 1e12) / hbmBw, 1)}</strong>. Compare a softmax against
            the matmul ridge and you will conclude, wrongly, that it is catastrophically inefficient.
          </p>
        </div>
      </SectionCard>

      {/* ---- 02 Systolic array ---- */}
      <SectionCard id="systolic" icon={Grid3x3} color={C.compute} number="02" title="How a Systolic Array Works">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Here is the mechanism the entire roofline rests on. A systolic array is a grid of tiny multiply-accumulate
            cells, each holding <strong>one weight, permanently</strong>. Activations march in from the left; partial
            sums march down. Every cell hands its result straight to its neighbour — nothing goes back to memory in
            between.
          </p>
          <p>
            Watch the diagonal <strong>wavefront</strong> fill the array. It takes <i>2n&minus;1</i> cycles to fill and
            the same to drain, but in between, all <i>n&sup2;</i> cells do a multiply-accumulate <em>every single
            cycle</em> — while the array reads only one new activation per row per cycle.
          </p>
        </div>

        <div className="glass rounded-xl p-5 mb-6">
          <SystolicArray n={4} rows={6} />
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-2">Why this is the whole roofline story</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              An <i>n &times; n</i> array performs <i>O(n&sup2;)</i> FLOPs per cycle while loading <i>O(n)</i> new values.
              The hardware therefore <em>manufactures</em> arithmetic intensity — it is structurally capable of ~<i>n</i>
              {' '}FLOPs per byte. On a 128&times;128 MXU that is the same order as the ridge point itself, which is not a
              coincidence: chip designers size the array so that a well-fed matmul just saturates HBM.
            </p>
          </div>
          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-2">And why small batches waste it</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              The array only pays off once it is full. With a batch of 1 you push a single row through a{' '}
              {hw.vendor === 'Google' ? '128×128' : '16×16 tensor-core'} grid, leaving almost every cell idle for almost
              every cycle, and you still paid the full price of loading the weights. That is the same fact as
              &ldquo;decode is memory-bound&rdquo; — seen from the silicon instead of from the equation.
            </p>
          </div>
        </div>
      </SectionCard>

      {/* ---- 03 What is a GPU ---- */}
      <SectionCard id="gpu" icon={Server} color={C.violet} number="03" title="What Is a GPU?">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Modern datacenter GPUs have converged on the same job — but they got there from graphics, and it shows. The
            single biggest structural difference is <strong>modularity</strong>: a TPU has one or two big TensorCores
            under one thread of control, while a GPU has <em>hundreds</em> of small, independent streaming
            multiprocessors (SMs).
          </p>
        </div>

        <div className="overflow-x-auto custom-scrollbar mb-6">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[11px] uppercase tracking-wide text-slate-400 border-b border-slate-200">
                <th className="py-2 pr-3 text-left">GPU term</th>
                <th className="py-2 pr-3 text-left">TPU equivalent</th>
                <th className="py-2 pr-3 text-right">H100</th>
                <th className="py-2 text-right">TPU v5p</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['SM (streaming multiprocessor)', 'TensorCore', '132', '2'],
                ['Warp scheduler', 'VPU slots', '528', '8'],
                ['Tensor Core', 'MXU', '528', '8'],
                ['SMEM (L1 cache)', 'VMEM', '32 MB', '128 MB'],
                ['Registers', 'Vector registers (VRegs)', '32 MB', '256 kB'],
              ].map((r) => (
                <tr key={r[0]} className="border-b border-slate-100">
                  <td className="py-2 pr-3 font-semibold text-slate-800">{r[0]}</td>
                  <td className="py-2 pr-3 text-slate-600">{r[1]}</td>
                  <td className="py-2 pr-3 text-right font-mono text-slate-700">{r[2]}</td>
                  <td className="py-2 text-right font-mono text-slate-700">{r[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          {[
            { t: 'Modularity cuts both ways', b: 'A GPU programmer launches dozens of kernels on independent SMs and things usually "just work". But those kernels can thrash L2 or fail to coalesce loads, and because the hardware — not the compiler — controls scheduling, it is much harder to reason about what actually happened.' },
            { t: 'TPUs get closer to peak', b: 'A TPU has a single thread of control and only vectorized instructions, so the compiler must pipeline every load and every MXU issue by hand. That is more burden up front, and it is exactly why TPUs reach ~95% of peak where GPUs sit at 80–85%.' },
            { t: 'TPUs carry more fast cache', b: 'A v5p has 128 MB of VMEM against an H100’s 32 MB of SMEM. Weights and activations that stay resident there are loaded at on-chip bandwidth, which can make TPUs disproportionately strong at inference when the model fits.' },
          ].map((x) => (
            <div key={x.t} className="glass rounded-xl p-5">
              <h3 className="font-bold text-slate-800 mb-2 text-sm">{x.t}</h3>
              <p className="text-sm text-slate-600 leading-relaxed">{x.b}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* ---- 04 Memory hierarchy ---- */}
      <SectionCard id="hierarchy" icon={Layers} color={C.sky} number="04" title="Memory Hierarchy, Chip by Chip">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            &ldquo;Bandwidth&rdquo; is never one number. Each tier below has its own byte/s figure, and therefore its own
            ridge — the intensity at which <code>{hw.id}</code>&rsquo;s matrix unit stops waiting on <em>that</em> link.
          </p>
        </div>
        <MemoryTiers hw={hw} />
      </SectionCard>

      {/* ---- 05 Networking ---- */}
      <SectionCard id="network" icon={Network} color={C.amber} number="05" title="Networking: Torus vs Domain">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Once a model needs more than one chip, the interconnect becomes a roofline of its own — and TPUs and GPUs
            solve the problem in opposite ways.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-2 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: C.accent }} /> TPU: an ICI torus
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Chips link directly to their nearest neighbours — 4 of them on inference generations (a 2D torus), 6 on
              v4/v5p (a 3D torus) — with wraparound edges. The links do not pass through hosts. A torus keeps the maximum
              hop count low as pods grow, so a v5p pod reaches 8,960 chips while every link stays short and cheap.
              Bandwidth per link is modest; there are simply a great many of them.
            </p>
          </div>
          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-2 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: C.violet }} /> GPU: a switched domain
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              GPUs get a small, very fast all-to-all domain: 8 GPUs on one NVLink/NVSwitch node, or 72 in a GB200 NVL72
              rack. Inside the domain any pair talks at full speed. Outside it you drop onto InfiniBand and lose an order
              of magnitude — which is why so many GPU scaling rules are really statements about where the node boundary
              falls.
            </p>
          </div>
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-3">{hw.id} as configured here</h3>
          <div className="grid sm:grid-cols-4 gap-3 text-sm">
            {[
              { l: 'Topology', v: hw.topology ?? '—' },
              { l: 'Domain size', v: hw.domainSize ? `${fmtNum(hw.domainSize)} chips` : '—' },
              { l: 'In-domain link', v: hw.linkBwGBs ? `${fmtNum(hw.linkBwGBs)} GB/s` : '—' },
              { l: 'Scale-out link', v: hw.scaleOutBwGBs ? `${fmtNum(hw.scaleOutBwGBs, 2)} GB/s` : '—' },
            ].map((x) => (
              <div key={x.l} className="glass rounded-lg p-3">
                <div className="text-[11px] uppercase tracking-wide text-slate-400">{x.l}</div>
                <div className="font-mono font-bold text-slate-900 mt-1 text-sm">{x.v}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-400 mt-3">
            Note the gap between the last two columns. Whatever you do inside the domain is roughly{' '}
            {hw.linkBwGBs && hw.scaleOutBwGBs ? `${(hw.linkBwGBs / hw.scaleOutBwGBs).toFixed(0)}×` : 'an order of magnitude'}{' '}
            cheaper than the same collective across it — which is the hardware fact behind every sharding rule below.
          </p>
        </div>
      </SectionCard>

      {/* ---- 06 Scaling rules ---- */}
      <SectionCard id="rules" icon={Sparkles} color={C.compute} number="06" title="Scaling Rules of Thumb">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Put the memory roofline and the network roofline together and a short list of thresholds falls out. These are
            the numbers to carry around; the derivations are in the Deep Dive problems.
          </p>
        </div>

        <div className="space-y-3 mb-6">
          {[
            { k: 'Data parallelism / FSDP', rule: 'B / X > C / W  →  ≈ 2,500 tokens per GPU',
              why: 'Gradients are the same size whatever the batch, so you amortize them with tokens. Below roughly 2.5k tokens per chip the backward-pass collective dominates.' },
            { k: 'Tensor parallelism', rule: 'Y < F · W / C  →  ≈ 8-way, one NVLink domain',
              why: 'Batch cancels out of this one entirely. For LLaMA-3’s F = 28,672 the arithmetic gives ~11-way in-node, i.e. about 8 once you round to real hardware — which is exactly how big a node is. Spanning two nodes buys you roughly 16-way and no more.' },
            { k: 'Expert parallelism', rule: 'F > 8C / W_node',
              why: 'MoEs carry E times the weights for k times the FLOPs, which makes data parallelism painful. Sharding along the expert dimension fixes it if the model is wide enough; otherwise you are limited to about 2-node EP.' },
            { k: 'Pipeline parallelism', rule: 'crosses nodes cheaply; needs large batches',
              why: 'Only activations cross stage boundaries, so PP tolerates slow links well. The costs are code complexity (zero-bubble scheduling) and that it usually rules out ZeRO-3.' },
          ].map((r) => (
            <div key={r.k} className="glass rounded-xl p-4">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <div className="font-bold text-slate-900">{r.k}</div>
                <div className="font-mono text-xs font-bold text-accent">{r.rule}</div>
              </div>
              <p className="text-sm text-slate-600 mt-1.5 leading-relaxed">{r.why}</p>
            </div>
          ))}
        </div>

        <div className="glass rounded-xl p-5 border-l-4" style={{ borderLeftColor: C.accent }}>
          <h3 className="font-bold text-slate-800 mb-2">The one asymmetry to remember</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            <strong>Batch size fixes memory rooflines. Model width fixes network rooflines.</strong> The HBM rule and the
            data-parallel rule both have <i>B</i> in them, so batching your way out works. The tensor-parallel rule and
            the 2-chip <i>D/2</i> rule do not contain <i>B</i> at all — no amount of batching helps, and the only levers
            are a wider model, a faster link, or less sharding. Lesson 9 walks through each parallelism strategy in turn.
          </p>
        </div>
      </SectionCard>
    </div>
  );
}
