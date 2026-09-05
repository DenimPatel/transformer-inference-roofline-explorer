import React, { useState, useMemo } from 'react';
import {
  ComposedChart, Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ReferenceArea, Scatter, AreaChart, Area, Bar, BarChart, Cell,
} from 'recharts';
import {
  Info, Calculator, Cpu, Network, Zap, Activity, BookOpen, MemoryStick, Gauge, ArrowDownWideNarrow,
  Layers, Sigma, Eye, ShieldCheck, Sparkles,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { HARDWARE_PROFILES, effectiveMfu, onChipBwRatio } from '../lib/hardware';
import ConceptTag from './ui/ConceptTag';
import KvUsageExplain from './ui/KvUsageExplain';

// ---------------------------------------------------------------------------
// Shared teaching palette + formatters
// ---------------------------------------------------------------------------
const C = {
  compute: '#22c48b',   // mint
  memory: '#f25f7d',    // rose
  ridge: '#f43f5e',
  accent: '#5b7cfa',
  accentSoft: '#8aa0ff',
  sky: '#0ea5e9',
  amber: '#f59e0b',
  violet: '#8b5cf6',
  slate: '#94a3b8',
  rose: '#f25f7d',
  ink: '#0b1220',
};

function fmtNum(v: number, digits = 0) {
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: digits });
}

function fmtBytes(b: number): string {
  if (b >= 1e12) return `${(b / 1e12).toFixed(1)} TB`;
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} KB`;
  return `${b.toFixed(0)} B`;
}

function fmtFlops(f: number): string {
  if (f >= 1e15) return `${(f / 1e15).toFixed(2)} PFLOP/s`;
  if (f >= 1e12) return `${(f / 1e12).toFixed(1)} TFLOP/s`;
  return `${(f / 1e9).toFixed(1)} GFLOP/s`;
}

// Generic glass tooltip used by every chart.
function ChartTip({ active, payload, label, title, unit = '' }: any) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="glass-tooltip px-3 py-2 text-xs space-y-1">
      {label !== undefined && label !== '' && (
        <div className="font-semibold text-slate-800">
          {title ? `${title}: ` : ''}{typeof label === 'number' ? fmtNum(label) : label}
        </div>
      )}
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-slate-500 capitalize">{p.name}:</span>
          <span className="font-mono font-semibold text-slate-800">
            {typeof p.value === 'number' ? fmtNum(p.value) : p.value}{p.unit || unit}
          </span>
        </div>
      ))}
    </div>
  );
}

// Renders a "bound" badge with semantic colour.
function BoundBadge({ compute }: { compute: boolean }) {
  return (
    <span className={cn(
      'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold',
      compute ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700',
    )}>
      <span className={cn('w-2 h-2 rounded-full', compute ? 'bg-emerald-500' : 'bg-rose-500')} />
      {compute ? 'Compute-bound' : 'Bandwidth-bound'}
    </span>
  );
}

// Reusable labelled slider.
function Slider({ label, value, min, max, step = 1, onChange, format }: any) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <label className="font-medium text-slate-700">{label}</label>
        <span className="font-mono text-slate-900">{format ? format(value) : value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="glass-slider w-full"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main tab
// ---------------------------------------------------------------------------
export default function DeepDiveTab({ hardwareProfileId }: { hardwareProfileId: string }) {
  const hw = HARDWARE_PROFILES.find((h) => h.id === hardwareProfileId) || HARDWARE_PROFILES[0];
  const peakFlops = hw.tflops * 1e12;
  const peakBw = hw.memBw * 1e12;
  const hardwareIntensity = peakFlops / peakBw;
  const criticalBatch = hardwareIntensity; // bf16 matmul becomes compute-bound when B > ridge

  // On-chip scratchpad (VMEM on TPUs, SMEM/L2 on GPUs) is the second bandwidth
  // tier. Profiles that do not publish it fall back to the book's ~22x figure.
  const onChipRatio = onChipBwRatio(hw);
  const onChipBw = peakBw * onChipRatio;
  const onChipRidge = peakFlops / onChipBw;

  // Vector unit (VPU / CUDA cores). Its much lower peak gives a second, far
  // smaller ridge — which is the right yardstick for elementwise ops.
  const vectorFlops = (hw.vectorTflops ?? hw.tflops / 60) * 1e12;
  const vectorRidge = vectorFlops / peakBw;

  const mfu = effectiveMfu(hw);

  const sections = [
    { id: 'framework', label: 'Roofline', icon: Activity },
    { id: 'intensity', label: 'Intensity', icon: Sigma },
    { id: 'low-intensity', label: 'Second Ridge', icon: ShieldCheck },
    { id: 'matmul', label: 'Matmul', icon: Cpu },
    { id: 'prefill-gen', label: 'Prefill vs Gen', icon: Gauge },
    { id: 'kv-cache', label: 'KV Cache', icon: MemoryStick },
    { id: 'latency', label: 'Latency vs TP', icon: ArrowDownWideNarrow },
    { id: 'network', label: 'Network', icon: Network },
    { id: 'attention', label: 'Attn FLOPs', icon: Eye },
    { id: 'quant', label: 'Quantization', icon: Zap },
    { id: 'memory', label: 'Memory Hierarchy', icon: Layers },
    { id: 'moe', label: 'MoE', icon: Sparkles },
    { id: 'problems', label: 'Problems', icon: BookOpen },
  ];

  return (
    <div className="pb-16 max-w-6xl mx-auto mt-6 px-4">
      {/* ---- Hero ---- */}
      <section className="text-center mb-10">
        <div className="inline-flex items-center gap-2 glass-chip px-3 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4">
          <BookOpen className="w-3.5 h-3.5 text-accent" /> Interactive deep dive
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
          The Roofline, <span className="text-accent">Intuitively</span>
        </h1>
        <p className="text-slate-500 max-w-2xl mx-auto leading-relaxed">
          Every plot below is live — drag the sliders to feel where an operation sits.
          Move <strong>left of the ridge</strong> and you are <em>bandwidth-bound</em> (wasting FLOPs);
          move <strong>right</strong> and you saturate the silicon.
        </p>
        {/* hardware KPIs */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-8 max-w-3xl mx-auto">
          <HeroKpi icon={Cpu} label="Peak compute" value={fmtFlops(peakFlops)} sub={`${hw.id}`} />
          <HeroKpi icon={MemoryStick} label="Memory bandwidth" value={fmtBytes(peakBw) + '/s'} sub="HBM" />
          <HeroKpi icon={Sigma} label="Ridge point" value={fmtNum(hardwareIntensity)} sub="FLOPs / byte" />
          <HeroKpi icon={Activity} label="Critical batch" value={`≈${fmtNum(criticalBatch)}`} sub="bf16 tokens" />
        </div>
      </section>

      {/* ---- Section nav ---- */}
      <nav className="sticky top-0 z-30 -mx-2 px-2 py-3 mb-8 blur-[1px] backdrop-blur-md bg-[#eef1fb]/70 rounded-2xl">
        <div className="flex gap-1.5 overflow-x-auto custom-scrollbar py-1">
          {sections.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className="shrink-0 inline-flex items-center gap-1.5 glass-chip px-3 py-1.5 text-[11px] font-semibold text-slate-600 hover:text-accent hover:border-accent/40 transition-colors"
            >
              <s.icon className="w-3.5 h-3.5" /> {s.label}
            </a>
          ))}
        </div>
      </nav>

      {/* ---- 1. Framework ---- */}
      <SectionCard id="framework" icon={Calculator} color={C.accent} number="01"
        title="Formalized Mathematical Framework"
        tags={['overlap', 'roofline', 'arithmetic-intensity', 'two-bandwidth-roofline', 'mfu']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            The runtime of an algorithm on hardware is governed by exactly two clocks: how long the <strong>math</strong> takes
            (<i>T<sub>math</sub></i>) and how long the <strong>data movement</strong> takes (<i>T<sub>comms</sub></i>).
          </p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Time equations:</strong> <i>T<sub>math</sub> = FLOPs &divide; (FLOPs/s)</i>, &nbsp;<i>T<sub>comms</sub> = Bytes &divide; (Bytes/s)</i>.</li>
            <li><strong>Lower vs upper bound:</strong> <i>T<sub>lower</sub> = max(T<sub>math</sub>, T<sub>comms</sub>)</i> assumes perfect overlap;
              <i>T<sub>upper</sub> = T<sub>math</sub> + T<sub>comms</sub></i> assumes none. Because <i>a + b &le; 2·max(a,b)</i>,
              the two bounds are never more than <strong>2&times;</strong> apart — that factor of two is exactly the room
              available for overlap tricks like <em>collective matmuls</em>.</li>
            <li><strong>Arithmetic intensity:</strong> <i>I = FLOPs &divide; Bytes</i>. Compare it to the hardware ridge
              <i> I<sub>hw</sub> = peak FLOPs/s &divide; bandwidth</i> to know which bound wins.</li>
          </ul>
          <p>
            One warning before you trust any of these numbers: <strong>&ldquo;bandwidth&rdquo; is not one number.</strong>
            A tensor may move within a chip, between chips in the same tightly-coupled domain, or across the datacenter
            network — three very different byte/s figures, so three different ridges. The right roofline is the one for
            the link the operation actually uses.
          </p>
        </div>

        <div className="grid sm:grid-cols-3 gap-3 mb-6">
          {[
            { tier: 'Within a chip', link: 'HBM ↔ compute cores', bw: `${hw.memBw.toFixed(2)} TB/s`,
              ridge: fmtNum(hardwareIntensity), note: 'The default roofline. Sets the critical batch.' },
            { tier: 'Within a domain', link: hw.linkBwGBs ? 'ICI / NVLink' : 'ICI / NVLink (not published)',
              bw: hw.linkBwGBs ? `${fmtNum(hw.linkBwGBs)} GB/s` : '—',
              ridge: hw.linkBwGBs ? fmtNum(peakFlops / (hw.linkBwGBs * 1e9)) : '—',
              note: 'Sharded matmuls. Threshold depends on D, not B.' },
            { tier: 'Beyond the domain', link: hw.scaleOutBwGBs ? 'DCN / InfiniBand' : 'DCN / InfiniBand (not published)',
              bw: hw.scaleOutBwGBs ? `${fmtNum(hw.scaleOutBwGBs, 2)} GB/s` : '—',
              ridge: hw.scaleOutBwGBs ? fmtNum(peakFlops / (hw.scaleOutBwGBs * 1e9)) : '—',
              note: 'Data parallelism across pods. Two orders of magnitude slower.' },
          ].map((t) => (
            <div key={t.tier} className="glass rounded-xl p-4">
              <div className="text-[11px] uppercase tracking-wide text-slate-400">{t.tier}</div>
              <div className="font-semibold text-slate-800 text-sm mt-0.5">{t.link}</div>
              <div className="font-mono text-lg font-bold text-slate-900 mt-2">{t.bw}</div>
              <div className="text-xs text-slate-500">ridge ≈ {t.ridge} FLOPs/byte</div>
              <div className="text-[11px] text-slate-400 mt-2 leading-snug">{t.note}</div>
            </div>
          ))}
        </div>

        <div className="glass rounded-xl p-4 mb-6 border-l-4" style={{ borderLeftColor: C.amber }}>
          <div className="font-bold text-slate-800 text-sm mb-1">Peak FLOPs is a marketing number</div>
          <p className="text-sm text-slate-600">
            No real kernel hits the datasheet figure. TPUs typically reach ~<strong>95%</strong> of peak in normal use;
            H100- and B200-class GPUs land around <strong>80&ndash;85%</strong>, because so much of a GPU&rsquo;s runtime
            behaviour (cache thrashing, uncoalesced loads, kernel launch overhead) sits outside the compiler&rsquo;s
            control. This site models <code className="bg-slate-100 px-1 rounded">{hw.id}</code> at{' '}
            <strong>{Math.round(mfu * 100)}% MFU</strong>, so its realistic ceiling is{' '}
            <strong>{fmtFlops(peakFlops * mfu)}</strong>, not {fmtFlops(peakFlops)}. Watch out for GPU spec sheets that
            quote tensor-core FLOPs <em>&ldquo;with sparsity&rdquo;</em> — that number is 2&times; the value you can
            actually reach on dense weights.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-5 mb-6">
          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-3 flex items-center">
              <Activity className="w-4 h-4 mr-2 text-accent" /> How Much Can Overlap Help?
            </h3>
            <OverlapBoundsChart />
            <p className="text-sm text-slate-500 mt-3">
              When <i>T<sub>math</sub> = T<sub>comms</sub></i> the gap between the bounds is largest — the sweet spot
              for scheduling compute behind communication.
            </p>
          </div>

          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-3 flex items-center">
              <Cpu className="w-4 h-4 mr-2 text-accent" /> Achievable FLOPs &amp; the Ridge
            </h3>
            <div className="space-y-3 text-sm text-slate-600">
              <div className="flex justify-between items-center p-3 glass rounded-lg">
                <span>Compute roof (peak)</span>
                <span className="font-mono font-bold text-emerald-600">{fmtFlops(peakFlops)}</span>
              </div>
              <div className="flex justify-between items-center p-3 glass rounded-lg">
                <span>Bandwidth roof slope</span>
                <span className="font-mono text-slate-700">BW &times; I = {fmtFlops(peakBw)} per 1 FLOP/B</span>
              </div>
              <div className="flex justify-between items-center p-3 glass rounded-lg">
                <span>Ridge crossover</span>
                <span className="font-mono font-bold text-rose-600">I<sub>hw</sub> ≈ {fmtNum(hardwareIntensity)}</span>
              </div>
              <p className="text-xs text-slate-400">
                Your <code className="bg-slate-100 px-1 rounded">{hw.id}</code> reaches its peak FLOPs/s only at an
                intensity of ≈{fmtNum(hardwareIntensity)} FLOPs/byte. Below that, bandwidth drags you down the slope.
              </p>
            </div>
          </div>
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-2 flex items-center">
            <Activity className="w-4 h-4 mr-2 text-accent" /> The Live Roofline
          </h3>
          <p className="text-sm text-slate-500 mb-4">
            Log&ndash;log throughput vs intensity. Below the ridge you ride the <span className="text-sky-600 font-medium">bandwidth slope</span>;
            above it you sit on the flat <span className="text-emerald-600 font-medium">compute roof</span>. The dots are real operations.
          </p>
          <div className="h-[380px] w-full">
            <RooflineChart peakFlops={peakFlops} peakBw={peakBw} ridge={hardwareIntensity} showOps />
          </div>

          <h3 className="font-bold text-slate-800 mt-8 mb-2 flex items-center">
            <Layers className="w-4 h-4 mr-2 text-accent" /> Two Bandwidths, Three Regions
          </h3>
          <p className="text-sm text-slate-500 mb-4">
            There are two ways to move an op onto the compute roof: raise its intensity (push right) or feed it from a
            faster memory (raise the slope). Comparing HBM against the on-chip scratchpad —{' '}
            <strong>{onChipRatio.toFixed(0)}&times;</strong> the bandwidth on this chip — splits the plane into three:
          </p>
          <div className="h-[360px] w-full">
            <RooflineChart peakFlops={peakFlops} peakBw={peakBw} ridge={hardwareIntensity}
              peakBw2={onChipBw} bwLabel="HBM" bw2Label="on-chip" />
          </div>
          <div className="grid sm:grid-cols-3 gap-3 mt-4 text-xs">
            <div className="glass rounded-lg p-3 border-l-4" style={{ borderLeftColor: C.rose }}>
              <div className="font-semibold text-slate-700">Red — I &lt; {fmtNum(onChipRidge, 1)}</div>
              <div className="text-slate-500 mt-1">Bandwidth-bound at <em>both</em> bandwidths. Faster memory does not
                save you; the op itself has to change.</div>
            </div>
            <div className="glass rounded-lg p-3 border-l-4" style={{ borderLeftColor: C.amber }}>
              <div className="font-semibold text-slate-700">Amber — {fmtNum(onChipRidge, 1)} &lt; I &lt; {fmtNum(hardwareIntensity)}</div>
              <div className="text-slate-500 mt-1">Bandwidth-bound from HBM, compute-bound from on-chip memory. This is
                exactly the band where tiling and prefetching pay off.</div>
            </div>
            <div className="glass rounded-lg p-3 border-l-4" style={{ borderLeftColor: C.compute }}>
              <div className="font-semibold text-slate-700">Green — I &gt; {fmtNum(hardwareIntensity)}</div>
              <div className="text-slate-500 mt-1">Compute-bound whatever you do. More bandwidth buys nothing; only more
                FLOP/s helps.</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* ---- 2. Arithmetic Intensity ---- */}
      <SectionCard id="intensity" icon={Info} color={C.sky} number="02"
        title="Arithmetic Intensity &amp; the Ridge Point"
        tags={['arithmetic-intensity', 'ridge-point', 'critical-batch']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Arithmetic intensity, <i>I = FLOPs &divide; bytes</i>, is the single most useful number in roofline analysis:
            it tells you how much math you get out of every byte you drag through memory, and therefore which side of the
            ridge the operation sits on. A bf16 matmul has <i>I ≈ B</i> — its per-replica <strong>token batch</strong> —
            so the famous rule falls out: <em>compute-bound iff <i>B &gt; I<sub>hw</sub></i></em>.
          </p>
        </div>
        <ArithmeticIntensitySection hardwareIntensity={hardwareIntensity} />
      </SectionCard>

      {/* ---- 3. Low-intensity ops and the second ridge ---- */}
      <SectionCard id="low-intensity" icon={ShieldCheck} color={C.rose} number="03"
        title="Low-Intensity Ops &amp; the Second Ridge"
        tags={['dot-product-intensity', 'vector-unit-ridge', 'ridge-point', 'memory-hierarchy']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Not every operation has a batch size to grow. Some are stuck at a low intensity <em>no matter what you
            do</em>, and the roofline tells you so before you write a line of code. The canonical case is the
            <strong> dot product</strong>.
          </p>
          <p>
            To compute <code>x · y</code> for <i>bf16[N]</i> vectors we load <i>2N</i> bytes for each input, perform
            <i> N</i> multiplies and <i>N&minus;1</i> adds, and write <i>2</i> bytes back:
          </p>
          <div className="glass rounded-xl p-4 font-mono text-sm text-slate-700 text-center">
            I(dot) = (N + N &minus; 1) / (2N + 2N + 2) = (2N &minus; 1) / (4N + 2) &nbsp;&rarr;&nbsp; <strong className="text-rose-600">1/2</strong>
          </div>
          <p>
            The <i>N</i>s cancel. Doubling the vector doubles both the math and the traffic, so the intensity is pinned
            at half a FLOP per byte forever. There is no batch knob, no tiling trick, no fusion that changes the
            asymptote — the dot product is bandwidth-bound on every accelerator ever built.
          </p>
        </div>

        <div className="glass rounded-xl p-5 mb-6">
          <h3 className="font-bold text-slate-800 mb-2 flex items-center">
            <Sigma className="w-4 h-4 mr-2 text-accent" /> Which Ridge Should You Compare Against?
          </h3>
          <p className="text-sm text-slate-500 mb-4">
            Here is the subtlety that trips people up. The famous ridge of ≈240 belongs to the <strong>matrix
            unit</strong> — the MXU on a TPU, the tensor cores on a GPU. A dot product never touches it. Elementwise and
            reduction work runs on the <strong>vector unit</strong> (VPU on TPUs, CUDA cores on GPUs), which has far
            fewer FLOP/s against the <em>same</em> HBM bandwidth — and therefore a far smaller ridge.
          </p>
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="glass rounded-lg p-4">
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Matrix unit (MXU / tensor cores)</div>
              <div className="font-mono text-lg font-bold text-slate-900">{fmtFlops(peakFlops)}</div>
              <div className="text-xs text-slate-500 mt-1">ridge ≈ <strong className="text-rose-600">{fmtNum(hardwareIntensity)}</strong> FLOPs/byte</div>
            </div>
            <div className="glass rounded-lg p-4">
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">Vector unit (VPU / CUDA cores)</div>
              <div className="font-mono text-lg font-bold text-slate-900">{fmtFlops(vectorFlops)}</div>
              <div className="text-xs text-slate-500 mt-1">ridge ≈ <strong className="text-amber-600">{fmtNum(vectorRidge, 1)}</strong> FLOPs/byte</div>
            </div>
          </div>
          <p className="text-sm text-slate-500 mt-4">
            On a TPU v5p the VPU manages roughly <code>7e12</code> FLOP/s per core, so its critical intensity is about
            <strong> 3</strong>, not 240. That is a <em>much</em> gentler bar — and the dot product still fails it, since
            &frac12; &lt; 3. The conclusion survives either yardstick, but picking the wrong ridge will make you
            mis-diagnose every softmax, layernorm and residual add in your model.
          </p>
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-2 flex items-center">
            <Activity className="w-4 h-4 mr-2 text-accent" /> Two Ridges, One Chip
          </h3>
          <p className="text-sm text-slate-500 mb-4">
            The dashed green roof is the vector unit&rsquo;s. Ops between the two ridges saturate the VPU but would be
            bandwidth-bound if you tried to run them on the MXU.
          </p>
          <div className="h-[340px] w-full">
            <RooflineChart peakFlops={peakFlops} peakBw={peakBw} ridge={hardwareIntensity}
              peakBw2={peakBw * (vectorFlops / peakFlops)} bwLabel="Matrix unit" bw2Label="Vector unit" />
          </div>
          <div className="grid sm:grid-cols-3 gap-3 mt-4 text-xs">
            {[
              { op: 'Dot product / axpy', i: '0.5', verdict: 'Bandwidth-bound everywhere' },
              { op: 'Softmax, layernorm, GELU', i: '~1–2', verdict: 'Bandwidth-bound; fuse them' },
              { op: 'Matmul, B = 1024', i: '~1024', verdict: 'Compute-bound on the MXU' },
            ].map((r) => (
              <div key={r.op} className="glass rounded-lg p-3">
                <div className="font-semibold text-slate-700">{r.op}</div>
                <div className="font-mono text-slate-500">I ≈ {r.i}</div>
                <div className="text-[11px] text-slate-400 mt-1">{r.verdict}</div>
              </div>
            ))}
          </div>
          <p className="text-sm text-slate-500 mt-4">
            <strong>Why this matters in practice:</strong> because low-intensity ops can never be compute-bound, the only
            way to speed them up is to <em>stop moving the bytes</em> — fuse them into the neighbouring matmul so the
            intermediate never round-trips to HBM. That is the entire argument for kernel fusion, and for FlashAttention.
          </p>
        </div>
      </SectionCard>

      {/* ---- 3. Matmul Math ---- */}
      <SectionCard id="matmul" icon={Cpu} color={C.compute} number="04"
        title="Matrix Multiplication Math"
        tags={['matmul-intensity', 'critical-batch']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Matmul is the fundamental operation of deep learning. For <i>X[B,D] &middot; Y[D,F] &rarr; Z[B,F]</i> the
            math is <i>2BDF</i> FLOPs, the traffic is <i>2BD + 2DF + 2BF</i> bytes, and the intensity
            is <i>I = 2BDF&divide;(2BD + 2DF + 2BF) ≈ B</i> when <i>B</i> is small relative to <i>D, F</i>.
          </p>
        </div>

        <div className="glass rounded-xl p-4 mb-6 border-l-4" style={{ borderLeftColor: C.violet }}>
          <div className="font-bold text-slate-800 text-sm mb-1">B is tokens, not sequences — and it is per-replica</div>
          <p className="text-sm text-slate-600">
            Almost every roofline here depends purely on the <em>number of tokens</em> in the matmul, whether or not they
            belong to the same sequence. Say you train with 512 sequences of 4096 tokens on 2048 GPUs. Your global batch
            is <code>512 &times; 4096 = 2M</code> tokens, but each chip sees <code>2M / 2048 ≈ 1024</code> tokens —
            comfortably past the ≈{fmtNum(criticalBatch)} ridge, so you are compute-bound. Quote that same batch in
            sequences (512/2048 &lt; 1) and you would have concluded the exact opposite.
          </p>
          <p className="text-sm text-slate-600 mt-2">
            The reason the relevant figure is <strong>per-replica</strong> rather than global is that sharding a matmul
            across more chips scales the available FLOP/s <em>and</em> the available HBM bandwidth by the same factor.
            The ratio between them — the ridge — does not move. So <i>B<sub>crit</sub></i> applies once per
            independent copy of the weights, however many chips that copy is spread over.
          </p>
        </div>

        <MatmulInteractiveSection hardwareIntensity={hardwareIntensity} />
      </SectionCard>

      {/* ---- 4. Prefill vs Generation ---- */}
      <SectionCard id="prefill-gen" icon={Activity} color={C.accentSoft} number="05"
        title="Prefill vs Generation: Why Inference Flips the Roofline"
        tags={['prefill', 'generation', 'attention-intensity']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Transformer inference is really <em>two different workloads</em>. Prefill reuses weights across <i>B&middot;T</i>
            tokens and has attention intensity <i>≈ T/2</i> — <strong>almost always compute-bound</strong>. Generation runs
            one token at a time (<i>T = 1</i>): weights are streamed fresh every step and attention intensity collapses to
            <i> ≈ 1</i> — <strong>almost always memory-bound</strong>. The only lever that pushes decode back toward the ridge
            is <em>batching many requests together</em>.
          </p>
        </div>
        <div className="grid lg:grid-cols-2 gap-5">
          <PrefillGenerationSection hw={hw} peakFlops={peakFlops} peakBw={peakBw} hardwareIntensity={hardwareIntensity} />
          <AttentionIntensitySection hardwareIntensity={hardwareIntensity} />
        </div>
        <div className="mt-6">
          <h3 className="font-bold text-slate-800 mb-1">How each K and V is actually used</h3>
          <p className="text-sm text-slate-500 mb-4">
            This is <em>why</em> the KV cache matters at all: at every later step, the cached keys select{' '}
            <em>where</em> to attend and the cached values carry <em>what</em> gets blended forward.
          </p>
          <KvUsageExplain />
        </div>
      </SectionCard>

      {/* ---- 5. KV Cache (NEW) ---- */}
      <SectionCard id="kv-cache" icon={MemoryStick} color={C.violet} number="06"
        title="The KV Cache: Where Inference Memory Goes"
        tags={['kv-cache', 'memory-bound', 'generation']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            During generation the dominant cost is not the weights — it is the <strong>KV cache</strong>: every token&rsquo;s cached
            key/value projections. It grows linearly with <em>context length &times; batch</em>, independent of model capacity once
            you fix the head count. That is why long contexts &times; large batches quickly balloon into terabytes and why
            <strong>GQA</strong>, <strong>quantization</strong>, and <strong>paged attention</strong> matter so much.
          </p>
          <p className="text-sm text-slate-500">
            This cache isn&rsquo;t just storage — its values get <em>used</em> on every later forward pass: the cached
            keys are scored against the new query and the cached values are blended by those weights (see{" "}
            <em>Prefill vs Generation</em>). Formula:{" "}
            <code className="bg-slate-100 px-1.5 rounded font-mono">KV = 2 &middot; bytes &middot; H &middot; K &middot; L &middot; T &middot; B</code>
            &nbsp;(keys and values, head dim, KV heads, layers, context, batch).
          </p>
        </div>
        <KVCacheSection hardwareIntensity={hardwareIntensity} hw={hw} />
      </SectionCard>

      {/* ---- 6. Latency vs Throughput (NEW) ---- */}
      <SectionCard id="latency" icon={ArrowDownWideNarrow} color={C.amber} number="07"
        title="Latency vs Throughput: The Pareto Tradeoff"
        tags={['latency-throughput', 'generation', 'critical-batch']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Bigger batches raise throughput (tokens/s) but each step must load more KV cache, so per-request latency grows too.
            You are walking a <strong>Pareto frontier</strong>. The theoretical ceiling is
            <i> max tokens/s = (B&middot;BW) / (B&middot;KV<sub>seq</sub> + params)</i> — throughput flattens once memory traffic
            saturates bandwidth, right around the critical batch. Adding chips is what moves the frontier out.
          </p>
        </div>
        <LatencyThroughputSection hw={hw} peakFlops={peakFlops} peakBw={peakBw} hardwareIntensity={hardwareIntensity} />
      </SectionCard>

      {/* ---- 7. Network ---- */}
      <SectionCard id="network" icon={Network} color={C.violet} number="08"
        title="Inter-Chip Network Rooflines"
        tags={['network-roofline', 'model-parallelism', 'ici-topology', 'nvlink-domain']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Sharding moves the bottleneck from HBM to the inter-chip fabric. Two chips splitting the contracting
            <i>D</i> dimension each trade <i>2BF</i> bytes of partial sums for <i>BDF</i> FLOPs, so the inter-chip intensity
            is <i>I = D/2</i> — it depends on the <strong>model dimension D, not the batch B</strong>. Bigger models are
            easier to shard; bigger batches don&rsquo;t help the fabric.
          </p>
        </div>
        <NetworkRooflineInteractiveSection hw={hw} />
      </SectionCard>

      {/* ---- 8. Attention FLOPs crossover (NEW) ---- */}
      <SectionCard id="attention" icon={Eye} color={C.sky} number="09"
        title="When Does Attention Dominate Compute?"
        tags={['attention-flops', 'attention-intensity']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Everyone says &ldquo;attention is quadratic&rdquo; — but its constant is tiny. Attention&rsquo;s share of layer FLOPs is
            ≈ <i>T / 8D</i>, so it only overtakes the MLP once the context exceeds ≈ <i>8&middot;D</i> tokens (~64k for a
            <i>D</i>&asymp;8k model). For typical contexts you are almost always <strong>matmul-limited</strong> in FLOPs —
            but attention still dominates <em>memory</em> via the KV cache.
          </p>
        </div>
        <AttentionFlopsSection />
      </SectionCard>

      {/* ---- 9. Quantization ---- */}
      <SectionCard id="quant" icon={Zap} color={C.amber} number="10"
        title="Quantization &amp; Mixed Precision"
        tags={['quantization', 'critical-batch']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            Quantizing <em>weights</em> trims the bytes-per-parameter and drops the critical batch; quantizing the
            <em>math</em> doubles peak FLOPs/s and pushes it back up. Governed by
            <i> B<sub>crit</sub> = &beta; &middot; I<sub>hw</sub></i> where
            <i> &beta; = bits/param &divide; bits/activation</i>. The sweet spot is <strong>int8 weights + bf16 compute</strong> —
            B_crit halves to ~120 with no quality sacrifice.
          </p>
        </div>
        <QuantizationInteractiveSection hardwareIntensity={hardwareIntensity} hw={hw} />
      </SectionCard>

      {/* ---- 10. Memory hierarchy & tiling (NEW) ---- */}
      <SectionCard id="memory" icon={Layers} color={C.sky} number="11"
        title="Memory Hierarchy &amp; Tiling: VMEM Changes Everything"
        tags={['memory-hierarchy', 'tiling', 'matmul-intensity']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            HBM is big but slow. On-chip scratchpad (VMEM on TPUs, SMEM on GPUs) is tiny but ~22&times; the bandwidth — so an op
            fed from VMEM only needs intensity ~10&ndash;20 to hit peak, versus ~240 from HBM. Big matmuls are cut into tiles
            that fit VMEM and re-loaded, which is why effective intensity drops to
            ≈ <i> bm&middot;bn/(bm+bn)</i>: <em>tuning the tile size is really tuning arithmetic intensity</em>.
          </p>
        </div>
        <MemoryHierarchySection hw={hw} hardwareIntensity={hardwareIntensity} onChipRatio={onChipRatio} onChipRidge={onChipRidge} />
      </SectionCard>

      {/* ---- 11. MoE (NEW) ---- */}
      <SectionCard id="moe" icon={Sparkles} color={C.compute} number="12"
        title="Mixture-of-Experts: A Hidden Batch Requirement"
        tags={['moe', 'critical-batch']}>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6">
          <p>
            MoE multiplies <em>stored</em> parameters by <i>E</i> but only reads <i>k</i> per token, so you need
            <i>E/k</i> &times; more concurrent tokens to saturate parameter bandwidth:
            <strong> compute-bound requires <i>B &gt; 120 &middot; E/k</i></strong>. For DeepSeek&nbsp;v3 (E=256, k=8) that is a
            startling <i>B &gt; 3,840</i> tokens — an enormous serving batch.
          </p>
        </div>
        <MoeSection />
      </SectionCard>

      {/* ---- 12. Worked Problems ---- */}
      <SectionCard id="problems" icon={BookOpen} color={C.rose} number="13"
        title="Worked Problems"
        tags={['matmul-intensity', 'quantization', 'arithmetic-intensity']}>
        <WorkedProblemsSection hw={hw} peakFlops={peakFlops} peakBw={peakBw} hardwareIntensity={hardwareIntensity} />
        <p className="text-sm text-slate-500 mt-6">
          Adapted from the scaling-book (Part 1, &ldquo;A Few Problems to Work&rdquo;). Reference copy stored in{" "}
          <code className="text-xs bg-slate-100 px-1 rounded">reference/scaling-book/roofline.md</code>.
        </p>
      </SectionCard>

      <p className="text-center text-xs text-slate-400 mt-12">
        Every curve is computed live from the formulas in the scaling-book reference material — no static images,
        so you can always drag, compare, and <em>feel</em> the roofline.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small layout pieces
// ---------------------------------------------------------------------------
function SectionCard({ id, icon: IconCmp, color, number, title, tags, children }: any) {
  return (
    <section id={id} className="glass-card p-6 sm:p-8 mb-8 scroll-mt-24">
      <div className="flex items-start justify-between gap-4 mb-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900">
          <span className="inline-flex items-center justify-center w-9 h-9 rounded-xl mr-3 text-white shrink-0"
            style={{ background: color }}>
            <IconCmp className="w-5 h-5" />
          </span>
          <span className="text-3xl font-black mr-3 opacity-20" style={{ color }}>{number}</span>
          {title}
        </h2>
      </div>
      <div className="flex flex-wrap gap-2 mb-5">
        {tags.map((t: string) => (
          <span key={t}><ConceptTag id={t} /></span>
        ))}
      </div>
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
// 1. Overlap bounds
// ---------------------------------------------------------------------------
function OverlapBoundsChart() {
  const data = useMemo(() => {
    const rows: any[] = [];
    for (let r = 0.15; r <= 6.5; r += 0.2) {
      rows.push({
        ratio: r,
        noOverlap: r + 1,
        perfectOverlap: Math.max(r, 1),
        saved: Math.round(((r + 1 - Math.max(r, 1)) / (r + 1)) * 100),
      });
    }
    return rows;
  }, []);

  return (
    <div>
      <div className="h-52 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="overlapFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="upperFill2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
            <XAxis dataKey="ratio" type="number" scale="log" domain={[0.1, 10]}
              tickFormatter={(v: any) => `${Number(v).toFixed(1)}`} />
            <YAxis />
            <Tooltip content={<ChartTip label="T_math/T_comms" />} />
            <Area type="monotone" dataKey="noOverlap" name="Upper (no overlap)" stroke={C.ridge} strokeWidth={2.5} fill="url(#upperFill2)" />
            <Area type="monotone" dataKey="perfectOverlap" name="Lower (perfect overlap)" stroke="#2563eb" strokeWidth={2.5} fill="url(#overlapFill)" />
            <ReferenceLine x={1} stroke="#475569" strokeDasharray="4 4" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
        {data.filter((d) => [0.5, 1, 2, 5].includes(Math.round(d.ratio))).map((d) => (
          <div key={d.ratio} className="glass rounded-md p-2 text-center">
            <div className="text-slate-400">ratio {d.ratio}</div>
            <div className="font-mono font-bold text-emerald-600">{d.saved}% saved</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The teaching Roofline (bandwidth slope + compute roof + operations)
// ---------------------------------------------------------------------------
function RooflineChart({
  peakFlops, peakBw, ridge, showOps = false, peakBw2, bwLabel = 'BW1 (HBM)', bw2Label = 'BW2 (on-chip)',
}: {
  peakFlops: number; peakBw: number; ridge: number; showOps?: boolean;
  /** Optional faster second bandwidth. When present the plot shows the scaling
   *  book's three regions: bandwidth-bound at both, bandwidth-bound only at the
   *  slower one, and compute-bound at both. */
  peakBw2?: number;
  bwLabel?: string;
  bw2Label?: string;
}) {
  // With two bandwidths the *faster* link reaches the compute roof at a smaller
  // intensity, so it owns the left (green) edge of the ambiguous band.
  const ridge2 = peakBw2 ? peakFlops / peakBw2 : undefined;
  const loRidge = ridge2 !== undefined ? Math.min(ridge, ridge2) : ridge;
  const hiRidge = ridge2 !== undefined ? Math.max(ridge, ridge2) : ridge;

  const decadeTicks = [0.1, 1, 10, 100, 1000, 10000, 100000];
  const minI = 0.05;
  const maxI = 200000;

  const data = useMemo(() => {
    const pts: any[] = [];
    for (let i = Math.log10(minI); i <= Math.log10(maxI); i += 0.02) {
      const intensity = Math.pow(10, i);
      const row: any = { intensity, achievable: Math.min(peakBw * intensity, peakFlops) };
      if (peakBw2) row.achievable2 = Math.min(peakBw2 * intensity, peakFlops);
      pts.push(row);
    }
    return pts;
  }, [peakFlops, peakBw, peakBw2]);

  const ops = useMemo(() => {
    if (!showOps) return [];
    const mk = (name: string, intensity: number, color: string) => ({
      name, intensity, achieved: Math.min(peakBw * intensity, peakFlops), color,
      bound: intensity >= ridge ? 'Compute-bound' : 'Bandwidth-bound',
    });
    return [
      mk('Dot product (VPU)', 0.5, C.ridge),
      mk('Attn – generate (~1)', 1, C.amber),
      mk('Matmul, B=64', 64, C.accent),
      mk('Attn – prefill T=512', 256, C.compute),
      mk('Matmul, B=1024', 1024, C.violet),
    ];
  }, [showOps, peakBw, peakFlops, ridge]);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
      <defs>
        <linearGradient id="bandwidthRoof" x1="0" y1="0" x2="0" y2="1">
          <stop offset="5%" stopColor="#f25f7d" stopOpacity={0.28} />
          <stop offset="95%" stopColor="#f25f7d" stopOpacity={0.02} />
        </linearGradient>
        <linearGradient id="computeRoof" x1="0" y1="0" x2="0" y2="1">
          <stop offset="5%" stopColor="#22c48b" stopOpacity={0.28} />
          <stop offset="95%" stopColor="#22c48b" stopOpacity={0.02} />
        </linearGradient>
      </defs>
      <CartesianGrid strokeDasharray="3 3" opacity={0.35} />
      <XAxis dataKey="intensity" scale="log" domain={[minI, maxI]} type="number" allowDataOverflow
        ticks={decadeTicks}
        tickFormatter={(v: any) => (Number(v) < 1 ? Number(v).toFixed(1) : fmtNum(Number(v)))}
        label={{ value: 'Arithmetic Intensity (FLOPs / Byte) — log', position: 'insideBottom', offset: -6, fontSize: 11, fill: C.slate }} />
      <YAxis scale="log" domain={['dataMin', 'dataMax']} type="number" width={58}
        tickFormatter={(v: any) => (v / 1e12) >= 1 ? `${(v / 1e12).toFixed(0)}T` : `${(v / 1e9).toFixed(0)}G`}
        label={{ value: 'Throughput', angle: -90, position: 'insideLeft', fontSize: 11, fill: C.slate }} />
      <Tooltip
        content={<ChartTip />}
        labelFormatter={(v: any) => `Intensity ${Number(v) < 1 ? Number(v).toFixed(2) : fmtNum(v)}`}
      />
      {(showOps || peakBw2) && <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '12px' }} />}

      {/* Bandwidth-bound at every bandwidth on offer (red). */}
      <ReferenceArea {...({ x1: minI, x2: loRidge, fill: '#f25f7d', fillOpacity: peakBw2 ? 0.16 : 0.07, strokeOpacity: 0 } as any)} />
      {/* Bandwidth-bound only at the slower bandwidth (amber) — buying bandwidth helps here. */}
      {ridge2 !== undefined && (
        <ReferenceArea {...({ x1: loRidge, x2: hiRidge, fill: '#f59e0b', fillOpacity: 0.18, strokeOpacity: 0 } as any)} />
      )}
      {/* Compute-bound at every bandwidth (green). */}
      <ReferenceArea {...({ x1: hiRidge, x2: maxI, fill: '#22c48b', fillOpacity: peakBw2 ? 0.16 : 0.07, strokeOpacity: 0 } as any)} />

      <ReferenceLine x={ridge} stroke={C.ridge} strokeDasharray="5 5" strokeWidth={1.5}
        label={{ position: 'top', value: `ridge ≈ ${fmtNum(ridge)}`, fill: C.ridge, fontSize: 11, fontWeight: 700 }} />
      {ridge2 !== undefined && (
        <ReferenceLine x={ridge2} stroke={C.compute} strokeDasharray="3 6" strokeWidth={1.5}
          label={{ position: 'top', value: `ridge₂ ≈ ${fmtNum(ridge2)}`, fill: C.compute, fontSize: 11, fontWeight: 700 }} />
      )}

      <Line type="monotone" dataKey="achievable" name={peakBw2 ? `Roof at ${bwLabel}` : 'Roof (achievable)'}
        stroke={C.sky} strokeWidth={3} dot={false} strokeLinecap="round" />
      {peakBw2 && (
        <Line type="monotone" dataKey="achievable2" name={`Roof at ${bw2Label}`}
          stroke={C.compute} strokeWidth={2.5} strokeDasharray="6 4" dot={false} strokeLinecap="round" />
      )}

      {showOps && (
        <Scatter data={ops} dataKey="achieved" name="Operations" shape={(p: any) => {
          const d = ops.find((o) => o.name === p.payload.name) || p.payload;
          return (
            <g transform={`translate(${p.cx},${p.cy})`}>
              <circle r={7} fill={d.color} fillOpacity={0.15} stroke={d.color} strokeWidth={2} />
              <circle r={3} fill={d.color} />
            </g>
          );
        }} />
      )}
      {showOps && ops.map((o) => (
        <Scatter key={o.name} data={[o]} dataKey="achieved" fill={o.color} shape={(p: any) =>
          <g transform={`translate(${p.cx},${p.cy})`}>
            <circle r={7} fill={o.color} fillOpacity={0.15} stroke={o.color} strokeWidth={2} />
            <circle r={3} fill={o.color} />
            <text x={-14} y={-9} textAnchor="start" fontSize={10} fill={C.slate} fontWeight={600}>{o.name}</text>
            <text x={-14} y={3} textAnchor="start" fontSize={10} fill={o.color} fontWeight={700}>{o.bound}</text>
          </g>
        } />
      ))}
    </ComposedChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// 2. Arithmetic intensity interactive
// ---------------------------------------------------------------------------
function ArithmeticIntensitySection({ hardwareIntensity }: { hardwareIntensity: number }) {
  const [B, setB] = useState(96);
  const [D, setD] = useState(4096);

  const intensity = (2 * B * D * D) / (2 * B * D + 2 * D * D + 2 * B * D);
  const isComputeBound = intensity > hardwareIntensity;

  const data = useMemo(() => {
    const pts: any[] = [];
    for (let batch = 1; batch <= 2048; batch *= 1.1) {
      const b = Math.round(batch);
      const i = (2 * b * D * D) / (2 * b * D + 2 * D * D + 2 * b * D);
      pts.push({ batch: b, intensity: i, approx: b });
    }
    return pts;
  }, [D]);

  return (
    <div className="glass rounded-xl p-5">
      <h3 className="font-bold text-slate-800 mb-1">Interactive: Where Does This Matmul Sit?</h3>
      <p className="text-sm text-slate-500 mb-4">
        Push the token batch <i>B</i> past the red ridge line and watch the operation cross from bandwidth- to compute-bound.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-5">
          <Slider label="Per-replica token batch (B)" value={B} min={1} max={2048} onChange={setB} />
          <Slider label="Hidden dim (D = F)" value={D} min={1024} max={16384} step={256} onChange={setD} />

          <div className="grid grid-cols-2 gap-2 text-sm mt-2">
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Exact intensity</div>
              <div className="font-mono font-bold text-slate-900 text-lg">{intensity.toFixed(1)}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">≈ B simplification</div>
              <div className="font-mono text-slate-900 text-lg">{B}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Hardware ridge</div>
              <div className="font-mono text-slate-900 text-lg">{hardwareIntensity.toFixed(1)}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Critical batch B_crit</div>
              <div className="font-mono text-slate-900 text-lg">≈{Math.round(hardwareIntensity)}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <BoundBadge compute={isComputeBound} />
            <span className="text-xs text-slate-400">
              {isComputeBound
                ? 'You are using essentially all the FLOPs/s.'
                : 'FLOPs are being wasted waiting on memory.'}
            </span>
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.35} />
              <XAxis dataKey="batch" type="number" scale="log" domain={[1, 2048]} allowDataOverflow
                ticks={[1, 4, 16, 64, 256, 1024]}
                tickFormatter={(v: any) => `${v}`} label={{ value: 'Token batch (log)', position: 'insideBottom', offset: -8, fontSize: 10, fill: C.slate }} />
              <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
              <Tooltip content={<ChartTip />} />
              <Line type="monotone" dataKey="approx" name="≈ B" stroke={C.slate} strokeWidth={2} strokeDasharray="4 4" dot={false} />
              <Line type="monotone" dataKey="intensity" name="Exact" stroke={C.sky} strokeWidth={3} dot={false} />
              <ReferenceLine y={hardwareIntensity} stroke={C.ridge} strokeDasharray="3 3"
                label={{ position: 'top', value: 'Ridge', fill: C.ridge, fontSize: 10, fontWeight: 700 }} />
              <ReferenceLine x={B} stroke="#475569" strokeWidth={1.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3. Matmul interactive (with tiling intensity)
// ---------------------------------------------------------------------------
function MatmulInteractiveSection({ hardwareIntensity }: { hardwareIntensity: number }) {
  const [B, setB] = useState(128);
  const [D, setD] = useState(4096);
  const [F, setF] = useState(4096);
  const [bm, setBm] = useState(128);

  const currentIntensity = (2 * B * D * F) / (2 * B * D + 2 * D * F + 2 * B * F);
  const tiledIntensity = (bm * bm) / (bm + bm);
  const bn = bm;

  const data = useMemo(() => {
    const pts: any[] = [];
    for (let batch = 1; batch <= 1024; batch *= 1.12) {
      const b = Math.round(batch);
      pts.push({ batch: b, intensity: (2 * b * D * F) / (2 * b * D + 2 * D * F + 2 * b * F) });
    }
    return pts;
  }, [D, F]);

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-5">
          <Slider label="Token batch (B)" value={B} min={1} max={1024} onChange={setB} />
          <Slider label="Dimension D" value={D} min={128} max={16384} step={128} onChange={setD} />
          <Slider label="Dimension F" value={F} min={128} max={16384} step={128} onChange={setF} />
          <Slider label="Tile size (bm = bn)" value={bm} min={16} max={512} step={16} onChange={setBm} />

          <div className="glass rounded-lg p-4">
            <div className="flex justify-between items-baseline">
              <div className="text-sm text-slate-500">Arithmetic intensity</div>
              <div className="text-2xl font-mono font-bold text-slate-900">{currentIntensity.toFixed(1)}</div>
            </div>
            <div className="flex flex-wrap items-center gap-2 mt-2">
              <BoundBadge compute={currentIntensity > hardwareIntensity} />
            </div>
            <div className="text-[11px] text-slate-400 mt-2">
              Tiled (on-chip reuse): I ≈ <span className="font-mono">{tiledIntensity.toFixed(0)}</span> — re-loading tiles from
              HBM lowers intensity below the naive <i>≈B</i> rule even for the same batch.
            </div>
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.35} />
              <XAxis dataKey="batch" label={{ value: 'Batch (B)', position: 'insideBottom', offset: -8, fontSize: 10, fill: C.slate }} />
              <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
              <Tooltip content={<ChartTip />} />
              <ReferenceLine y={hardwareIntensity} stroke={C.ridge} strokeDasharray="3 3"
                label={{ position: 'top', value: 'Ridge', fill: C.ridge, fontSize: 10, fontWeight: 700 }} />
              <Line type="monotone" dataKey="intensity" name="Intensity" stroke={C.accent} strokeWidth={3} dot={false} />
              <ReferenceLine x={B} stroke="#475569" strokeWidth={1.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4. Prefill vs Generation
// ---------------------------------------------------------------------------
function PrefillGenerationSection({ peakFlops, peakBw, hardwareIntensity }: any) {
  const [batch, setBatch] = useState(32);
  const [context, setContext] = useState(4096);

  const prefillTokens = batch * context;
  const prefillAttention = context / 2;
  const generateAttention = 1;

  const data = [
    { name: 'Prefill attention', intensity: prefillAttention, class: 'compute',
      desc: `≈ T/2 = ${prefillAttention.toFixed(0)} FLOPs/B — way above the ridge (${fmtNum(hardwareIntensity)}), compute-bound.` },
    { name: 'Prefill matmuls', intensity: prefillTokens, class: 'compute',
      desc: `weights reused over ~${fmtNum(prefillTokens)} tokens — compute-bound.` },
    { name: 'Generate attention', intensity: generateAttention, class: 'memory',
      desc: `≈ ST/(S+T) ≈ 1 — a small constant, always memory-bound.` },
    { name: `Generate × batch ${batch}`, intensity: (batch * context) / (batch * context + context), class: 'middle',
      desc: `batching joins KV reads so weights amortize slightly; still usually below the ridge.` },
  ];

  return (
    <div className="glass rounded-xl p-5">
      <h3 className="font-bold text-slate-800 mb-3">Where Each Phase Sits</h3>
      <div className="space-y-4">
        <Slider label="Concurrent requests (batch)" value={batch} min={1} max={1024} onChange={setBatch} />
        <Slider label="Context length" value={context} min={256} max={65536} step={256} onChange={setContext}
          format={(v: number) => v.toLocaleString()} />
      </div>
      <div className="h-64 mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} layout="vertical" margin={{ top: 10, right: 24, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
            <XAxis type="number" scale="log" domain={[0.5, 'dataMax']}
              label={{ value: 'Arithmetic intensity (log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
            <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
            <Tooltip content={<ChartTip />} labelFormatter={(l: any, p: any) => p[0]?.payload?.desc || l} />
            <ReferenceLine x={hardwareIntensity} stroke={C.ridge} strokeDasharray="4 4"
              label={{ position: 'top', value: 'ridge', fill: C.ridge, fontSize: 10 }} />
            <Bar dataKey="intensity" name="Intensity" radius={[0, 5, 5, 0]} barSize={26}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.class === 'compute' ? C.compute : d.class === 'middle' ? C.amber : C.memory} />
              ))}
            </Bar>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <ul className="text-xs text-slate-500 mt-3 space-y-1.5">
        {data.map((d) => (
          <li key={d.name} className="flex gap-2">
            <span className="w-2 h-2 rounded-full mt-1 shrink-0"
              style={{ background: d.class === 'compute' ? C.compute : d.class === 'middle' ? C.amber : C.memory }} />
            <span>{d.desc}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NEW: Attention intensity ST/(S+T) — prefill vs generation
// ---------------------------------------------------------------------------
function AttentionIntensitySection({ hardwareIntensity }: { hardwareIntensity: number }) {
  const [context, setContext] = useState(2048);
  const data = useMemo(() => {
    const pts: any[] = [];
    for (let s = 16; s <= 65536; s *= 1.12) {
      const prefill = s / 2;                       // S = T
      const genPrefill = (s * 1) / (s + 1);         // generation T = 1, S growing
      pts.push({ context: s, prefill, generation: genPrefill });
    }
    return pts;
  }, []);

  const crossover = hardwareIntensity * 2; // T/2 = I_hw  => T = 2*I_hw

  return (
    <div className="glass rounded-xl p-5">
      <h3 className="font-bold text-slate-800 mb-3">Attention Intensity: ST/(S+T)</h3>
      <p className="text-sm text-slate-500 mb-3">
        Prefill (intensity ≈ <i>T/2</i>, rising with context) crosses the ridge; generation is pinned at <strong>≈1</strong>.
      </p>
      <Slider label="Prompt context (tokens), for reference" value={context} min={16} max={65536}
        onChange={setContext} format={(v: number) => v.toLocaleString()} />
      <div className="h-60 mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="attnGen" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={C.amber} stopOpacity={0.35} />
                <stop offset="95%" stopColor={C.amber} stopOpacity={0.03} />
              </linearGradient>
              <linearGradient id="attnPrefill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={C.compute} stopOpacity={0.35} />
                <stop offset="95%" stopColor={C.compute} stopOpacity={0.03} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
            <XAxis dataKey="context" type="number" scale="log" domain={['dataMin', 'dataMax']}
              tickFormatter={(v: any) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
              label={{ value: 'Context / sequence length (log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
            <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
            <Tooltip content={<ChartTip />} />
            <ReferenceLine y={hardwareIntensity} stroke={C.ridge} strokeDasharray="4 4"
              label={{ position: 'top', value: `ridge ${fmtNum(hardwareIntensity)}`, fill: C.ridge, fontSize: 10 }} />
            <ReferenceLine x={crossover} stroke="#475569" strokeDasharray="4 4"
              label={{ position: 'top', value: 'crossover ≈ 2·ridge', fill: C.slate, fontSize: 10 }} />
            <Area type="monotone" dataKey="generation" name="Generation (T=1)" stroke={C.amber} strokeWidth={2.5}
              fill="url(#attnGen)" dot={false} />
            <Area type="monotone" dataKey="prefill" name="Prefill (S=T)" stroke={C.compute} strokeWidth={2.5}
              fill="url(#attnPrefill)" dot={false} />
            <ReferenceLine x={context} stroke="#475569" strokeWidth={1.5} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-400 mt-3">
        Prefill attention becomes compute-bound once <i>T/2 &gt; I<sub>hw</sub></i>, i.e. <i>T &gt; ~{fmtNum(crossover)}</i> tokens.
        Generation never does — no amount of batching or head tuning changes that constant ≈1.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NEW: KV cache growth
// ---------------------------------------------------------------------------
function KVCacheSection({ hardwareIntensity, hw }: any) {
  const [context, setContext] = useState(8192);
  const [batch, setBatch] = useState(32);
  const [heads, setHeads] = useState(8);       // KV heads K
  const [headDim, setHeadDim] = useState(128); // H
  const [layers, setLayers] = useState(64);    // L
  const [bpv, setBpv] = useState(2);           // bytes per float

  const kvPerToken = 2 * bpv * headDim * heads * layers;
  const kvPerSeq = kvPerToken * context;
  const kvTotal = kvPerSeq * batch;

  // fit against HBM capacity for the selected hardware
  const data = useMemo(() => {
    const pts: any[] = [];
    for (let t = 256; t <= 131072; t *= 1.2) {
      pts.push({ context: t, perSeq: (kvPerToken * t) / 1e9, total: (kvPerToken * t * batch) / 1e9 });
    }
    return pts;
  }, [kvPerToken, batch]);

  const capacity = (hw.capacity || 80) * 1e9;
  const maxBatchForCtx = Math.floor(capacity / kvPerSeq);
  const pctOfCapacity = (kvTotal / capacity) * 100;

  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <div className="glass rounded-xl p-5">
        <h3 className="font-bold text-slate-800 mb-1">KV memory = 2·bytes·H·K·L·T·B</h3>
        <p className="text-sm text-slate-500 mb-4">Drag context &amp; batch to watch KV memory explode.</p>
        <div className="space-y-4">
          <Slider label="Context length (T)" value={context} min={256} max={131072}
            onChange={setContext} format={(v: number) => v.toLocaleString()} />
          <Slider label="Concurrent requests (B)" value={batch} min={1} max={512} onChange={setBatch} />
          <Slider label="KV heads (K) — GQA" value={heads} min={1} max={128} onChange={setHeads} />
          <Slider label="Head dim (H)" value={headDim} min={32} max={256} step={16} onChange={setHeadDim} />
          <Slider label="Layers (L)" value={layers} min={8} max={128} step={8} onChange={setLayers} />
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Bytes per float</label>
              <span className="font-mono text-slate-900">{bpv} B</span>
            </div>
            <div className="grid grid-cols-3 gap-1.5">
              {[{ b: 2, l: 'bf16/fp16' }, { b: 1, l: 'int8/fp8' }, { b: 0.5, l: 'fp4' }].map((o) => (
                <button key={o.b} onClick={() => setBpv(o.b)}
                  className={cn('glass rounded-md py-1.5 text-[11px] font-semibold transition-colors',
                    bpv === o.b ? 'bg-accent text-white border-accent' : 'text-slate-600 hover:border-accent/40')}>
                  {o.l}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">Per token</span>
            <span className="font-mono font-semibold text-slate-800">{fmtBytes(kvPerToken)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">Per sequence ({fmtNum(context, 0)} ctx)</span>
            <span className="font-mono font-semibold text-slate-800">{fmtBytes(kvPerSeq)}</span>
          </div>
          <div className="flex justify-between text-sm border-t border-slate-200 pt-2">
            <span className="text-slate-500">Total (×{batch})</span>
            <span className="font-mono font-bold text-slate-900 text-lg">{fmtBytes(kvTotal)}</span>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>% of {hw.id} HBM ({fmtBytes(capacity)})</span>
              <span className="font-mono">{pctOfCapacity.toFixed(0)}%</span>
            </div>
            <div className="h-2.5 rounded-full bg-slate-200 overflow-hidden">
              <div className={cn('h-full rounded-full transition-all', pctOfCapacity > 100 ? 'bg-rose-500' : 'bg-violet-500')}
                style={{ width: `${Math.min(pctOfCapacity, 100)}%` }} />
            </div>
            <p className={cn('text-xs mt-2', maxBatchForCtx > 0 ? 'text-slate-500' : 'text-rose-600 font-semibold')}>
              {maxBatchForCtx > 0
                ? `Fits ≈ ${fmtNum(maxBatchForCtx)} requests of this context in one ${hw.id}.`
                : 'KV cache alone already exceeds this accelerator\'s HBM — you need GQA, quantization, or more chips.'}
            </p>
          </div>
        </div>
      </div>

      <div className="glass rounded-xl p-5">
        <h3 className="font-bold text-slate-800 mb-3">Growth vs Context &amp; Batch</h3>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="kvTotal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.violet} stopOpacity={0.4} />
                  <stop offset="95%" stopColor={C.violet} stopOpacity={0.03} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis dataKey="context" type="number" scale="log" domain={['dataMin', 'dataMax']}
                tickFormatter={(v: any) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                label={{ value: 'Context (log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
              <YAxis tickFormatter={(v: any) => `${v} GB`}
                label={{ value: 'Memory (GB)', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
              <Tooltip content={<ChartTip unit=" GB" />} />
              <ReferenceLine y={capacity / 1e9} stroke={C.ridge} strokeDasharray="4 4"
                label={{ position: 'top', value: `${hw.id} HBM`, fill: C.ridge, fontSize: 10 }} />
              <Line type="monotone" dataKey="perSeq" name="Per sequence" stroke={C.amber} strokeWidth={2.5} dot={false} />
              <Area type="monotone" dataKey="total" name="Total (×batch)" stroke={C.violet} strokeWidth={2.5}
                fill="url(#kvTotal)" dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-slate-400 mt-3">
          Linear in both direction — but the slope is set by <em>H·K·L</em>. Cutting KV heads (GQA) or dtype slices the whole
          curve, while batching raises it uniformly. This is why long-context × batch is what OOMs you, not the weights.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NEW: Latency vs throughput
// ---------------------------------------------------------------------------
function LatencyThroughputSection({ peakFlops, peakBw, hardwareIntensity, hw }: any) {
  const [paramsB, setParamsB] = useState(30);
  const [kvKb, setKvKb] = useState(100);      // KV bytes per token (kB)
  const [chips, setChips] = useState(16);
  const [context, setContext] = useState(8192);

  const bw = peakBw * chips;
  const flops = peakFlops * chips;
  const bytesPerParam = hw.bytesPerParam || 2;
  const paramsBytes = paramsB * 1e9 * bytesPerParam;
  const kvPerSeq = kvKb * 1024 * context;

  const data = useMemo(() => {
    const pts: any[] = [];
    for (let b = 1; b <= 512; b += 4) {
      const stepMs = ((b * kvPerSeq + paramsBytes) / bw) * 1000;
      const tps = b / (stepMs / 1000);
      pts.push({ B: b, stepMs, tps });
    }
    return pts;
  }, [kvPerSeq, paramsBytes, bw]);

  const critical = Math.max(1, Math.round(paramsBytes / kvPerSeq));

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Slider label="Model params (Billions)" value={paramsB} min={1} max={500} onChange={setParamsB} />
          <Slider label="KV size (kB / token)" value={kvKb} min={10} max={400} step={10} onChange={setKvKb} />
          <Slider label="Accelerators (chips)" value={chips} min={1} max={64} onChange={setChips} />
          <Slider label="Context length" value={context} min={1024} max={32768}
            onChange={setContext} format={(v: number) => v.toLocaleString()} />

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Total bandwidth</div>
              <div className="font-mono font-bold text-slate-900">{fmtBytes(bw) + '/s'}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Total FLOPs/s</div>
              <div className="font-mono font-bold text-slate-900">{fmtFlops(flops)}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Params loaded / step</div>
              <div className="font-mono font-bold text-slate-900">{fmtBytes(paramsBytes)}</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">KV / seq</div>
              <div className="font-mono font-bold text-slate-900">{fmtBytes(kvPerSeq)}</div>
            </div>
          </div>
          <p className="text-xs text-slate-400">
            Model: {paramsB}B {bytesPerParam === 2 ? 'bf16' : bytesPerParam === 1 ? 'int8' : 'fp4'} on {chips}×{hw.id}.
            (The example defaults mirror the scaling-book&rsquo;s 30B-on-16-chip case.)
          </p>
        </div>

        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="tpFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.compute} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={C.compute} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis dataKey="B" label={{ value: 'Batch size', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
              <YAxis yAxisId="tps" tickFormatter={(v: any) => `${fmtNum(v)}`}
                label={{ value: 'tokens/s', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.compute }} />
              <YAxis yAxisId="ms" orientation="right" tickFormatter={(v: any) => `${v}`}
                label={{ value: 'step ms', angle: 90, position: 'insideRight', fontSize: 10, fill: C.amber }} />
              <Tooltip content={<ChartTip />} />
              <ReferenceLine x={critical} stroke={C.ridge} strokeDasharray="4 4"
                label={{ position: 'top', value: `KV = weights ≈ ${fmtNum(critical)}`, fill: C.ridge, fontSize: 10 }} />
              <Area yAxisId="tps" type="monotone" dataKey="tps" name="Throughput (tok/s)" stroke={C.compute}
                strokeWidth={3} fill="url(#tpFill)" dot={false} />
              <Line yAxisId="ms" type="monotone" dataKey="stepMs" name="Step time (ms)" stroke={C.amber}
                strokeWidth={2.5} strokeDasharray="5 3" dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      <p className="text-xs text-slate-400 mt-4">
        Throughput rises fast at first, then flattens as memory bandwidth saturates near the critical batch — but latency keeps growing.
        To get more of <em>both</em>, add chips (move the whole frontier right), or shrink the KV cache.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 7. Network roofline
// ---------------------------------------------------------------------------
type LinkTier = 'domain' | 'scaleout' | 'host';

function NetworkRooflineInteractiveSection({ hw }: any) {
  const [D, setD] = useState(4096);
  const [tier, setTier] = useState<LinkTier>('domain');

  // Real per-chip link bandwidths where the vendor publishes them; the book's
  // TPU v5e figures stand in for chips that do not.
  const tiers: { id: LinkTier; label: string; gbps: number; sub: string }[] = [
    { id: 'domain', label: hw.vendor === 'Google' ? 'ICI (in-pod)' : 'NVLink (in-node)',
      gbps: (hw.linkBwGBs ?? 45) * 8, sub: hw.topology ?? 'tightly-coupled domain' },
    { id: 'scaleout', label: 'DCN / InfiniBand', gbps: (hw.scaleOutBwGBs ?? 6.25) * 8, sub: 'across pods' },
    { id: 'host', label: 'PCIe to host', gbps: (hw.hostBwGBs ?? 16) * 8, sub: 'offload / staging' },
  ];
  const active = tiers.find((t) => t.id === tier) ?? tiers[0];
  const networkBwGbps = active.gbps;

  const netBwBytes = (networkBwGbps * 1e9) / 8;
  const chipFlops = hw.tflops * 1e12;
  const threshold = chipFlops / netBwBytes; // D/2 > threshold  => D > 2*threshold
  const currentIntensity = D / 2;
  const criticalD = 2 * threshold;

  // curve of the 2-chip roofline throughput vs D
  const data = useMemo(() => {
    const pts: any[] = [];
    for (let d = 512; d <= 65536; d *= 1.12) {
      const flopTime = (d) / chipFlops;      // proxy: math per byte communicated
      // throughput approximation: FLOPs achieved per unit time along a fixed B,F
      const B = 256, F = 4096;
      const tf = 2 * B * d * F;
      const mathT = tf / chipFlops;
      const commT = (2 * B * F) / netBwBytes;
      pts.push({ D: d, achievable: Math.min(tf / mathT, tf / commT) / 1e12 });
    }
    return pts;
  }, [chipFlops, netBwBytes]);

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <Slider label="Model dimension (D)" value={D} min={512} max={32768} step={512} onChange={setD} />

          <div>
            <div className="text-xs font-medium text-slate-500 mb-2">Which link are the partial sums crossing?</div>
            <div className="grid grid-cols-3 gap-1.5">
              {tiers.map((t) => (
                <button key={t.id} onClick={() => setTier(t.id)}
                  className={cn('glass rounded-md py-1.5 px-1 text-[11px] font-semibold transition-colors',
                    tier === t.id ? 'bg-accent text-white border-accent' : 'text-slate-600 hover:border-accent/40')}>
                  {t.label}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              {active.sub} · <span className="font-mono">{fmtNum(networkBwGbps)} Gbps</span> per chip
              {hw.linkBwGBs ? '' : ' (book default — this chip does not publish link bandwidth)'}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Inter-chip intensity</div>
              <div className="font-mono font-bold text-slate-900 text-lg">{currentIntensity.toFixed(0)}</div>
              <div className="text-[10px] text-slate-400">= D/2</div>
            </div>
            <div className="glass rounded-lg p-3">
              <div className="text-slate-400 text-xs">Compute-bound when</div>
              <div className="font-mono font-bold text-slate-900 text-lg">D &gt; {fmtNum(criticalD)}</div>
              <div className="text-[10px] text-slate-400">independent of B</div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <BoundBadge compute={currentIntensity > threshold} />
            <span className="text-xs text-slate-400">Threshold intensity {threshold.toFixed(0)}</span>
          </div>

          <p className="text-xs text-slate-500 leading-relaxed">
            Step down a tier and the critical <i>D</i> jumps by the same factor the bandwidth fell. The in-domain link is
            usually fast enough to shard a real model across; the scale-out fabric almost never is, which is why tensor
            parallelism stops at the edge of an NVLink node or ICI slice and data parallelism takes over beyond it.
          </p>

          <div className="glass rounded-lg p-4 flex items-center justify-between">
            <div className="text-center flex-1 space-y-1">
              <div className="p-3 bg-violet-100 text-violet-800 rounded-lg font-bold text-sm">Chip 1 · half of D</div>
              <div className="font-mono text-[11px] text-violet-400 animate-pulse">↻ {fmtBytes(netBwBytes)}/s partial sums</div>
            </div>
            <div className="px-2 text-slate-300 font-black">⇄</div>
            <div className="text-center flex-1 space-y-1">
              <div className="p-3 bg-violet-100 text-violet-800 rounded-lg font-bold text-sm">Chip 2 · half of D</div>
              <div className="font-mono text-[11px] text-violet-400 animate-pulse">↻ {fmtBytes(netBwBytes)}/s partial sums</div>
            </div>
          </div>
        </div>

        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis dataKey="D" type="number" scale="log" domain={['dataMin', 'dataMax']}
                tickFormatter={(v: any) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                label={{ value: 'Model dimension D (log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
              <YAxis tickFormatter={(v: any) => `${v} T`}
                label={{ value: 'Achievable TFLOP/s', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
              <Tooltip content={<ChartTip unit=" TFLOP/s" />} />
              <ReferenceLine x={criticalD} stroke={C.ridge} strokeDasharray="4 4"
                label={{ position: 'top', value: `D>${fmtNum(criticalD)}`, fill: C.ridge, fontSize: 10 }} />
              <Line type="monotone" dataKey="achievable" name="2-chip throughput" stroke={C.violet} strokeWidth={3} dot={false} />
              <ReferenceLine x={D} stroke="#475569" strokeWidth={1.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NEW: Attention vs matmul FLOPs crossover (T/8D)
// ---------------------------------------------------------------------------
function AttentionFlopsSection() {
  const [dim, setDim] = useState(4096);
  const models = [
    { label: 'LLaMA-ish (D=4096)', d: 4096 },
    { label: 'Gemma-27B (D=4608)', d: 4608 },
    { label: 'Large (D=8192)', d: 8192 },
    { label: 'Huge (D=12288)', d: 12288 },
  ];

  const data = useMemo(() => {
    const pts: any[] = [];
    for (let t = 128; t <= 131072; t *= 1.1) {
      pts.push({ context: t, fraction: t / (8 * dim) });
    }
    return pts;
  }, [dim]);

  const crossover = 8 * dim;

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <h3 className="font-bold text-slate-800 mb-3">Attention share ≈ T/8D</h3>
          <p className="text-sm text-slate-500 mb-4">
            Attention&rsquo;s fraction of layer FLOPs is tiny at short context and only overtakes the MLP beyond <i>8&middot;D</i>.
          </p>
          <div className="space-y-2">
            {models.map((m) => (
              <button key={m.label} onClick={() => setDim(m.d)}
                className={cn('w-full glass rounded-lg p-2.5 text-left text-sm transition-colors',
                  dim === m.d ? 'bg-accent text-white border-accent' : 'text-slate-600 hover:border-accent/40')}>
                <span className="font-semibold">{m.label}</span>
                <span className={cn('block font-mono text-xs', dim === m.d ? 'text-white/70' : 'text-slate-400')}>
                  crossover ≈ {fmtNum(8 * m.d, 0)} tokens ({fmtNum((8 * m.d) / 1000, 0)}k)
                </span>
              </button>
            ))}
          </div>
          <div className="glass rounded-lg p-3 mt-4">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">At your context</span>
            </div>
            <div className="text-xl font-mono font-bold text-slate-900 mt-1">
              {crossover <= 131072 ? (crossover / 1000).toFixed(0) : '>131'}k
            </div>
            <div className="text-xs text-slate-400">tokens before attention dominates</div>
          </div>
        </div>

        <div className="lg:col-span-2 h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="attnFrac" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.sky} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={C.sky} stopOpacity={0.03} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis dataKey="context" type="number" scale="log" domain={['dataMin', 'dataMax']}
                tickFormatter={(v: any) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                label={{ value: 'Context length (log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
              <YAxis tickFormatter={(v: any) => `${Math.round(v * 100)}%`} domain={[0, 1]}
                label={{ value: '% of layer FLOPs', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
              <Tooltip content={<ChartTip />} />
              <ReferenceLine y={0.5} stroke="#475569" strokeDasharray="4 4"
                label={{ position: 'top', value: 'attention = 50%', fill: C.slate, fontSize: 10 }} />
              <ReferenceLine x={crossover} stroke={C.ridge} strokeDasharray="4 4"
                label={{ position: 'top', value: 'T = 8D', fill: C.ridge, fontSize: 10 }} />
              <Area type="monotone" dataKey="fraction" name="Attention share" stroke={C.sky} strokeWidth={3}
                fill="url(#attnFrac)" dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-400 mt-2">
            Below <i>8&middot;D</i> tokens you are matmul-limited in FLOPs; attention only wins at enormous contexts. But remember:
            during generation attention still dominates <em>memory</em> via KV, regardless of these FLOPs.
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 9. Quantization
// ---------------------------------------------------------------------------
function QuantizationInteractiveSection({ hardwareIntensity, hw }: any) {
  const B = 256;
  const D = 4096;
  const F = 4096;

  const int1 = (2 * B * D * F) / (2 * B * D + 2 * D * F + 2 * B * F);
  const int2 = (2 * B * D * F) / (1 * B * D + 1 * D * F + 1 * B * F);
  const int3 = (2 * B * D * F) / (2 * B * D + 1 * D * F + 2 * B * F);
  const int4 = (2 * B * D * F) / (2 * B * D + 2 * B * D * F + 2 * B * F);

  const rows = [
    { name: 'BF16', intensity: int1, bCrit: hardwareIntensity, note: 'weights & compute bf16', peak: '1×' },
    { name: 'Mixed int8 w', intensity: int3, bCrit: hardwareIntensity / 2, note: 'int8 weights, bf16 compute → B_crit halves', peak: '1×' },
    { name: 'Pure int8', intensity: int2, bCrit: hardwareIntensity, note: 'int8 both; 2× compute → B_crit back to base', peak: '2×' },
    { name: 'Batch-specific', intensity: int4, bCrit: 2, note: 'unique weights per token → always bound', peak: '1×' },
  ];

  const critData = [
    { name: 'bf16', tpu: hardwareIntensity, h100: hardwareIntensity },
    { name: 'int8 w', tpu: hardwareIntensity / 2, h100: hardwareIntensity / 2 },
    { name: 'pure int8', tpu: hardwareIntensity, h100: hardwareIntensity },
  ];

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid lg:grid-cols-2 gap-6">
        <div>
          <h3 className="font-bold text-slate-800 mb-3">Intensity by scheme (B = {B})</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
                <Tooltip content={<ChartTip />} />
                <ReferenceLine y={hardwareIntensity} stroke={C.ridge} strokeDasharray="4 4"
                  label={{ position: 'top', value: 'ridge', fill: C.ridge, fontSize: 10 }} />
                <Bar dataKey="intensity" name="Intensity" radius={[6, 6, 0, 0]} barSize={40}>
                  {rows.map((d, i) => (
                    <Cell key={i} fill={d.intensity > hardwareIntensity ? C.compute : C.memory} />
                  ))}
                </Bar>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            Mixed int8-weights pushes below the ridge at a high batch; batch-specific weights collapse to a constant ≈2.
          </p>
        </div>

        <div>
          <h3 className="font-bold text-slate-800 mb-3">Critical batch by scheme &amp; chip</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={critData} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis label={{ value: 'B_crit', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
                <Tooltip content={<ChartTip />} />
                <Bar dataKey="tpu" name="TPU class" fill={C.sky} radius={[6, 6, 0, 0]} />
                <Bar dataKey="h100" name="H100 class" fill={C.violet} radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto mt-4">
        <table className="w-full text-sm text-slate-600">
          <thead>
            <tr className="text-left text-slate-400 text-xs uppercase tracking-wider border-b border-slate-200">
              <th className="py-2 pr-3">Scheme</th>
              <th className="py-2 pr-3">Peak math</th>
              <th className="py-2 pr-3">Intensity (B=256)</th>
              <th className="py-2 pr-3">Critical batch B_crit</th>
              <th className="py-2">Effect</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className="border-b border-slate-100">
                <td className="py-2 pr-3 font-medium">{r.name}</td>
                <td className="py-2 pr-3 font-mono">{r.peak}</td>
                <td className="py-2 pr-3 font-mono">{r.intensity.toFixed(1)}</td>
                <td className="py-2 pr-3 font-mono">≈{r.bCrit.toFixed(0)}</td>
                <td className="py-2 text-slate-500">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NEW: Memory hierarchy & tiling intensity
// ---------------------------------------------------------------------------
function MemoryHierarchySection({ hw, hardwareIntensity, onChipRatio, onChipRidge }: {
  hw: any; hardwareIntensity: number; onChipRatio: number; onChipRidge: number;
}) {
  const [tile, setTile] = useState(128);

  const tileData = useMemo(() => {
    const pts: any[] = [];
    const N = 512;
    for (let bm = 8; bm <= N; bm *= 1.1) {
      pts.push({ bm, intensity: (bm * bm) / (bm + bm) });
    }
    return pts;
  }, []);

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid lg:grid-cols-2 gap-6">
        <div>
          <h3 className="font-bold text-slate-800 mb-3">Ridge point depends on the memory tier</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[
                { name: 'Fed from HBM', intensity: Math.round(hardwareIntensity) },
                { name: 'Fed from on-chip', intensity: Math.max(1, Math.round(onChipRidge)) },
              ]} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis label={{ value: 'Intensity to hit peak', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
                <Tooltip content={<ChartTip />} />
                <Bar dataKey="intensity" name="intensity" radius={[6, 6, 0, 0]} barSize={60}>
                  <Cell fill={C.memory} />
                  <Cell fill={C.compute} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            On <code>{hw.id}</code> the on-chip scratchpad ({hw.onChipMemMB ? `${hw.onChipMemMB} MB` : 'VMEM / SMEM'})
            runs at ~<strong>{onChipRatio.toFixed(0)}&times;</strong> HBM bandwidth
            {hw.onChipBwTBs ? ` (${hw.onChipBwTBs} TB/s vs ${hw.memBw} TB/s)` : ' (book estimate — this chip does not publish the figure)'},
            so an op fed from it needs only ~<strong>{fmtNum(onChipRidge, 1)}</strong> intensity to saturate the matrix
            unit, versus ~{fmtNum(hardwareIntensity)} from HBM.
          </p>
        </div>

        <div>
          <h3 className="font-bold text-slate-800 mb-3">Tiling sets your effective intensity</h3>
          <Slider label="Tile size (bm = bn)" value={tile} min={8} max={512} step={8} onChange={setTile} />
          <div className="h-48 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={tileData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis dataKey="bm" type="number" scale="log" domain={['dataMin', 'dataMax']}
                  tickFormatter={(v: any) => v}
                  label={{ value: 'Tile dim (log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
                <YAxis label={{ value: 'I ≈ bm·bn/(bm+bn)', angle: -90, position: 'insideLeft', fontSize: 9, fill: C.slate }} />
                <Tooltip content={<ChartTip />} />
                <ReferenceLine y={hardwareIntensity} stroke={C.ridge} strokeDasharray="4 4"
                  label={{ position: 'top', value: 'HBM ridge', fill: C.ridge, fontSize: 10 }} />
                <Line type="monotone" dataKey="intensity" name="Tiled intensity" stroke={C.sky} strokeWidth={3} dot={false} />
                <ReferenceLine x={tile} stroke="#475569" strokeWidth={1.5} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            I ≈ <span className="font-mono">{((tile * tile) / (tile + tile)).toFixed(0)}</span> at bm = bn = {tile}.
            Equal squares maximize the harmonic mean — which is why square tiles are standard.
          </p>
        </div>
      </div>

      <div className="glass rounded-xl p-5 mt-6">
        <h3 className="font-bold text-slate-800 mb-3">Where bm·bn/(bm+bn) comes from</h3>
        <div className="text-sm text-slate-600 space-y-3">
          <p>
            The clean <i>I &asymp; B</i> result assumes each operand is loaded from HBM exactly once. Real matmuls are
            too big for that: an <i>(m,k) &middot; (k,n)</i> product is cut into tiles of <i>bm &times; bk</i> and{' '}
            <i>bk &times; bn</i> that fit in on-chip memory, and each tile gets re-loaded for every tile of the other
            operand it has to meet. With <i>tm = m/bm</i>, <i>tn = n/bn</i>, <i>tk = k/bk</i>:
          </p>
          <div className="glass rounded-lg p-4 font-mono text-xs text-slate-700 space-y-1 overflow-x-auto">
            <div>FLOPs = 2 · tm · tn · tk · bm · bn · bk</div>
            <div>Bytes = 2 · tm · tn · ( tk · (bm·bk + bk·bn) + bm·bn )</div>
            <div className="pt-1 text-slate-500">drop the output term (bm·bn), cancel tm·tn·tk and bk:</div>
            <div className="text-slate-900 font-bold">I = bm·bn / (bm + bn)</div>
          </div>
          <p>
            Note what disappeared: <strong>k vanishes entirely</strong>. Contracting over a longer dimension adds math
            and traffic in equal measure, so a skinny-but-deep matmul is no more efficient than a shallow one. Only the
            tile&rsquo;s <em>output</em> shape sets the intensity — which is the real reason a fatter on-chip memory
            makes a chip faster, and why <code>bm = bn</code> is the optimum for a fixed tile area.
          </p>
          <p className="text-slate-500">
            At <span className="font-mono">bm = bn = {tile}</span> the effective intensity is{' '}
            <span className="font-mono font-bold">{((tile * tile) / (tile + tile)).toFixed(0)}</span>, which on this chip is{' '}
            <strong className={(tile / 2) >= hardwareIntensity ? 'text-emerald-600' : 'text-rose-600'}>
              {(tile / 2) >= hardwareIntensity ? 'above' : 'below'}
            </strong>{' '}
            the HBM ridge of {fmtNum(hardwareIntensity)}. That is the whole reason tile size is a performance knob and
            not an implementation detail.
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NEW: MoE critical batch
// ---------------------------------------------------------------------------
function MoeSection() {
  const [expertPercent, setExpertPercent] = useState(64); // E/k sparsity (log-ish)
  const presets = [
    { label: 'Dense (E/k = 1)', ratio: 1 },
    { label: 'Small MoE (E/k = 8)', ratio: 8 },
    { label: 'Typical (E/k = 32)', ratio: 32 },
    { label: 'DeepSeek v3 (E/k = 32)', ratio: 32, deepseek: true },
  ];
  const ratio = Math.round(2 ** (expertPercent / 10));
  const crit = 120 * ratio;

  const data = useMemo(() => {
    const pts: any[] = [];
    for (let b = 1; b <= 8192; b *= 1.1) {
      const bv = Math.round(b);
      pts.push({ B: bv, dense: Math.min(1, bv / 240), moe: Math.min(1, bv / crit) });
    }
    return pts;
  }, [crit]);

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="font-bold text-slate-800">Compute-bound requires B &gt; 120·(E/k)</h3>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Sparsity factor (E/k)</label>
              <span className="font-mono text-slate-900">{ratio}</span>
            </div>
            <input type="range" min={0} max={70} value={expertPercent} onChange={(e) => setExpertPercent(Number(e.target.value))}
              className="glass-slider w-full" />
            <div className="flex flex-wrap gap-1.5 mt-2">
              {presets.map((p) => (
                <button key={p.label} onClick={() => setExpertPercent(Math.round(Math.log2(p.ratio) * 10))}
                  className="glass rounded-md px-2 py-1 text-[11px] font-semibold text-slate-600 hover:border-accent/40 transition-colors">
                  {p.label}
                </button>
              ))}
            </div>
          </div>
          <div className="glass rounded-lg p-4">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Critical batch (int8 w, bf16 math)</span>
              <span className="font-mono font-bold text-lg text-slate-900">{fmtNum(crit, 0)}</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Stored params scale ×E but only <i>k</i> are read per token — so you need <i>E/k</i> &times; more concurrent tokens
              to saturate parameter bandwidth. DeepSeek&nbsp;v3 (E=256, k=8) needs B &gt; 3,840.
            </p>
          </div>
        </div>

        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="moeFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.compute} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={C.compute} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis dataKey="B" type="number" scale="log" domain={['dataMin', 'dataMax']}
                tickFormatter={(v: any) => fmtNum(v)}
                label={{ value: 'Serving batch (log)', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
              <YAxis tickFormatter={(v: any) => `${Math.round(v * 100)}%`} domain={[0, 1]}
                label={{ value: 'Utilization', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
              <Tooltip content={<ChartTip />} />
              <ReferenceLine x={240} stroke={C.sky} strokeDasharray="4 4"
                label={{ position: 'top', value: 'dense B_crit', fill: C.sky, fontSize: 10 }} />
              <ReferenceLine x={crit} stroke={C.ridge} strokeDasharray="4 4"
                label={{ position: 'top', value: `MoE B_crit ${fmtNum(crit, 0)}`, fill: C.ridge, fontSize: 10 }} />
              <Area type="monotone" dataKey="dense" name="Dense utilization" stroke={C.sky} strokeWidth={2.5}
                fill="url(#moeFill)" dot={false} />
              <Line type="monotone" dataKey="moe" name="MoE utilization" stroke={C.compute} strokeWidth={3} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 12. Worked problems
// ---------------------------------------------------------------------------
function WorkedProblemsSection({ peakFlops, peakBw, hardwareIntensity }: any) {
  const hbmBw = 8.2e11;
  const bf16Flops = 1.97e14;
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const toggle = (key: string) => setRevealed((r) => ({ ...r, [key]: !r[key] }));

  const Answer = ({ id, children }: any) => revealed[id] ? (
    <div className="text-xs bg-emerald-50 border border-emerald-200 text-emerald-900 rounded-lg p-3 mt-2 space-y-1">{children}</div>
  ) : (
    <button onClick={() => toggle(id)} className="text-xs text-accent hover:underline mt-2">Show answer</button>
  );
  const Q = ({ children }: any) => (
    <div className="glass rounded-lg p-4">
      <div className="text-sm text-slate-700 leading-relaxed">{children}</div>
    </div>
  );

  const q3Data = useMemo(() => {
    const rows: any[] = [];
    const roofline = (B: number, DD: number, FF: number) => {
      const tf = 2 * B * DD * FF;
      const ft = tf / bf16Flops;
      const ct = (2 * B * DD + DD * FF + 2 * B * FF) / hbmBw;
      return tf / Math.max(ft, ct);
    };
    for (let b = 1; b <= 512; b += 8) {
      rows.push({ B: b, big: roofline(b, 4096, 4096) / 1e12, small: roofline(b, 1024, 1024) / 1e12 });
    }
    return rows;
  }, []);

  return (
    <div className="space-y-4">
      <Q>
        <strong>Q1 — int8 matmul:</strong> <i>X[B,D] ·<sub>D</sub> Y[D,F] → Z[B,F]</i> in int8 (1 byte/param) vs bf16 (2 bytes/param).
        <ol className="list-decimal pl-5 mt-2 text-xs text-slate-500">
          <li>Bytes loaded / written?</li>
          <li>Total OPs?</li>
          <li>Arithmetic intensity?</li>
          <li>T_math, T_comms, and reasonable lower/upper runtime bounds?</li>
        </ol>
        <Answer id="q1">
          <p><strong>1.</strong> Load <i>BD + DF</i> bytes (1 byte/param), write <i>BF</i>.</p>
          <p><strong>2.</strong> Still <i>2BDF</i> OPs (int8 just runs faster).</p>
          <p><strong>3.</strong> <i>I = 2BDF/(BD+DF+BF) ≈ 2B</i>. int8 ridge = <i>3.94e14/8.2e11 = 480</i>, so rule <i>B &gt; 240</i> — basically unchanged.</p>
          <p><strong>4.</strong> <i>T_math = 2BDF/3.94e14</i>, <i>T_comms = (BD+DF+BF)/8.2e11</i>. Lower = max, upper = sum.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q2 — int8 weights + bf16 math:</strong> weights int8, activations/compute bf16, <i>1.97e14</i> bf16 FLOPs/s.
        At what batch size do we become compute-bound?
        <Answer id="q2">
          <p><i>2BDF</i> bf16 FLOPs but only <i>DF</i> weight bytes. Compute-bound when <i>2B &gt; 240 → B<sub>crit</sub> &gt; 120</i> — half of bf16. Easy quantization win.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q3 — roofline vs B:</strong> peak FLOPs/s vs B for <i>D=F=4096</i> and <i>D=F=1024</i> (exact bytes).
        Larger D/F reaches peak sooner; D=F=1024 roughly doubles the critical batch.
        <div className="h-56 mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={q3Data} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
              <XAxis dataKey="B" label={{ value: 'Batch', position: 'insideBottom', offset: -6, fontSize: 10, fill: C.slate }} />
              <YAxis tickFormatter={(v: any) => `${Number(v).toFixed(0)}`}
                label={{ value: 'TFLOP/s', angle: -90, position: 'insideLeft', fontSize: 10, fill: C.slate }} />
              <Tooltip content={<ChartTip unit=" TFLOP/s" />} />
              <Line type="monotone" dataKey="big" name="D=F=4096" stroke={C.accent} strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="small" name="D=F=1024" stroke={C.amber} strokeWidth={3} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <Answer id="q3">
          <p>Both curves saturate at the hardware peak (~{(bf16Flops / 1e12).toFixed(0)} TFLOP/s on the textbook TPU), but the bigger model crosses the ridge at a smaller batch. Small matmuls need ~2&times; the batch to become compute-bound.</p>
          <p className="text-[10px] text-slate-400">This mirrors the Interactive Lab: small batches are memory-bound, and the crossover is set by the hardware ridge (≈{fmtNum(hardwareIntensity)} here).</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q4 — batch-specific weights:</strong> <i>int8[B,D] ·<sub>D</sub> int8[B,D,F] → int8[B,F]</i>, a different matrix per batch element. Arithmetic intensity?
        <Answer id="q4">
          <p>FLOPs = <i>2BDF</i>. Comms = <i>BD + BDF + BF</i>. Since <i>BDF</i> dominates, <i>I ≈ 2</i>, constant — <strong>always communication-bound</strong> regardless of B.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q5 — H100 memory roofline:</strong> using the H100 spec sheet, find the batch at which a bf16 matmul becomes compute-bound (tensor-core bf16 ≈2&times; true value due to sparsity).
        <Answer id="q5">
          <p>Reported bf16 = <i>1.979e15</i> "with sparsity"; true = <i>9.89e14</i>. With <i>3.35e12</i> bytes/s: <i>B<sub>crit</sub> ≈ 295</i> tokens — similar to TPUs.</p>
        </Answer>
      </Q>

      <div className="pt-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        Network rooflines — from the TPU &amp; GPU chapters
      </div>

      <Q>
        <strong>Q6 — data parallelism:</strong> In the backward pass of pure DP / FSDP over an axis of size <i>X</i>, each
        layer does <i>8BDF</i> FLOPs spread over the axis and must move <i>8DF</i> bytes of gradients over the collective
        network. At what per-chip batch does DP stop being comms-bound?
        <Answer id="q6">
          <p><i>T<sub>math</sub> = 8BDF/(X·C)</i> and <i>T<sub>comms</sub> = 8DF/W</i>. Setting <i>T<sub>math</sub> &gt; T<sub>comms</sub></i>,
          the <i>DF</i> cancels and you get the clean rule <i><strong>B/X &gt; C/W</strong></i> — per-chip batch above the
          chip&rsquo;s FLOPs-to-collective-bandwidth ratio.</p>
          <p>On an H100 node that ratio works out to roughly <strong>2500 tokens per GPU</strong>. Note the shape of the
          answer: like the HBM roofline it depends on <em>batch</em>, unlike the tensor-parallel roofline below.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q7 — tensor parallelism:</strong> TP over an axis of size <i>Y</i> needs an AllGather and a ReduceScatter of
        the <em>activations</em>: <i>T<sub>math</sub> = 4BDF/(Y·C)</i>, <i>T<sub>comms</sub> = 4BD/W</i>. How far can you
        shard before going comms-bound, and why does that number match the size of a physical node?
        <Answer id="q7">
          <p>Both sides carry <i>4BD</i>, so it cancels: compute-bound requires <i><strong>Y &lt; F·W/C</strong></i>.
          <strong> B disappears entirely</strong> — you cannot batch your way out of a TP bottleneck, exactly as with the
          2-chip <i>D/2</i> roofline.</p>
          <p>Within an NVLink node <i>W/C</i> gives about <i>F/2200</i>. For LLaMA-3&rsquo;s <i>F = 28,672</i> that is
          ~11-way, i.e. about 8-way once you round to a real topology — which is precisely why an NVLink domain is 8 GPUs.
          Cross a node boundary and <i>W</i> collapses, so TP effectively stops at the domain edge.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q8 — which knob moves which roofline?</strong> You are comms-bound. For each of HBM, tensor parallelism and
        data parallelism, does raising the batch size help?
        <Answer id="q8">
          <p><strong>HBM:</strong> yes — <i>I ≈ B</i>, so more tokens per weight load is the entire fix.</p>
          <p><strong>Data parallelism:</strong> yes — the rule is <i>B/X &gt; C/W</i>, so a bigger per-chip batch buys headroom.</p>
          <p><strong>Tensor parallelism:</strong> <em>no</em> — the rule <i>Y &lt; F·W/C</i> has no <i>B</i> in it. Only a wider
          model, a faster link, or less sharding helps. This is the single most useful asymmetry to remember: batch fixes
          memory rooflines, model width fixes network rooflines.</p>
        </Answer>
      </Q>
    </div>
  );
}
