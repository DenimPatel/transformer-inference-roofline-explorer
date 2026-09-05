import { useMemo, useState, type ReactNode } from 'react';
import {
  LineChart, Line, BarChart, Bar, ComposedChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine, Cell, Area,
} from 'recharts';
import {
  Cpu, Zap, TrendingUp, Scale, Database, Gauge, Lightbulb, Flame,
  ArrowRight, BookOpen, BarChart3,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { HARDWARE_PROFILES, effectiveMfu, type HardwareProfile } from '../lib/hardware';
import { computeRow, maxThroughput, type PhysicalUnits, type RowInputs } from '../lib/roofline';
import KpiCard from './ui/KpiCard';
import SliderControl from './ui/SliderControl';
import InfoPopover from './ui/InfoPopover';
import { ConceptTag } from './ui/ConceptTag';

/* ------------------------------------------------------------------ *
 *  Data helpers
 * ------------------------------------------------------------------ */

interface ProfileRun {
  profile: HardwareProfile;
  u: PhysicalUnits;
  x: RowInputs;
  bCrit: number;          // ridge point (tokens) — flops / memBw
  maxThroughput: number;  // tokens/s (compute ceiling)
  mfu: number;
  peakTflops: number;     // raw, precision-scaled
  achievedTflops: number; // peak * mfu
}

function buildRun(profile: HardwareProfile, units: PhysicalUnits, rowInputs: RowInputs): ProfileRun {
  const flops = profile.tflops * 1e12 * (2 / profile.bytesPerParam);
  const memBw = profile.memBw * 1e12;
  const u: PhysicalUnits = {
    flops,
    memBw,
    memCap: profile.capacity * 1e9,
    totalParams: units.totalParams,
    activeParams: units.activeParams,
    bytesPerParam: profile.bytesPerParam,
    bytesPerToken: units.bytesPerToken,
    hardwareRatio: flops / memBw,
  };
  const x: RowInputs = {
    ...rowInputs,
    tdpWatts: profile.tdp,
    hardwarePrice: profile.price,
    isLiquid: /rubin|groq|liquid/i.test(profile.id),
  };
  const mfu = effectiveMfu(profile);
  return {
    profile,
    u,
    x,
    bCrit: u.hardwareRatio,
    maxThroughput: maxThroughput(u),
    mfu,
    peakTflops: flops / 1e12,
    achievedTflops: (flops * mfu) / 1e12,
  };
}

/* ------------------------------------------------------------------ *
 *  Presentational atoms
 * ------------------------------------------------------------------ */

interface ChartCardProps {
  title: ReactNode;
  subtitle?: ReactNode;
  conceptId?: string;
  tags?: string[];
  rationale?: string;
  wide?: boolean;
  children: ReactNode;
}

function ChartCard({ title, subtitle, conceptId, tags, rationale, wide, children }: ChartCardProps) {
  return (
    <div className={cn('glass-card p-5 flex flex-col min-h-0', wide && 'lg:col-span-2 xl:col-span-3')}>
      <div className="mb-3 shrink-0">
        <div className="flex items-start justify-between gap-2">
          <h3 className="text-base font-bold text-slate-800 leading-tight">{title}</h3>
          <span className="flex items-center gap-2 shrink-0">
            {tags?.map((t) => <span key={t} className="inline-flex"><ConceptTag id={t} /></span>)}
            {conceptId && <InfoPopover conceptId={conceptId} className="text-slate-300" />}
          </span>
        </div>
        {subtitle && <p className="text-sm text-slate-500 mt-0.5">{subtitle}</p>}
        {rationale && (
          <p className="mt-2 text-[12px] leading-relaxed text-slate-600 bg-white/60 border border-slate-200/60 rounded-lg px-3 py-2">
            <Lightbulb className="inline -mt-0.5 mr-1.5 text-[var(--color-amber)]" style={{ width: 13, height: 13 }} />
            {rationale}
          </p>
        )}
      </div>
      <div className="flex-1 w-full min-h-[300px]">{children}</div>
    </div>
  );
}

const TooltipShell = ({ children }: { children: ReactNode }) => (
  <div className="glass-tooltip p-3 rounded-xl text-xs space-y-1 text-slate-700 min-w-[200px] z-50">{children}</div>
);

const fmtInt = (n: number) => n.toLocaleString('en-US', { maximumFractionDigits: 0 });
const fmtMs = (n: number) => `${n.toFixed(1)} ms`;
const fmtUsd = (n: number) => `$${n.toFixed(2)}`;

const AXIS = { stroke: '#cbd5e1', tick: { fill: '#64748b', fontSize: 11 } };
const BatchLogAxis = ({ l }: { l?: boolean }) =>
  <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
    tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} {...AXIS}
    label={l ? { value: 'Batch size (tokens, log)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 } : undefined} />;

/* ------------------------------------------------------------------ *
 *  Tooltips
 * ------------------------------------------------------------------ */

function MultiTooltip({ titleKey, formatter, active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const sorted = [...payload].sort((a, b) => b.value - a.value);
  return (
    <TooltipShell>
      <div className="font-bold text-slate-900 mb-2 border-b border-slate-200 pb-1">
        {titleKey ? titleKey(label) : `Batch: ${Number(label).toLocaleString()} tokens`}
      </div>
      {sorted.map((e: any) => (
        <div key={e.name} className="flex justify-between items-center gap-4">
          <span className="flex items-center gap-2 min-w-0">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: e.stroke || e.color }} />
            <span className="font-medium truncate">{e.name}</span>
          </span>
          <span className="font-mono whitespace-nowrap">{formatter(e.value, e.payload)}</span>
        </div>
      ))}
    </TooltipShell>
  );
}

/* ------------------------------------------------------------------ *
 *  Main component
 * ------------------------------------------------------------------ */

interface Props {
  units: PhysicalUnits;
  rowInputs: RowInputs;
  selectedProfiles: string[];
}

export default function CompetitiveAnalysis({ units, rowInputs, selectedProfiles }: Props) {
  const [batch, setBatch] = useState(256);

  const runs = useMemo(
    () =>
      HARDWARE_PROFILES.filter((p) => selectedProfiles.includes(p.id)).map((p) => buildRun(p, units, rowInputs)),
    [selectedProfiles, units, rowInputs]
  );

  // Per-profile row across the log-spaced batch sizes (for latency/throughput/cost/energy lines)
  const fullCurve = useMemo(() => {
    // reuse the same log spacing by sampling computeRow on a shared set of sizes
    const sizes: number[] = [];
    for (let i = 0; i <= 16; i += 0.4) sizes.push(Math.round(Math.pow(2, i)));
    const uniq = Array.from(new Set(sizes)).filter((s) => s > 0);
    return runs.map((r) => ({ id: r.profile.id, color: r.profile.color, rows: uniq.map((B) => computeRow(B, r.u, r.x)) }));
  }, [runs]);

  const combined = useMemo(() => {
    const curves = fullCurve;
    const sizes = curves[0]?.rows.map((r) => r.batchSize) ?? [];
    return sizes.map((B) => {
      const row: any = { batchSize: B };
      curves.forEach((c) => {
        const r = c.rows.find((rr) => rr.batchSize === B)!;
        row[`lat_${c.id}`] = r.latency;
        row[`thr_${c.id}`] = r.throughput;
        row[`cost_${c.id}`] = r.totalCost1M;
        row[`energy_${c.id}`] = r.joulesPerToken;
      });
      return row;
    });
  }, [fullCurve]);

  // Snapshot at the user-selected batch
  const snapshot = useMemo(() => {
    const rows = runs.map((r) => {
      const c = computeRow(batch, r.u, r.x);
      return {
        id: r.profile.id,
        color: r.profile.color,
        latency: c.latency,
        throughput: c.throughput,
        cost: c.totalCost1M,
        energy: c.joulesPerToken,
        tokensPerDollar: c.totalCost1M > 0 ? 1e6 / c.totalCost1M : Infinity,
        isMem: c.isMemoryBound,
        computeUtil: Math.min(1, c.tCompute / c.tMemory) * 100,
      };
    });
    return rows.sort((a, b) => a.cost - b.cost);
  }, [runs, batch]);

  // Roofline overlay data (one curve per profile)
  const rooflineData = useMemo(() => {
    const curves = runs.map((r) => {
      const pts: any[] = [];
      const minI = 0.05, maxI = 5e5;
      for (let i = Math.log10(minI); i <= Math.log10(maxI); i += 0.08) {
        const intensity = Math.pow(10, i);
        pts.push({ intensity, [`ach_${r.profile.id}`]: Math.min(r.u.memBw * intensity, r.u.flops) });
      }
      return { id: r.profile.id, color: r.profile.color, pts, flops: r.u.flops, bw: r.u.memBw, bCrit: r.bCrit };
    });
    return curves;
  }, [runs]);

  // Critical batch bar data
  const bCritData = useMemo(
    () => runs.map((r) => ({ name: r.profile.id, color: r.profile.color, bCrit: r.bCrit, mfu: r.mfu })),
    [runs]
  );

  // Capacity: max concurrent sequences (tokens worth of KV) that fit in HBM
  const capacityData = useMemo(() => {
    const weightBytes = units.totalParams * rowInputs.bytesPerParam;
    const perToken = units.bytesPerToken;
    return runs.map((r) => {
      const avail = Math.max(0, r.u.memCap - weightBytes);
      const seqs = perToken > 0 ? avail / perToken : 0;
      return { name: r.profile.id, color: r.profile.color, capacity: Math.max(0, seqs), paramsFit: units.totalParams * rowInputs.bytesPerParam / r.u.memCap };
    });
  }, [runs, units, rowInputs.bytesPerParam]);

  // Achieved FLOPs (MFU-adjusted)
  const flopsData = useMemo(
    () => runs.map((r) => ({ name: r.profile.id, color: r.profile.color, peak: r.peakTflops, achieved: r.achievedTflops, mfu: r.mfu })),
    [runs]
  );

  const maxBMem = useMemo(
    () => runs.map((r) => ({ name: r.profile.id, color: r.profile.color, memBw: r.u.memBw / 1e12, mem: r.u.memCap / 1e9 })),
    [runs]
  );

  if (runs.length === 0) {
    return (
      <div className="glass-card p-10 text-center text-slate-500 text-sm">
        Select at least one hardware profile from the left panel to begin comparing.
      </div>
    );
  }

  const bestCost = snapshot[0];

  return (
    <div className="space-y-5">
      {/* Intro banner */}
      <div className="glass rounded-2xl p-5">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl text-white flex items-center justify-center shrink-0"
            style={{ background: 'linear-gradient(135deg,#5b7cfa,#f25f7d)', boxShadow: '0 8px 22px -8px rgba(91,124,250,0.7)' }}>
            <BarChart3 className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-800">Head-to-head accelerator comparison</h2>
            <p className="text-sm text-slate-500 mt-0.5 max-w-3xl">
              Every accelerator is bounded by the same three resources — <em>how fast it does math</em>, <em>how fast it moves bytes</em>, and <em>how much it can store</em>.
              Which one "wins" depends entirely on whether you are <strong>compute-bound</strong> or <strong>memory-bound</strong>. Explore the charts below — each one teaches a different slice of that decision.
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              <ConceptTag id="roofline" />
              <ConceptTag id="ridge-point" />
              <ConceptTag id="compute-bound" />
              <ConceptTag id="memory-bound" />
              <ConceptTag id="critical-batch" />
              <ConceptTag id="tco" />
              <button
                type="button"
                onClick={() => setBatch(240)}
                className="glass-chip px-2.5 py-1 text-[10.5px] font-medium text-slate-600 hover:bg-white/90"
              >
                Jump to B_crit ≈ ridge
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Top KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard icon={<Gauge className="w-3.5 h-3.5" />} label="Cheapest / 1M tokens"
          value={bestCost.cost.toFixed(2)} unit="$" subValue={bestCost.id} conceptId="tco" />
        <KpiCard icon={<Cpu className="w-3.5 h-3.5" />} label="Fastest per-token"
          value={fmtMs(snapshot.reduce((a, b) => (b.latency < a.latency ? b : a)).latency)}
          subValue={snapshot.reduce((a, b) => (b.latency < a.latency ? b : a)).id} conceptId="latency-throughput" />
        <KpiCard icon={<Zap className="w-3.5 h-3.5" />} label="Top tokens / sec"
          value={fmtInt(snapshot.reduce((a, b) => (b.throughput > a.throughput ? b : a)).throughput)} unit="tok/s"
          subValue={snapshot.reduce((a, b) => (b.throughput > a.throughput ? b : a)).id} conceptId="latency-throughput" />
        <KpiCard icon={<Flame className="w-3.5 h-3.5" />} label="Best energy / token"
          value={snapshot.reduce((a, b) => (b.energy < a.energy ? b : a)).energy.toFixed(3)} unit="J"
          subValue={snapshot.reduce((a, b) => (b.energy < a.energy ? b : a)).id} conceptId="tco" />
      </div>

      {/* Interactive batch snapshot */}
      <div className="glass-card p-5">
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-[var(--color-accent)]" />
              Compare at batch size
            </span>
            <InfoPopover conceptId="critical-batch" iconSize={13} />
          </div>
          <button type="button" onClick={() => setBatch(512)} className="glass-chip px-2.5 py-1 text-xs text-slate-600 hover:bg-white/90">512</button>
          <button type="button" onClick={() => setBatch(1024)} className="glass-chip px-2.5 py-1 text-xs text-slate-600 hover:bg-white/90">1k</button>
          <button type="button" onClick={() => setBatch(4096)} className="glass-chip px-2.5 py-1 text-xs text-slate-600 hover:bg-white/90">4k</button>
          <div className="w-full lg:w-72 ml-auto">
            <SliderControl label="Current Batch (tokens)" value={batch} min={1} max={32768} step={1} onChange={setBatch} unit="tok" logScale conceptId="critical-batch" />
          </div>
        </div>
        <p className="text-[13px] text-slate-500 mb-3">
          At this batch, <strong>{bestCost.id}</strong> is cheapest ({fmtUsd(bestCost.cost)}/1M tokens). The winner changes as you slide the batch — that is the latency-throughput tradeoff working.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <ChartCard title="Cost at this batch" subtitle={`USD per 1M tokens, batch = ${batch.toLocaleString()}`} conceptId="tco" tags={['tco']}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={snapshot} layout="vertical" margin={{ top: 4, right: 30, left: 8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" {...AXIS} tickFormatter={(v) => `$${v.toFixed(2)}`} />
                <YAxis type="category" dataKey="id" width={190} {...AXIS} tick={{ fill: '#334155', fontSize: 11 }} />
                <Tooltip content={<MultiTooltip formatter={(v: any) => fmtUsd(v)} titleKey={(l) => `Hardware: ${l}`} />} />
                <Bar dataKey="cost" radius={[0, 6, 6, 0]} isAnimationActive>
                  {snapshot.map((s) => <Cell key={s.id} fill={s.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
          <ChartCard title="Throughput at this batch" subtitle="Tokens per second (decode) across the whole accelerator" conceptId="latency-throughput" tags={['latency-throughput']}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={snapshot} layout="vertical" margin={{ top: 4, right: 30, left: 8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" {...AXIS} tickFormatter={(v) => fmtInt(v)} />
                <YAxis type="category" dataKey="id" width={190} {...AXIS} tick={{ fill: '#334155', fontSize: 11 }} />
                <Tooltip content={<MultiTooltip formatter={(v: any) => `${fmtInt(v)} tok/s`} titleKey={(l) => `Hardware: ${l}`} />} />
                <Bar dataKey="throughput" radius={[0, 6, 6, 0]} isAnimationActive>
                  {snapshot.map((s) => <Cell key={s.id} fill={s.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>

      {/* Roofline overlay (wide) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
        <ChartCard wide title="Roofline comparison" subtitle="Achievable FLOPs/s vs arithmetic intensity (log-log) — the deeper you go left, the more bandwidth-bound"
          conceptId="roofline" tags={['roofline', 'ridge-point', 'arithmetic-intensity']}
          rationale="Each line is the theoretical ceiling: on the left it rises at memory-bandwidth (the sloped 'bandwidth roof'), then flattens at peak FLOPs at the ridge. An operation is compute-bound only to the right of its own ridge point. A decoder with a small batch sits far to the left — memory-bound — on every accelerator.">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rooflineData[0]?.pts ?? []} margin={{ top: 20, right: 24, left: 0, bottom: 20 }}>
              <defs>
                {rooflineData.map((c) => (
                  <linearGradient key={c.id} id={`roof_${c.id}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={c.color} stopOpacity={0.25} />
                    <stop offset="95%" stopColor={c.color} stopOpacity={0} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" />
              <XAxis dataKey="intensity" scale="log" domain={['dataMin', 'dataMax']} type="number" {...AXIS}
                tickFormatter={(v) => (Number(v) < 1 ? Number(v).toFixed(1) : Number(v).toFixed(0))}
                label={{ value: 'Arithmetic intensity (FLOPs / byte) — log', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }} />
              <YAxis scale="log" domain={['dataMin', 'dataMax']} type="number" {...AXIS}
                tickFormatter={(v) => (v / 1e12).toFixed(0)}
                label={{ value: 'Throughput (TFLOP/s) — log', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip labelFormatter={(v: any) => `Intensity: ${Number(v).toFixed(2)}`} formatter={(v: any, n: any) => [`${(Number(v) / 1e12).toFixed(1)} TFLOP/s`]} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              {rooflineData.map((c) => (
                <Area key={c.id} type="monotone" dataKey={`ach_${c.id}`} name={c.id} stroke={c.color} strokeWidth={2.5}
                  fill={`url(#roof_${c.id})`} dot={false} isAnimationActive={false} />
              ))}
              {rooflineData.map((c) => (
                <ReferenceLine key={`ref_${c.id}`} x={c.bCrit} stroke={c.color} strokeDasharray="3 3" opacity={0.5}
                  label={{ value: `${c.id.split(' ')[0]} ridge`, position: 'top', fill: c.color, fontSize: 9 }} />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Critical batch size (B_crit)" subtitle="Tokens per step needed to cross the ridge and become compute-bound"
          conceptId="critical-batch" tags={['critical-batch', 'ridge-point']}
          rationale="B_crit = Peak FLOPs/s ÷ memory bandwidth (× β for precision). Below this your per-token batch is memory-bound and you waste FLOPs. It is a single, human-scale number that ranks how hard each chip is to saturate — Groq's enormous bandwidth makes it nearly impossible to compute-bound.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={bCritData} layout="vertical" margin={{ top: 4, right: 30, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" horizontal={false} />
              <XAxis type="number" {...AXIS} tickFormatter={(v) => fmtInt(v)} />
              <YAxis type="category" dataKey="name" width={190} {...AXIS} tick={{ fill: '#334155', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any) => `${fmtInt(v)} tokens`} titleKey={(l) => `Hardware: ${l}`} />} />
              <Bar dataKey="bCrit" radius={[0, 6, 6, 0]} isAnimationActive>
                {bCritData.map((d, i) => <Cell key={d.name} fill={i === 0 ? '#f43f5e' : d.color} />)}
              </Bar>
              <ReferenceLine x={batch} stroke="#1e293b" strokeWidth={1.5} strokeDasharray="4 4"
                label={{ value: `your batch ${fmtInt(batch)}`, position: 'top', fill: '#1e293b', fontSize: 10 }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Achieved FLOPs after MFU" subtitle="Real-world peak vs the headline number (TPUs reach ~95%, GPUs ~80-85%)"
          conceptId="mfu" tags={['mfu', 'flops']}
          rationale="Advertised TFLOP/s are never fully realized. This chart shows the gap between peak (dashed) and achievable (solid bar) compute for each chip. When comparing 'which is fastest', comparing achieved FLOPs is more honest than comparing datasheets.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={flopsData} margin={{ top: 4, right: 20, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="name" {...AXIS} angle={-25} textAnchor="end" height={70} tick={{ fill: '#334155', fontSize: 10 }} />
              <YAxis {...AXIS} tickFormatter={(v) => `${Number(v / 1000).toFixed(1)}P`} label={{ value: 'TFLOP/s', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any, p: any) => `${fmtInt(v)} TFLOP/s (${Math.round((p.mfu ?? 1) * 100)}% MFU)`} titleKey={(l) => `Hardware: ${l}`} />} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar dataKey="achieved" name="Achieved" radius={[4, 4, 0, 0]} isAnimationActive>
                {flopsData.map((d) => <Cell key={d.name} fill={d.color} />)}
              </Bar>
              <Bar dataKey="peak" name="Peak (spec)" fillOpacity={0.25} radius={[4, 4, 0, 0]} isAnimationActive>
                {flopsData.map((d) => <Cell key={d.name} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Latency vs batch size" subtitle="Decode latency per step vs per-token batch (log-log)"
          conceptId="latency-throughput" tags={['latency-throughput', 'memory-bound']}
          rationale="At small batches latency is set by the fixed cost of streaming all weights from HBM (memory-bound). As you batch up, latency climbs gently. Smaller latency at a given batch = more bandwidth per parameter stream.">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combined} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} {...AXIS}
                label={{ value: 'Batch (log)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
              <YAxis {...AXIS} scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => v.toFixed(1)}
                label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any) => fmtMs(v)} titleKey={(l) => `Batch: ${Number(l).toLocaleString()} tokens`} />} />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              {runs.map((r) => <Line key={`lat_${r.profile.id}`} type="monotone" dataKey={`lat_${r.profile.id}`} name={r.profile.id} stroke={r.profile.color} strokeWidth={2} dot={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Throughput vs batch size" subtitle="Tokens / second across the accelerator (log-log)"
          conceptId="latency-throughput" tags={['latency-throughput']}
          rationale="Higher is better. Curves climb steeply then flatten as each chip saturates. Notice the winner at small batch (high bandwidth, e.g. Groq) vs large batch (raw FLOPs) can be different — that flip is exactly what the roofline predicts.">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combined} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} {...AXIS}
                label={{ value: 'Batch (log)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
              <YAxis {...AXIS} scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => fmtInt(v)}
                label={{ value: 'Tokens / s', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any) => `${fmtInt(v)} tok/s`} titleKey={(l) => `Batch: ${Number(l).toLocaleString()} tokens`} />} />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              {runs.map((r) => <Line key={`thr_${r.profile.id}`} type="monotone" dataKey={`thr_${r.profile.id}`} name={r.profile.id} stroke={r.profile.color} strokeWidth={2} dot={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Latency–throughput frontier" subtitle="Pareto tradeoff: each point is a batch size on one accelerator"
          conceptId="latency-throughput" tags={['latency-throughput', 'tco']}
          rationale="Small batches are fast but under-utilize hardware; large batches are slow but efficient. Each line traces one accelerator's tradeoff, and 'best' means where your application wants to sit. Doubling per-token latency can buy a ~100× drop in per-token cost.">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
              <XAxis type="number" dataKey="thr" {...AXIS} scale="log" domain={['auto', 'auto']} tickFormatter={(v) => fmtInt(v)}
                label={{ value: 'Throughput (tokens/s) — log', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
              <YAxis type="number" dataKey="lat" {...AXIS} scale="log" domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(1)}
                label={{ value: 'Latency (ms) — log', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any, p: any) => (p ? fmtMs(p.lat) + ' · ' + fmtInt(p.thr) + ' tok/s' : '')} titleKey={() => 'Point on frontier (batch increases right→)'} />} />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              {fullCurve.map((c) => (
                <Scatter key={`sc_${c.id}`} data={c.rows.map((r) => ({ thr: r.throughput, lat: r.latency }))} name={c.id}
                  line={{ stroke: c.color, strokeWidth: 1.5, strokeDasharray: '3 3' }} fill={c.color} fillOpacity={0.6} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Second grouping: economics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
        <ChartCard title="Total cost vs batch" subtitle="USD per 1M tokens across batch sizes (log)"
          conceptId="tco" tags={['tco']}
          rationale="Cost = electricity (Joules × PUE × $/kWh) + amortized hardware. Cost per token falls as you batch up (efficient utilization) but bottoms out — the right operating point is the curve's low knee, not the most expensive fast chip.">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combined} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} {...AXIS}
                label={{ value: 'Batch (log)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
              <YAxis {...AXIS} scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => `$${Number(v).toFixed(2)}`}
                label={{ value: 'Cost (USD/1M)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any) => fmtUsd(v)} titleKey={(l) => `Batch: ${Number(l).toLocaleString()} tokens`} />} />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              {runs.map((r) => <Line key={`cost_${r.profile.id}`} type="monotone" dataKey={`cost_${r.profile.id}`} name={r.profile.id} stroke={r.profile.color} strokeWidth={2} dot={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Energy per token vs batch" subtitle="Joules per token (absolute power efficiency)"
          conceptId="tco" tags={['tco']}
          rationale="Equal to power draw × step time ÷ tokens. Batched workloads are more energy-efficient per token because fixed weight-loading is amortized. This is the metric that decides your electricity bill and carbon footprint.">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combined} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} {...AXIS}
                label={{ value: 'Batch (log)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
              <YAxis {...AXIS} scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => Number(v).toFixed(3)}
                label={{ value: 'Joules / token', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any) => `${Number(v).toFixed(3)} J`} titleKey={(l) => `Batch: ${Number(l).toLocaleString()} tokens`} />} />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              {runs.map((r) => <Line key={`energy_${r.profile.id}`} type="monotone" dataKey={`energy_${r.profile.id}`} name={r.profile.id} stroke={r.profile.color} strokeWidth={2} dot={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Concurrent sequences your HBM can hold" subtitle="KV-cache capacity (how many sequences fit alongside the weights)"
          conceptId="kv-cache" tags={['kv-cache', 'memory-hierarchy']}
          rationale="Generated tokens each need a fresh KV cache. Fewer concurrent sequences = smaller batch = worse FLOPs utilization. This is often the real ceiling, more than math speed — huge-HBM chips (Vera, H200, Rubin) fit far more concurrent users.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={capacityData} layout="vertical" margin={{ top: 4, right: 30, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" horizontal={false} />
              <XAxis type="number" {...AXIS} tickFormatter={(v) => fmtInt(v)} label={{ value: 'concurrent sequences', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={190} {...AXIS} tick={{ fill: '#334155', fontSize: 11 }} />
              <Tooltip content={<MultiTooltip formatter={(v: any) => `${fmtInt(v)} sequences`} titleKey={(l) => `Hardware: ${l}`} />} />
              <Bar dataKey="capacity" radius={[0, 6, 6, 0]} isAnimationActive>
                {capacityData.map((d) => <Cell key={d.name} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Spec table */}
      <div className="glass-card p-5 overflow-x-auto custom-scrollbar">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-[var(--color-accent)]" />
          <h3 className="text-base font-bold text-slate-800">Accelerator datasheet & derived ranking</h3>
          <InfoPopover conceptId="ridge-point" className="text-slate-300" />
        </div>
        <table className="w-full text-xs min-w-[900px]">
          <thead>
            <tr className="text-left text-slate-400 uppercase tracking-wider border-b border-slate-200/80">
              <th className="py-2 pr-3">Hardware</th>
              <th className="py-2 pr-3">Vendor</th>
              <th className="py-2 pr-3 text-right">Peak (TFLOP)</th>
              <th className="py-2 pr-3 text-right">Mem BW (TB/s)</th>
              <th className="py-2 pr-3 text-right">HBM (GB)</th>
              <th className="py-2 pr-3 text-right">Ridge / B_crit</th>
              <th className="py-2 pr-3 text-right">Max tok/s</th>
              <th className="py-2 pr-3 text-right">TDP (W)</th>
              <th className="py-2 pr-3 text-right">Est. cost</th>
              <th className="py-2 pr-3 text-right">MFU</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.profile.id} className="border-b border-slate-100/80 last:border-0">
                <td className="py-2 pr-3">
                  <span className="flex items-center gap-2 font-semibold text-slate-700">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: r.profile.color }} />
                    {r.profile.id}
                  </span>
                </td>
                <td className="py-2 pr-3 text-slate-500">{r.profile.vendor}</td>
                <td className="py-2 pr-3 text-right font-mono">{fmtInt(r.peakTflops)}</td>
                <td className="py-2 pr-3 text-right font-mono">{r.u.memBw / 1e12} TB/s</td>
                <td className="py-2 pr-3 text-right font-mono">{r.u.memCap / 1e9}</td>
                <td className="py-2 pr-3 text-right font-mono">{fmtInt(r.bCrit)}</td>
                <td className="py-2 pr-3 text-right font-mono">{fmtInt(r.maxThroughput)}</td>
                <td className="py-2 pr-3 text-right font-mono">{fmtInt(r.profile.tdp)}</td>
                <td className="py-2 pr-3 text-right font-mono">${fmtInt(r.profile.price)}</td>
                <td className="py-2 pr-3 text-right font-mono">{(r.mfu * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Decision framework */}
      <div className="glass rounded-2xl p-5 text-[13px] text-slate-600 leading-relaxed">
        <h3 className="font-bold mb-3 flex items-center gap-2 text-slate-800">
          <Scale className="w-4 h-4 text-[var(--color-accent)]" /> How to actually choose: filter by regime first, then by cost
          <span className="ml-auto flex gap-2"><ConceptTag id="roofline" /><ConceptTag id="tco" /><ConceptTag id="prefill" /><ConceptTag id="generation" /></span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
          <div className="flex items-start gap-2">
            <InfoPopover conceptId="prefill" iconSize={13} className="mt-0.5" />
            <span><strong className="text-slate-800">Prefill is compute-bound.</strong> Long prompts reuse weights across many tokens, so you want peak FLOPs achieved (MFU) and low TTFT. Rank by achieved TFLOP/s.</span>
          </div>
          <div className="flex items-start gap-2">
            <InfoPopover conceptId="generation" iconSize={13} className="mt-0.5" />
            <span><strong className="text-slate-800">Generation is memory-bound.</strong> Every token streams the whole parameter set, so you want bandwidth and HBM capacity (bigger KV batch) — Groq's huge bandwidth shines here.</span>
          </div>
          <div className="flex items-start gap-2">
            <InfoPopover conceptId="tco" iconSize={13} className="mt-0.5" />
            <span><strong className="text-slate-800">Cost = FLOPs per dollar.</strong> The reference material's rule: serve on whichever is cheapest in FLOPs/$ — unless HBM or interconnect bandwidth is the true bottleneck, which it usually is for generation.</span>
          </div>
          <div className="flex items-start gap-2">
            <InfoPopover conceptId="critical-batch" iconSize={13} className="mt-0.5" />
            <span><strong className="text-slate-800">Batching makes or breaks it.</strong> To get compute-bound and efficient, your per-replica token batch must exceed B_crit — often 240-300 tokens on GPUs/TPUs, far higher on bandwidth-rich LPUs.</span>
          </div>
        </div>
        <div className="mt-4 flex items-start gap-2 bg-white/60 border border-slate-200/60 rounded-lg px-4 py-3">
          <BookOpen className="w-4 h-4 text-[var(--color-amber)] mt-0.5 shrink-0" />
          <p>
            <strong className="text-slate-800">Sanity-check the tradeoff:</strong> at a small batch (4 tokens) a 30B model on 16 TPU v5e runs a step in ≈<span className="font-mono">2.5 ms</span>; at batch 256 it is ≈<span className="font-mono">21 ms</span> — but far more tokens per step. Latency and throughput really are opposite ends of the same curve.
            <span className="ml-1 inline-flex"><ArrowRight className="w-3 h-3" /></span>
          </p>
        </div>
      </div>
    </div>
  );
}
