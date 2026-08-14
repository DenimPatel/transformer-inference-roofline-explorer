import { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { physicalUnits, computeRow } from '../../lib/roofline';

const REGIONS = [
  { id: 'US Hyperscale', priceKwh: 0.07, pue: 1.1 },
  { id: 'US Commercial', priceKwh: 0.12, pue: 1.4 },
  { id: 'Europe Avg', priceKwh: 0.22, pue: 1.35 },
  { id: 'Green / Hydro', priceKwh: 0.04, pue: 1.08 },
];

const UNITS = physicalUnits({ flopsTera: 989, memBwTera: 3.35, memCapGb: 80, totalParamsB: 7, activeParamsB: 7, bytesPerParam: 2, bytesPerTokenKb: 128 });

export default function LessonCost({ onComplete }: { onComplete: () => void }) {
  const [region, setRegion] = useState(REGIONS[0]);
  const [utilization, setUtilization] = useState(65);

  const X = useMemo(
    () => ({
      totalParams: 7e9, activeParams: 7e9, bytesPerParam: 2, contextLen: 4096, bytesPerToken: 128 * 1024,
      tdpWatts: 700, hardwarePrice: 30000, priceKwh: region.priceKwh, pue: region.pue,
      utilization, amortizationYears: 3,
    }),
    [region, utilization]
  );

  const curve = useMemo(() => {
    const out = [];
    for (let i = 2; i <= 12; i += 0.5) {
      const B = Math.round(Math.pow(2, i));
      out.push(computeRow(B, UNITS, X));
    }
    return out;
  }, [X]);

  const best = useMemo(() => curve.reduce((a, b) => (b.totalCost1M < a.totalCost1M ? b : a)), [curve]);

  return (
    <LessonShell
      number={6}
      title="Total Cost of Ownership"
      subtitle="Electricity plus amortized hardware — and why location, utilization, and batch all move the price."
      sourceRef="reference/scaling-book/applied-inference.md — Visualizing the Latency Throughput Tradeoff"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="tco" />
          <ConceptTag id="latency-throughput" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          TCO ≈ Electricity (Joules × PUE × $/kWh) + Hardware amortized over its lifetime at your utilization. It falls
          with batch size until diminishing returns. Pick a region and drag utilization to see the cost curve move.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <div className="text-xs font-medium text-slate-600 mb-1.5">Geography / Power</div>
            <div className="flex flex-wrap gap-2">
              {REGIONS.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  onClick={() => setRegion(r)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-full transition-colors ${region.id === r.id ? 'bg-[var(--color-accent)] text-white' : 'glass-chip text-slate-600 hover:bg-white/80'}`}
                >
                  {r.id}
                </button>
              ))}
            </div>
          </div>
          <SliderControl label="Hardware utilization" value={utilization} min={10} max={100} step={1} onChange={setUtilization} unit="%" conceptId="tco" />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Stat label="Electricity price" value={`$${region.priceKwh}`} unit="/kWh" />
          <Stat label="PUE" value={region.pue.toFixed(2)} unit="" />
          <Stat label="Min TCO" value={`$${best.totalCost1M.toFixed(2)}`} unit={'/1M tkns @ batch ' + best.batchSize.toLocaleString()} />
        </div>

        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={curve} margin={{ top: 8, right: 12, left: -6, bottom: 12 }}>
              <defs>
                <linearGradient id="costFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#5b7cfa" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#5b7cfa" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="batchSize" type="number" scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} label={{ position: 'bottom', value: 'Batch (log)', fontSize: 10 }} />
              <YAxis tickFormatter={(v) => `$${Number(v).toFixed(0)}`} label={{ value: 'USD / 1M', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any, n: any) => [`$${Number(v).toFixed(2)}`, n]} labelFormatter={(v) => `Batch ${Number(v).toLocaleString()}`} />
              <Area type="monotone" dataKey="totalCost1M" name="Total TCO" stroke="#5b7cfa" strokeWidth={3} fill="url(#costFill)" dot={false} />
              <ReferenceLine x={best.batchSize} stroke="#22c48b" strokeWidth={2} label={{ position: 'top', value: 'min', fill: '#15896a', fontSize: 10 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'PUE (Power Usage Effectiveness) captures…',
            options: ['Compute speed', 'Cooling overhead in electricity cost', 'Batch size', 'KV cache size'],
            answer: 1,
            explain: 'PUE multiplies the electricity cost to account for cooling/infrastructure overhead.',
          },
          {
            q: 'TCO per token typically…',
            options: [
              'Rises monotonically with batch',
              'Falls with batch until diminishing returns',
              'Is independent of batch',
              'Depends only on FLOPs',
            ],
            answer: 1,
            explain: 'Larger batches amortize fixed parameter reads, cutting cost, until KV-memory costs bring diminishing returns.',
          },
        ]}
      />
    </LessonShell>
  );
}

function Stat({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="glass rounded-xl px-4 py-3">
      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{label}</div>
      <div className="text-lg font-mono font-bold text-slate-800 mt-0.5">
        {value} <span className="text-xs font-normal text-slate-400">{unit}</span>
      </div>
    </div>
  );
}
