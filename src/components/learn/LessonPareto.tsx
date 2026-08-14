import { useMemo, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ComposedChart,
} from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { physicalUnits, computeRow, maxThroughput } from '../../lib/roofline';

const UNITS = physicalUnits({ flopsTera: 989, memBwTera: 3.35, memCapGb: 80, totalParamsB: 7, activeParamsB: 7, bytesPerParam: 2, bytesPerTokenKb: 128 });
const X = { totalParams: 7e9, activeParams: 7e9, bytesPerParam: 2, contextLen: 4096, bytesPerToken: 128 * 1024, tdpWatts: 700, hardwarePrice: 30000, priceKwh: 0.07, pue: 1.1, utilization: 65, amortizationYears: 3 };

export default function LessonPareto({ onComplete }: { onComplete: () => void }) {
  const [batch, setBatch] = useState(64);
  const sizes = useMemo(() => {
    const out = [];
    for (let i = 2; i <= 12; i += 0.5) out.push(Math.round(Math.pow(2, i)));
    return out;
  }, []);

  const curve = useMemo(() => sizes.map((B) => computeRow(B, UNITS, X)), [sizes]);
  const mtp = maxThroughput(UNITS);
  const row = computeRow(batch, UNITS, X);

  return (
    <LessonShell
      number={4}
      title="Latency vs Throughput & the Critical Batch"
      subtitle="Small batches are fast but waste the hardware; big batches are efficient but add latency and KV-memory pressure."
      sourceRef="reference/scaling-book/inference.md — Theoretical estimates for LLM latency and throughput"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="latency-throughput" />
          <ConceptTag id="critical-batch" />
          <ConceptTag id="generation" />
        </div>
        <p className="text-sm leading-relaxed text-slate-600">
          This is the tradeoff you tune with batch size. Drag the batch slider and watch latency climb while throughput
          (and cost-efficiency) climb even faster — until diminishing returns set in near the critical batch.
        </p>

        <SliderControl label="Concurrent requests (batch)" value={batch} min={4} max={4096} step={1} logScale onChange={setBatch} unit="seqs" conceptId="critical-batch" />
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Stat label="Per-step latency" value={`${row.latency.toFixed(1)}`} unit="ms" />
          <Stat label="Throughput" value={row.throughput.toLocaleString(undefined, { maximumFractionDigits: 0 })} unit="tok/s" />
          <Stat label="Critical batch (max thrpt)" value={mtp.toLocaleString(undefined, { maximumFractionDigits: 0 })} unit="tokens" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curve} margin={{ top: 8, right: 10, left: -12, bottom: 12 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
                <XAxis dataKey="batchSize" type="number" scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} label={{ position: 'bottom', value: 'Batch (log)', fontSize: 10 }} />
                <YAxis tickFormatter={(v) => `${v.toFixed(1)}`} label={{ value: 'ms', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip formatter={(v: any) => [`${Number(v).toFixed(1)} ms`, 'Latency']} labelFormatter={(v) => `Batch ${Number(v).toLocaleString()}`} />
                <Line type="monotone" dataKey="latency" stroke="#f25f7d" strokeWidth={3} dot={false} />
                <ReferenceLine x={batch} stroke="#475569" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={curve} margin={{ top: 8, right: 10, left: -12, bottom: 12 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
                <XAxis dataKey="batchSize" type="number" scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} label={{ position: 'bottom', value: 'Batch (log)', fontSize: 10 }} />
                <YAxis tickFormatter={(v) => `${v.toFixed(0)}`} label={{ value: 'tok/s', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip formatter={(v: any) => [`${Number(v).toFixed(0)} tok/s`, 'Throughput']} labelFormatter={(v) => `Batch ${Number(v).toLocaleString()}`} />
                <Line type="monotone" dataKey="throughput" stroke="#22c48b" strokeWidth={3} dot={false} />
                <ReferenceLine x={batch} stroke="#475569" strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'The throughput-vs-latency tradeoff: raising the batch size…',
            options: [
              'Always reduces latency',
              'Raises throughput but increases per-step latency and KV memory',
              'Does nothing',
              'Only affects cost, not latency',
            ],
            answer: 1,
            explain: 'Bigger batches raise tokens/s but each step reads more KV cache, raising latency and memory.',
          },
          {
            q: 'Throughput gains from batching flatten out near…',
            options: ['B = 1', 'The critical batch size', 'B = 16', 'There is no ceiling'],
            answer: 1,
            explain: 'Once you cross the critical batch you are compute-bound; further gains are marginal.',
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
      <div className="text-xl font-mono font-bold text-slate-800 mt-0.5">
        {value} <span className="text-xs font-normal text-slate-400">{unit}</span>
      </div>
    </div>
  );
}
