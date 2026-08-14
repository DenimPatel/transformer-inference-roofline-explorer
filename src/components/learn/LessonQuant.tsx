import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { cn } from '../../lib/utils';

const RIDGE = 295; // H100 bf16

const SCHEMES = [
  { id: 'bf16', name: 'bf16 w + bf16 math', bCrit: RIDGE, note: 'baseline' },
  { id: 'int8w', name: 'int8 weights, bf16 math', bCrit: RIDGE / 2, note: 'B_crit halves — easy win' },
  { id: 'int8', name: 'int8 w + int8 math', bCrit: RIDGE, note: 'FLOPs×2 cancels the byte win' },
  { id: 'moe', name: 'Batch-specific (MoE)', bCrit: Infinity, note: 'intensity ≈ 2 — always bound' },
];

export default function LessonQuant({ onComplete }: { onComplete: () => void }) {
  const [B, setB] = useState(120);

  const data = SCHEMES.map((s) => ({ name: s.name, bCrit: Math.min(s.bCrit, 4000), infinite: s.bCrit === Infinity, id: s.id, note: s.note }));
  const isComputeAt = (bCrit: number) => B >= bCrit;

  return (
    <LessonShell
      number={7}
      title="Quantization & the β Rule"
      subtitle="Lowering bytes-per-parameter changes intensity and moves the critical batch size."
      sourceRef="reference/scaling-book/inference.md — Linear operations: what bottlenecks us?"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="quantization" />
          <ConceptTag id="critical-batch" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          The critical batch obeys <code className="text-xs">B_crit = β · I_hw</code>, where{' '}
          <code className="text-xs">β = bits/param ÷ bits/activation</code>. int8 weights with bf16 math halve β and
          thus B_crit — you become compute-bound sooner. int8 math doubles FLOPs/s, cancelling the win.
        </p>

        <SliderControl label="Current token batch (B)" value={B} min={1} max={1024} step={1} onChange={setB} unit="tokens" conceptId="critical-batch" />

        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="name" tick={{ fontSize: 9.5 }} />
              <YAxis tickFormatter={(v) => (v === 4000 ? '∞' : `${v}`)} label={{ value: 'B_crit', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any, n: any, p: any) => [p.payload.infinite ? '∞ (always bound)' : `${Number(v).toFixed(0)}`, 'Critical batch']} labelFormatter={(l, payload) => payload?.[0]?.payload?.note || l} />
              <ReferenceLine y={B} stroke="#475569" strokeDasharray="3 3" label={{ position: 'top', value: `your B=${B}`, fill: '#475569', fontSize: 10 }} />
              <Bar dataKey="bCrit" radius={[6, 6, 0, 0]}>
                {data.map((d) => (
                  <Cell key={d.id} fill={d.infinite ? '#f25f7d' : B >= d.bCrit ? '#22c48b' : '#c7d2fe'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {data.map((d) => (
            <div key={d.id} className="glass rounded-xl px-3 py-2 flex items-center justify-between gap-2">
              <span className="text-[12px] text-slate-600">{d.name}</span>
              <span className={cn('text-[11px] font-bold', d.infinite ? 'text-rose-500' : isComputeAt(d.bCrit) ? 'text-emerald-600' : 'text-amber-600')}>
                {d.infinite ? 'always memory-bound' : B >= d.bCrit ? 'compute-bound' : 'memory-bound'}
              </span>
            </div>
          ))}
        </div>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'int8 weights with bf16 compute does what to B_crit?',
            options: ['Halves it', 'Doubles it', 'Leaves it unchanged', 'Makes the model dense'],
            answer: 0,
            explain: 'β (bits/param ÷ bits/activation) halves, so B_crit halves — you hit the ridge sooner.',
          },
          {
            q: 'int8 weights AND int8 math…',
            options: [
              'Halves B_crit again',
              'Roughly cancels out, returning B_crit to baseline',
              'Always bound',
              'Triples B_crit',
            ],
            answer: 1,
            explain: 'int8 math doubles FLOPs/s (raises I_hw) while int8 weights halve β — the two effects cancel.',
          },
        ]}
      />
    </LessonShell>
  );
}
