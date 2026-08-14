import { useMemo, useState } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import { cn } from '../../lib/utils';

const RIDGE = 295; // H100 bf16

export default function LessonIntensity({ onComplete }: { onComplete: () => void }) {
  const [B, setB] = useState(64);
  const [D, setD] = useState(4096);
  const F = D;

  const intensity = (2 * B * D * F) / (2 * B * D + 2 * D * F + 2 * B * F);
  const bound = intensity > RIDGE ? 'Compute-bound' : 'Memory-bound';

  const data = useMemo(() => {
    const pts = [];
    for (let batch = 1; batch <= 1024; batch *= 1.15) {
      const b = Math.round(batch);
      const i = (2 * b * D * F) / (2 * b * D + 2 * D * F + 2 * b * F);
      pts.push({ batch: b, intensity: i, approx: b });
    }
    return pts;
  }, [D, F]);

  return (
    <LessonShell
      number={2}
      title="Arithmetic Intensity & The B Rule"
      subtitle="How much math you get per byte — and why a matmul is compute-bound iff its token batch exceeds the ridge."
      sourceRef="reference/scaling-book/roofline.md — Matrix multiplication"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="arithmetic-intensity" />
          <ConceptTag id="matmul-intensity" />
          <ConceptTag id="critical-batch" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          For <code className="text-xs">X[B,D]·Y[D,F]→Z[B,F]</code> the intensity is{' '}
          <code className="text-xs">I = 2BDF/(2BD+2DF+2BF)</code>. When B is small relative to D and F this collapses to{' '}
          <strong className="text-slate-800">I ≈ B</strong> — the matmul's intensity *is* its token batch size. So on an
          H100, a bf16 matmul is compute-bound roughly when <strong className="text-slate-800">B &gt; ~295</strong>.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-5">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="font-medium text-slate-600">Token batch size (B)</span>
                <span className="font-mono text-slate-800">{B}</span>
              </div>
              <input type="range" min={1} max={1024} value={B} onChange={(e) => setB(Number(e.target.value))} className="glass-slider w-full" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="font-medium text-slate-600">Model width D (=F)</span>
                <span className="font-mono text-slate-800">{D}</span>
              </div>
              <input type="range" min={1024} max={16384} step={256} value={D} onChange={(e) => setD(Number(e.target.value))} className="glass-slider w-full" />
            </div>
            <GlassCard className="p-4 space-y-2" hover>
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">Arithmetic intensity</span>
                <span className="font-mono font-bold text-slate-800">{intensity.toFixed(1)} FLOPs/B</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">≈ B rule</span>
                <span className="font-mono text-slate-600">{B}</span>
              </div>
              <div className={cn('text-sm font-bold mt-1', bound.includes('Compute') ? 'text-emerald-600' : 'text-amber-600')}>
                {bound.includes('Compute') ? '✅ Compute-bound' : '⚠️ Memory-bound'}
              </div>
              <p className="text-[11px] text-slate-400">
                Critical batch ≈ {RIDGE} tokens. Cross it and you stop wasting FLOPs.
              </p>
            </GlassCard>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 10, right: 10, left: -14, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
                <XAxis dataKey="batch" type="number" scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => `${v}`} label={{ value: 'Token batch (log)', position: 'bottom', fontSize: 10 }} />
                <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip />
                <Line type="monotone" dataKey="approx" name="≈ B" stroke="#94a3b8" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                <Line type="monotone" dataKey="intensity" name="Exact" stroke="#5b7cfa" strokeWidth={3} dot={false} />
                <ReferenceLine y={RIDGE} stroke="#f25f7d" strokeDasharray="3 3" label={{ position: 'top', value: 'Ridge', fill: '#f25f7d', fontSize: 10 }} />
                <ReferenceLine x={B} stroke="#475569" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'A matmul with batch B=10 on an H100 (ridge ~295) is…',
            options: ['Compute-bound', 'Memory-bound', 'Bandwidth-irrelevant', 'Overlapping'],
            answer: 1,
            explain: 'B=10 < 295, so intensity is far below the ridge — you are memory-bound and waste FLOPs.',
          },
          {
            q: 'Why does intensity ≈ B for a big matmul?',
            options: [
              'Because B is large relative to D and F',
              'Because D,F ≈ 2BDF simplify the denominator',
              'Because when B << D,F the 2DF term dominates the denominator',
              'Because bandwidth cancels out',
            ],
            answer: 2,
            explain: 'When B << D and F, I = 2BDF/(2DF + small terms) ≈ B.',
          },
        ]}
      />
    </LessonShell>
  );
}
