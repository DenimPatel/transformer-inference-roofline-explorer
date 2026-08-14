import { useMemo, useState } from 'react';
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';

const RIDGE = 240; // TPU bf16

export default function LessonAttention({ onComplete }: { onComplete: () => void }) {
  const [seq, setSeq] = useState(4096);
  const [D, setD] = useState(8192);

  const prefillI = seq / 2;
  const genI = seq / (seq + 1);
  const crossover = 8 * D;

  const prefillCurve = useMemo(() => {
    const rows: any[] = [];
    for (let s = 64; s <= 65536; s *= 1.4) {
      const ss = Math.round(s);
      rows.push({ seq: ss, prefill: ss / 2, gen: ss / (ss + 1) });
    }
    return rows;
  }, []);

  const flopsCurve = useMemo(() => {
    const rows: any[] = [];
    for (let s = 256; s <= 262144; s *= 1.4) {
      const ss = Math.round(s);
      rows.push({ seq: ss, ratio: ss / (8 * D) });
    }
    return rows;
  }, [D]);

  return (
    <LessonShell
      number={10}
      title="Attention Deep Dive"
      subtitle="Attention intensity ST/(S+T) flips from compute-bound prefill to always-memory-bound generation."
      sourceRef="reference/scaling-book/inference.md — What about attention? · reference/scaling-book/transformers.md — Appendix A & Fractional cost of attention"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="attention-intensity" />
          <ConceptTag id="flash-attention" />
          <ConceptTag id="local-attention" />
          <ConceptTag id="attention-flops" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          Attention moves <code className="text-xs">4BSD + 4BTD</code> bytes for <code className="text-xs">4BSTD</code> FLOPs, giving
          intensity <code className="text-xs">I = ST/(S+T)</code>. In <strong>prefill</strong> (S = T) that is ≈ <code className="text-xs">T/2</code> —
          comfortably above the ridge for any real prompt. In <strong>generation</strong> (T = 1) it collapses to a constant{' '}
          <code className="text-xs">≈ 1</code> — always memory-bound.
        </p>

        <SliderControl label="Sequence length (S)" value={seq} min={128} max={65536} step={128} onChange={setSeq} unit="tkns" logScale conceptId="attention-intensity" />
        <SliderControl label="Model dim (D)" value={D} min={1024} max={16384} step={256} onChange={setD} unit="" conceptId="attention-flops" />

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-500">
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-slate-400">Prefill intensity</div>
            <div className="font-mono font-bold text-slate-800">{prefillI.toFixed(0)}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-slate-400">Generation intensity</div>
            <div className="font-mono font-bold text-slate-800">{genI.toFixed(2)}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-slate-400">Ridge</div>
            <div className="font-mono font-bold text-slate-800">{RIDGE}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-slate-400">Attn dominates past</div>
            <div className="font-mono font-bold text-slate-800">{(8 * D).toLocaleString()} tkns</div>
          </div>
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="font-bold text-slate-800 mb-3">Attention intensity vs sequence length</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={prefillCurve} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="seq" scale="log" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} label={{ value: 'Sequence length (log)', position: 'bottom', fontSize: 10 }} />
              <YAxis scale="log" type="number" domain={[0.5, 'dataMax']} tickFormatter={(v) => (v < 1 ? v.toFixed(1) : v.toFixed(0))} label={{ value: 'Intensity (FLOPs/B, log)', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any, n: any) => [`${Number(v).toFixed(2)}`, n]} labelFormatter={(v: any) => `S=${v}`} />
              <ReferenceLine y={RIDGE} stroke="#f43f5e" strokeDasharray="3 3" label={{ position: 'top', value: 'Ridge', fill: '#f43f5e', fontSize: 10 }} />
              <Line type="monotone" dataKey="prefill" name="Prefill (≈ S/2)" stroke="#10b981" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="gen" name="Generation (≈1)" stroke="#f59e0b" strokeWidth={3} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-2 mt-3 text-[11px] text-slate-500">
          <span className="glass-chip px-2.5 py-1">At S={seq.toLocaleString()}: prefill intensity {prefillI.toFixed(0)} {prefillI >= RIDGE ? '(compute-bound ✓)' : '(still below ridge)'}</span>
          <span className="glass-chip px-2.5 py-1">Generation intensity ≈ {genI.toFixed(2)} — always bandwidth-bound</span>
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="font-bold text-slate-800 mb-3">When does attention dominate FLOPs? (ratio = T / 8D)</h3>
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={flopsCurve} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
              <defs>
                <linearGradient id="attnFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#7f6bf0" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#7f6bf0" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="seq" scale="log" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} label={{ value: 'Sequence length (log)', position: 'bottom', fontSize: 10 }} />
              <YAxis tickFormatter={(v) => Number(v).toFixed(1)} label={{ value: 'Attention / matmul FLOPs', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [`${Number(v).toFixed(2)}×`, 'Attention vs matmul']} labelFormatter={(v: any) => `S=${Number(v).toLocaleString()}`} />
              <ReferenceLine y={1} stroke="#10b981" strokeDasharray="3 3" label={{ position: 'top', value: 'attention = matmul (T = 8D)', fill: '#10b981', fontSize: 10 }} />
              <ReferenceLine x={crossover} stroke="#f43f5e" strokeDasharray="3 3" label={{ position: 'top', value: `${crossover.toLocaleString()}`, fill: '#f43f5e', fontSize: 10 }} />
              <Area type="monotone" dataKey="ratio" name="Attn/Matmul" stroke="#7f6bf0" strokeWidth={3} fill="url(#attnFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[13px] text-slate-600 mt-3">
          For D ≈ {D.toLocaleString()} the crossover is near {crossover.toLocaleString()} tokens. Below that, MLP matmuls dominate FLOPs; attention is a
          memory problem (KV cache) even when it is not a FLOPs problem.
        </p>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'During generation, attention arithmetic intensity is…',
            options: [
              '≈ T/2 and compute-bound',
              '≈ 1 (constant) and memory-bandwidth-bound',
              '≈ 8D and compute-bound',
              'infinite',
            ],
            answer: 1,
            explain: 'With T = 1, intensity ST/(S+T) ≈ S/(S+1) ≈ 1 — constant and far below the ridge, so decode attention is always bandwidth-bound.',
          },
          {
            q: 'Flash attention matters most because it…',
            options: [
              'Makes attention compute-bound',
              'Fuses attention on-chip and never materializes the big [B,S,T] matrix, raising intensity',
              'Adds more KV heads',
              'Removes the need for a KV cache',
            ],
            answer: 1,
            explain: 'By keeping chunks in VMEM with running max/O/L statistics, it avoids an O(B·S·T) HBM round-trip and raises attention\'s arithmetic intensity.',
          },
        ]}
      />
    </LessonShell>
  );
}
