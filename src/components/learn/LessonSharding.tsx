import { useState } from 'react';
import { ComposedChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { cn } from '../../lib/utils';

const ALPHA = 2550; // v5p ICI arithmetic intensity (bf16)
const RIDGE = 295; // HBM critical batch reference

type Strategy = 'dp' | 'tensor' | 'pipeline' | 'moe';

const STRATS: Record<Strategy, { name: string; desc: string }> = {
  dp: { name: 'DP / FSDP', desc: 'Data parallelism shards the batch; compute-bound when per-device token batch > C/W_ici.' },
  tensor: { name: 'Tensor (Megatron)', desc: 'Shards weights/activations; comms-bound when the sharding degree Y > F/2550.' },
  pipeline: { name: 'Pipeline', desc: 'Shards layers; communication is tiny but watch the pipeline bubble.' },
  moe: { name: 'Expert (MoE)', desc: 'Sparsity E/k inflates the generation critical batch.' },
};

export default function LessonSharding({ onComplete }: { onComplete: () => void }) {
  const [strategy, setStrategy] = useState<Strategy>('dp');
  const [Bdev, setBdev] = useState(1200);
  const [Y, setY] = useState(8);
  const [F, setF] = useState(28672); // LLaMA-3-70B
  const [E, setE] = useState(256);
  const [k, setK] = useState(8);

  const dpBound = Bdev >= ALPHA;
  const tensorBound = Y <= F / ALPHA; // compute-bound if Y not greater than F/alpha
  const tensorLimit = Math.floor(F / ALPHA);
  const moeCrit = Math.round((RIDGE / 2) * (E / k));
  const moeBound = Bdev >= moeCrit;

  const regime = (() => {
    switch (strategy) {
      case 'dp':
        return { bound: dpBound, text: dpBound ? 'Compute-bound (batch is big enough)' : 'Communication-bound (batch too small to hide comms)', optimistic: dpBound };
      case 'tensor':
        return { bound: tensorBound, text: tensorBound ? `Compute-bound (Y=${Y} ≤ F/α ≈ ${tensorLimit})` : `Communication-bound (Y=${Y} > F/α ≈ ${tensorLimit})`, optimistic: tensorBound };
      case 'pipeline':
        return { bound: true, text: 'Low communication — the cost is the pipeline bubble, not bandwidth. Microbatch to shrink it.', optimistic: true };
      case 'moe':
        return { bound: moeBound, text: moeBound ? `Compute-bound (B=${Bdev} ≥ ${moeCrit})` : `Memory-bound (B=${Bdev} < B_crit ≈ ${moeCrit})`, optimistic: moeBound };
    }
  })();

  const data = [
    { name: 'DP/FSDP', threshold: ALPHA, unit: 'per-device token batch' },
    { name: 'Tensor (F/α)', threshold: tensorLimit, unit: 'max sharding degree' },
    { name: 'MoE (B_crit)', threshold: moeCrit, unit: 'token batch' },
  ].map((d) => ({ ...d, current: 0 }));

  const currentRollup = (() => {
    if (strategy === 'dp') return { label: 'Your per-device batch', value: Bdev, unit: 'tokens' };
    if (strategy === 'tensor') return { label: 'Your sharding degree Y', value: Y, unit: 'chips' };
    if (strategy === 'pipeline') return { label: 'Stages', value: Y, unit: 'chips' };
    return { label: 'Your per-device batch', value: Bdev, unit: 'tokens' };
  })();

  return (
    <LessonShell
      number={9}
      title="Distributing the Model"
      subtitle="Data, tensor, pipeline and expert parallelism — and the roofline rules that say when each becomes communication-bound."
      sourceRef="reference/scaling-book/training.md — Data / Tensor / Pipeline Parallelism · reference/scaling-book/gpus.md — Expert Parallelism"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="fsdp" />
          <ConceptTag id="tensor-parallelism" />
          <ConceptTag id="pipeline-parallelism" />
          <ConceptTag id="collectives" />
          <ConceptTag id="moe" />
        </div>

        <div className="flex flex-wrap gap-2">
          {(Object.keys(STRATS) as Strategy[]).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setStrategy(s)}
              className={cn('rounded-xl px-3 py-1.5 text-xs font-semibold transition-colors', strategy === s ? 'text-white' : 'glass-chip text-slate-600 hover:bg-white/80')}
              style={strategy === s ? { background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' } : undefined}
            >
              {STRATS[s].name}
            </button>
          ))}
        </div>

        <p className="text-sm leading-relaxed text-slate-600">{STRATS[strategy].desc}</p>

        {strategy === 'dp' && (
          <SliderControl label="Per-device token batch (B/X)" value={Bdev} min={1} max={8000} step={1} onChange={setBdev} unit="tokens" logScale conceptId="fsdp" />
        )}
        {strategy === 'tensor' && (
          <>
            <SliderControl label="Sharding degree Y" value={Y} min={1} max={64} step={1} onChange={setY} unit="chips" conceptId="tensor-parallelism" />
            <SliderControl label="Feed-forward dim F" value={F} min={4096} max={65536} step={1024} onChange={setF} unit="" conceptId="network-roofline" />
          </>
        )}
        {strategy === 'pipeline' && (
          <SliderControl label="Pipeline stages" value={Y} min={2} max={64} step={1} onChange={setY} unit="chips" conceptId="pipeline-parallelism" />
        )}
        {strategy === 'moe' && (
          <>
            <SliderControl label="Per-device token batch (B)" value={Bdev} min={1} max={8000} step={1} onChange={setBdev} unit="tokens" logScale conceptId="critical-batch" />
            <div className="grid grid-cols-2 gap-2">
              <SliderControl label="Experts (E)" value={E} min={4} max={512} step={4} onChange={setE} unit="" conceptId="moe" />
              <SliderControl label="Active (k)" value={k} min={1} max={16} step={1} onChange={setK} unit="" conceptId="moe" />
            </div>
          </>
        )}

        <div className="glass rounded-xl p-4 flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] text-slate-400">{currentRollup.label}</div>
            <div className="text-xl font-mono font-bold text-slate-800">{currentRollup.value.toLocaleString()} <span className="text-xs font-normal text-slate-500">{currentRollup.unit}</span></div>
          </div>
          <div className={cn('text-right text-sm font-bold rounded-xl px-3 py-2', regime.optimistic ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700')}>
            {regime.optimistic ? 'Compute-bound' : 'Communication-bound'}
          </div>
        </div>
        <p className="text-[13px] text-slate-600 leading-relaxed">{regime.text}</p>
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="font-bold text-slate-800 mb-3">Boundaries to remember</h3>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis scale="log" domain={[10, 80000]} type="number" tickFormatter={(v) => v.toLocaleString()} label={{ value: 'Boundary (log)', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [Number(v).toLocaleString(), 'Boundary']} />
              <Bar dataKey="threshold" fill="#c7d2fe" radius={[6, 6, 0, 0]}>
                <Cell fill="#5b7cfa" />
                <Cell fill="#7f6bf0" />
                <Cell fill="#f59e0b" />
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <ul className="text-[12.5px] text-slate-600 space-y-1.5 mt-3">
          <li><strong>DP/FSDP:</strong> compute-bound only when per-device token batch ≳ 2550 (v5p ICI) — otherwise bandwidth-bound.</li>
          <li><strong>Tensor:</strong> keep sharding ≤ F/2550; past that every layer's AllGather/ReduceScatter dominates the critical path.</li>
          <li><strong>Expert/MoE:</strong> sparsity E/k inflates B_crit to ≈ 120·(E/k) — serving MoE needs very large batches.</li>
          <li><strong>Pipeline:</strong> nearly free communication, but fill the pipeline bubble with microbatches.</li>
        </ul>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'Data parallelism (DP/FSDP) becomes communication-bound when…',
            options: [
              'The per-device token batch is large',
              'The per-device token batch is small (below ~2550 on a v5p)',
              'The sharding degree exceeds F/2550',
              'There is only one chip',
            ],
            answer: 1,
            explain: 'Small batches leave too little math to hide the AllReduce/AllGather, so comms dominate — the roofline rule is B/X > C/W_ici.',
          },
          {
            q: 'Why does MoE raise the critical generation batch so dramatically?',
            options: [
              'Because experts need int8 math',
              'Because only 1/E of weights are active, so sparsity E/k multiplies B_crit (≈ 120·E/k)',
              'Because MoE removes the KV cache',
              'Because experts run on the vector unit',
            ],
            answer: 1,
            explain: 'Most weights sit idle for a given token; you must batch enough concurrent tokens to amortize the active FLOPs — for DeepSeek that is B > 3840.',
          },
        ]}
      />
    </LessonShell>
  );
}
