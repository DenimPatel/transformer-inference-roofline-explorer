import { useMemo, useState } from 'react';
import { ComposedChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { SystolicArray } from '../Hardware';
import { cn } from '../../lib/utils';

// TPU v5e figures from the scaling book, used as the worked example.
const MXU_FLOPS = 1.97e14;   // bf16 FLOPs/s
const VPU_FLOPS = 7e12;      // per-core vector unit
const HBM_BW = 8.2e11;       // bytes/s

const MXU_RIDGE = MXU_FLOPS / HBM_BW;  // ~240
const VPU_RIDGE = VPU_FLOPS / HBM_BW;  // ~8.5 (≈3 per core in the book's framing)

export default function LessonInsideChip({ onComplete }: { onComplete: () => void }) {
  const [n, setN] = useState(128);

  // An n x n weight-stationary array does ~n MACs per byte streamed in.
  const arrayIntensity = n;

  const ops = useMemo(() => ([
    { name: 'Dot product', intensity: 0.5, unit: 'VPU', color: '#ef4444' },
    { name: 'Softmax / layernorm', intensity: 1.5, unit: 'VPU', color: '#f59e0b' },
    { name: 'Matmul, B=64', intensity: 64, unit: 'MXU', color: '#3b82f6' },
    { name: 'Matmul, B=1024', intensity: 1024, unit: 'MXU', color: '#8b5cf6' },
  ]), []);

  return (
    <LessonShell
      number={12}
      title="Inside the Chip"
      subtitle="The MXU manufactures arithmetic intensity; the VPU has a ridge of its own. Knowing which unit runs your op is half of diagnosing it."
      sourceRef="reference/scaling-book/tpus.md — What Is a TPU? and Appendix B · reference/scaling-book/gpus.md — What Is a GPU?"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="systolic-array" />
          <ConceptTag id="tpu-architecture" />
          <ConceptTag id="arithmetic-intensity" />
          <ConceptTag id="ridge-point" />
          <ConceptTag id="vector-unit-ridge" />
          <ConceptTag id="dot-product-intensity" />
          <ConceptTag id="sm-streaming-multiprocessor" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          Every ridge point you have used so far is a ratio of two hardware numbers. This lesson goes and finds them. A
          TPU&rsquo;s <strong>TensorCore</strong> holds three things that matter: the <strong>MXU</strong> (a systolic
          array that does the matmuls), the <strong>VPU</strong> (a much smaller vector unit for elementwise work), and{' '}
          <strong>VMEM</strong> (the on-chip scratchpad feeding both). A GPU has the same three jobs, spread across
          hundreds of small SMs instead of one or two big cores.
        </p>

        <div className="grid sm:grid-cols-2 gap-3">
          <div className="glass rounded-xl p-4">
            <div className="text-[11px] uppercase tracking-wide text-slate-400">Matrix unit (MXU)</div>
            <div className="font-mono font-bold text-slate-900 text-lg mt-1">{(MXU_FLOPS / 1e12).toFixed(0)} TFLOP/s</div>
            <div className="text-xs text-slate-500 mt-1">
              ridge = <span className="font-mono font-bold text-rose-600">{MXU_RIDGE.toFixed(0)}</span> FLOPs/byte
            </div>
          </div>
          <div className="glass rounded-xl p-4">
            <div className="text-[11px] uppercase tracking-wide text-slate-400">Vector unit (VPU)</div>
            <div className="font-mono font-bold text-slate-900 text-lg mt-1">{(VPU_FLOPS / 1e12).toFixed(0)} TFLOP/s</div>
            <div className="text-xs text-slate-500 mt-1">
              ridge = <span className="font-mono font-bold text-amber-600">{VPU_RIDGE.toFixed(1)}</span> FLOPs/byte
            </div>
          </div>
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          Those are <em>two different rooflines on one chip</em>. A softmax judged against 240 looks like a disaster;
          judged against its own unit it is merely bad. And a <strong>dot product</strong> —{' '}
          <code className="text-xs">I = (2N&minus;1)/(4N+2) → 1/2</code> — fails both, at any N, forever. Low-intensity
          ops cannot be fixed by batching; they can only be <em>fused</em> so the bytes never move.
        </p>
      </GlassCard>

      <GlassCard className="p-5 space-y-4">
        <h3 className="font-bold text-slate-800">The systolic array: where the FLOPs actually happen</h3>
        <p className="text-sm leading-relaxed text-slate-600">
          Each cell holds one weight permanently. Activations march in from the left, partial sums march down, and
          neighbouring cells hand results to each other without ever touching memory. Watch the diagonal wavefront fill
          the grid — after that, every cell does a multiply-accumulate on every cycle.
        </p>
        <SystolicArray n={4} rows={6} />

        <SliderControl
          label="Array dimension (n x n)"
          value={n}
          min={8}
          max={512}
          step={8}
          onChange={setN}
          unit=""
          logScale
          conceptId="systolic-array"
        />
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">MACs per cycle</div>
            <div className="font-mono font-bold text-slate-800">{(n * n).toLocaleString()}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">New values loaded</div>
            <div className="font-mono font-bold text-slate-800">{n}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">Structural intensity</div>
            <div className="font-mono font-bold text-slate-800">≈ {arrayIntensity}</div>
          </div>
        </div>
        <div className={cn('text-sm font-semibold rounded-xl px-3 py-2',
          arrayIntensity >= MXU_RIDGE ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700')}>
          {arrayIntensity >= MXU_RIDGE
            ? `An ${n}x${n} array can manufacture ~${n} FLOPs per byte — enough to sit above the ${MXU_RIDGE.toFixed(0)} ridge, if you keep it fed.`
            : `An ${n}x${n} array tops out around ${n} FLOPs per byte, below the ${MXU_RIDGE.toFixed(0)} ridge. This is why real MXUs are 128x128 or 256x256.`}
        </div>
        <p className="text-xs text-slate-400">
          <i>O(n²)</i> FLOPs per cycle against <i>O(n)</i> loads: the array&rsquo;s shape <em>creates</em> arithmetic
          intensity. That the answer lands near the ridge point is no accident — designers size the array so a well-fed
          matmul just saturates HBM.
        </p>
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="font-bold text-slate-800 mb-3">Which ridge applies to which op</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={ops} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis scale="log" domain={[0.3, 4000]} type="number"
                tickFormatter={(v: any) => (v < 1 ? v.toFixed(1) : v.toFixed(0))}
                label={{ value: 'Intensity (FLOPs/B, log)', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any, _n: any, p: any) => [`${Number(v).toFixed(1)} FLOPs/B`, `Runs on the ${p.payload.unit}`]} />
              <ReferenceLine y={VPU_RIDGE} stroke="#f59e0b" strokeDasharray="3 3"
                label={{ position: 'top', value: `VPU ridge ${VPU_RIDGE.toFixed(0)}`, fill: '#f59e0b', fontSize: 10 }} />
              <ReferenceLine y={MXU_RIDGE} stroke="#f43f5e" strokeDasharray="3 3"
                label={{ position: 'top', value: `MXU ridge ${MXU_RIDGE.toFixed(0)}`, fill: '#f43f5e', fontSize: 10 }} />
              <Bar dataKey="intensity" radius={[6, 6, 0, 0]}>
                {ops.map((o) => <Cell key={o.name} fill={o.color} />)}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-slate-400 mt-3">
          The two elementwise ops sit below <em>both</em> lines, so no hardware choice rescues them. The matmuls climb
          with batch, which is the entire content of the B-rule you learned in Lesson 2.
        </p>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'A dot product has arithmetic intensity 1/2. Which ridge should you compare it against?',
            options: [
              'The MXU ridge (~240), because that is the chip’s headline number',
              'The vector-unit ridge, because elementwise and reduction ops run on the VPU, not the MXU',
              'Neither — dot products do not have an arithmetic intensity',
              'The inter-chip network ridge',
            ],
            answer: 1,
            explain: 'Reductions and elementwise math run on the VPU / CUDA cores, which have far lower peak FLOPs against the same HBM and therefore a much smaller ridge (~3 per core on a v5p). The dot product is comms-bound either way, but using the wrong ridge mis-diagnoses every softmax and layernorm in your model.',
          },
          {
            q: 'Why does an n x n systolic array have high arithmetic intensity by construction?',
            options: [
              'It performs O(n²) multiply-accumulates per cycle while loading only O(n) new values',
              'It stores the entire model on-chip',
              'It skips the additions and only does multiplies',
              'It runs at a higher clock speed than the vector unit',
            ],
            answer: 0,
            explain: 'Weights stay resident in the cells and results pass directly between neighbours, so the array does n² MACs per cycle against n new inputs — roughly n FLOPs per byte. That is why a 128x128 MXU lands near the same order as the ridge point itself.',
          },
          {
            q: 'You are decoding with batch size 1 on a 128x128 MXU. What is happening in the array?',
            options: [
              'The array is fully utilized because the weights are already loaded',
              'Almost every cell is idle almost every cycle, yet you still paid to load all the weights',
              'The array switches to running on the VPU instead',
              'The array processes 128 tokens in parallel automatically',
            ],
            answer: 1,
            explain: 'One row through a 128-wide grid leaves the array nearly empty while the full weight load is still charged to HBM. "Decode is memory-bound" and "the MXU is starved at batch 1" are the same fact seen from the equation and from the silicon.',
          },
        ]}
      />
    </LessonShell>
  );
}
