import { useState } from 'react';
import GlassCard from '../ui/GlassCard';
import InfoPopover from '../ui/InfoPopover';
import { ConceptTag } from '../ui/ConceptTag';
import InteractiveRoofline from './InteractiveRoofline';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';

const PEAK_FLOPS = 9.89e14; // H100 bf16 (without sparsity)
const PEAK_BW = 3.35e12;    // H100 HBM

export default function LessonRoofline({ onComplete }: { onComplete: () => void }) {
  const [intensity, setIntensity] = useState(1);
  const ridge = PEAK_FLOPS / PEAK_BW;

  return (
    <LessonShell
      number={1}
      title="The Roofline: Why Does Inference Take Time?"
      subtitle="An operation is bounded by how fast we do math, how fast we move bytes, and how much memory we have."
      sourceRef="reference/scaling-book/roofline.md — Where Does the Time Go?"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="flops" />
          <ConceptTag id="bandwidth" />
          <ConceptTag id="roofline" />
          <ConceptTag id="ridge-point" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          Running anything on an accelerator is bounded by exactly <strong>three</strong> hardware facts: how fast it can
          do math (<span className="font-mono text-xs">OPs/second</span>), how fast it can move bytes
          (<span className="font-mono text-xs">bytes/second</span>), and how much it can hold at all
          (<span className="font-mono text-xs">bytes</span>). The first two set your <em>speed</em>; the third decides
          whether the job runs on this chip at all.
        </p>

        <div className="grid sm:grid-cols-3 gap-2">
          {[
            { k: 'Math speed', v: `${(PEAK_FLOPS / 1e12).toFixed(0)} TFLOP/s`, d: 'T_math = FLOPs ÷ FLOPs/s' },
            { k: 'Bandwidth', v: `${(PEAK_BW / 1e12).toFixed(2)} TB/s`, d: 'T_comms = Bytes ÷ bytes/s' },
            { k: 'Capacity', v: '80 GB', d: 'A hard wall, not a slope' },
          ].map((x) => (
            <div key={x.k} className="glass rounded-xl px-3 py-2">
              <div className="text-[11px] text-slate-400">{x.k}</div>
              <div className="font-mono font-bold text-slate-800">{x.v}</div>
              <div className="text-[10px] text-slate-400 mt-0.5 font-mono">{x.d}</div>
            </div>
          ))}
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          The first two compete for time, and they can usually overlap, so an operation runs in about{' '}
          <strong className="text-slate-800">max(T<sub>math</sub>, T<sub>comms</sub>)</strong> — with{' '}
          T<sub>math</sub> + T<sub>comms</sub> as the no-overlap upper bound. Since{' '}
          <i>a + b ≤ 2·max(a, b)</i>, those two bounds are never more than <strong>2&times;</strong> apart, which is why
          the simple max is good enough to reason with.
        </p>

        <p className="text-sm leading-relaxed text-slate-600">
          <strong>Capacity behaves differently.</strong> It is not a rate you can trade against, it is a wall: exceed
          HBM and the model simply does not run, no matter how favourable your intensity. That is why the KV cache
          (Lesson 5) so often decides your deployment before the roofline ever gets a say.
        </p>

        <p className="text-sm leading-relaxed text-slate-600">
          The <strong className="text-slate-800">roofline plot</strong> draws the best FLOPs/s you can achieve for a
          given arithmetic intensity. Below the <span className="text-rose-500 font-semibold">ridge</span> you are
          <em> memory-bound</em> (waiting on bytes); at or above it you are <em>compute-bound</em> (using the FLOPs).
        </p>

        <div>
          <p className="text-sm font-medium text-slate-700 mb-2">
            Drag the operating point across the ridge to feel the crossover (H100):
          </p>
          <InteractiveRoofline
            peakFlops={PEAK_FLOPS}
            peakBw={PEAK_BW}
            intensity={intensity}
            onIntensityChange={(i) => setIntensity(Math.max(0.05, Math.min(200000, i)))}
            opLabel="Your op"
          />
        </div>

        <p className="text-[12px] text-slate-400">
          For this H100 the ridge sits at ≈{ridge.toFixed(0)} FLOPs/B. Anything below it wastes FLOPs — which is exactly
          what happens to most transformer <em>generation</em> steps (see Lesson 3).
        </p>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'Rooflines describe three hardware limits. Which one is a hard wall rather than a rate you trade against?',
            options: [
              'Peak FLOPs/s',
              'Memory bandwidth',
              'Memory capacity',
              'Arithmetic intensity',
            ],
            answer: 2,
            explain: 'FLOPs/s and bandwidth are rates — they set how long an op takes, and you trade between them via arithmetic intensity. Capacity is binary: if the weights plus KV cache exceed HBM, the job does not run at all.',
          },
          {
            q: 'Which resource bounds you when you are memory-bound?',
            options: ['Math speed (FLOPs/s)', 'Bytes moved per second', 'Number of layers', 'The optimizer'],
            answer: 1,
            explain: 'Memory-bound means bandwidth (bytes/s) is the bottleneck — you are waiting on data, wasting FLOPs.',
          },
          {
            q: "The ridge point is defined as…",
            options: [
              'Number of parameters ÷ context length',
              'Peak FLOPs/s ÷ bandwidth',
              'Batch size ÷ KV cache',
              'Latency × throughput',
            ],
            answer: 1,
            explain: 'I_hw = Peak FLOPs/s ÷ bandwidth — the intensity where you saturate the accelerator.',
          },
        ]}
      />
    </LessonShell>
  );
}
