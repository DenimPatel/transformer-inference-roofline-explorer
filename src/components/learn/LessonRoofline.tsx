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
          A deep learning model is just a stack of matrix multiplications. Two things compete for time:{' '}
          <strong className="text-slate-800">doing the math</strong> (T<sub>math</sub> = FLOPs ÷ FLOPs/s) and{' '}
          <strong className="text-slate-800">moving the data</strong> (T<sub>comms</sub> = Bytes ÷ Bandwidth). Because
          they can overlap, an operation runs in ~max(T<sub>math</sub>, T<sub>comms</sub>).
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
