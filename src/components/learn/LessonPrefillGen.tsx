import { useState } from 'react';
import { motion } from 'motion/react';
import { Layers, Zap } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import { cn } from '../../lib/utils';

const BLOCKS = 14;

function FlowBlocks({ count, color }: { count: number; color: string }) {
  return (
    <div className="flex gap-1 overflow-hidden">
      {Array.from({ length: count }).map((_, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0.15, scale: 0.85 }}
          animate={{ opacity: [0.15, 1, 0.15], scale: [0.9, 1, 0.9] }}
          transition={{ duration: 1.6, delay: (i % 6) * 0.28, repeat: Infinity, ease: 'easeInOut' }}
          className="h-6 w-5 rounded-md"
          style={{ backgroundColor: color, boxShadow: `0 0 12px ${color}66` }}
        />
      ))}
    </div>
  );
}

export default function LessonPrefillGen({ onComplete }: { onComplete: () => void }) {
  const [go, setGo] = useState(true);

  return (
    <LessonShell
      number={3}
      title="Prefill vs Generation: Why Inference Flips the Roofline"
      subtitle="Inference is two different workloads. Prefill is compute-bound; generation is almost always memory-bound."
      sourceRef="reference/scaling-book/inference.md — What about attention?"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="prefill" />
          <ConceptTag id="generation" />
          <ConceptTag id="kv-cache" />
          <ConceptTag id="memory-bound" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          <strong className="text-slate-800">Prefill</strong> processes a long prompt all at once — weights are reused
          across thousands of tokens, so it saturates the FLOPs. <strong className="text-slate-800">Generation</strong>{' '}
          emits one token at a time and must stream the entire parameter set plus its KV cache every step — so bytes
          dominate. That is why the Interactive Lab's decode model sits below the ridge.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="glass rounded-2xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="text-[var(--color-accent)]" style={{ width: 16, height: 16 }} />
              <h4 className="font-bold text-slate-700">Prefill (compute-bound)</h4>
            </div>
            <p className="text-xs text-slate-500 mb-3">Processes all prompt tokens at once. Weight re-use → high intensity.</p>
            <span className="inline-block px-2 py-1 rounded-full text-[11px] font-bold bg-emerald-100 text-emerald-700 mb-3">✔ Compute-bound</span>
            <FlowBlocks color="#5b7cfa" count={BLOCKS} />
            <p className="text-[11px] text-slate-400 mt-3">Attention intensity ∝ T/2 — way above the ridge.</p>
          </div>

          <div className="glass rounded-2xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="text-amber-500" style={{ width: 16, height: 16 }} />
              <h4 className="font-bold text-slate-700">Generation (memory-bound)</h4>
            </div>
            <p className="text-xs text-slate-500 mb-3">One token per step; streams all weights + KV cache each time.</p>
            <span className="inline-block px-2 py-1 rounded-full text-[11px] font-bold bg-amber-100 text-amber-700 mb-3">⚠️ Memory-bound</span>
            <FlowBlocks color="#f5a623" count={BLOCKS} />
            <p className="text-[11px] text-slate-400 mt-3">Attention intensity ≈ ST/(S+T) ≈ 1 — constant & below the ridge.</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setGo((g) => !g)}
            className="rounded-xl px-4 py-2 text-sm font-semibold text-white"
            style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}
          >
            {go ? 'Pause animation' : 'Play animation'}
          </button>
          {go && (
            <span className={cn('text-xs text-emerald-500 font-medium')}>Tokens flowing…</span>
          )}
        </div>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'Why is generation almost always memory-bound?',
            options: [
              'It does huge amounts of math per token',
              'Each step streams the full parameter set + KV cache for just one new token',
              'The GPU is too small',
              'Attention is compute-heavy',
            ],
            answer: 1,
            explain: 'Generating one token means loading all parameters + KV cache (bytes) for a tiny amount of math (FLOPs).',
          },
          {
            q: 'What makes prefill compute-bound?',
            options: [
              'It runs on CPUs',
              'Prompts are short',
              'Weights are reused across many tokens (high arithmetic intensity)',
              'KV cache is small',
            ],
            answer: 2,
            explain: 'Prefill weights are amortized over the whole prompt, pushing intensity above the ridge.',
          },
        ]}
      />
    </LessonShell>
  );
}
