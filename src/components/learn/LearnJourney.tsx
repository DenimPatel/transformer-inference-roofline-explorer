import { useEffect, useState } from 'react';
import type { ComponentType, CSSProperties } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Gauge, Percent, Layers, Timer, Database, DollarSign, SlidersHorizontal, ArrowLeft, ArrowRight, Play, CheckCircle2, FlaskConical,
  MemoryStick, Network, BrainCircuit, Workflow, Cpu, Grid3x3,
} from 'lucide-react';
import LessonRoofline from './LessonRoofline';
import LessonIntensity from './LessonIntensity';
import LessonPrefillGen from './LessonPrefillGen';
import LessonPareto from './LessonPareto';
import LessonKVCache from './LessonKVCache';
import LessonCost from './LessonCost';
import LessonQuant from './LessonQuant';
import LessonMemory from './LessonMemory';
import LessonSharding from './LessonSharding';
import LessonAttention from './LessonAttention';
import LessonServing from './LessonServing';
import LessonInsideChip from './LessonInsideChip';
import LessonNetworkRoofline from './LessonNetworkRoofline';
import { cn } from '../../lib/utils';
import { HARDWARE_PROFILES } from '../../lib/hardware';
import { MODEL_PROFILES } from '../../lib/models';
import { CONCEPTS } from '../../lib/concepts';

const STORAGE_KEY = 'roofline-learn-progress';

interface LessonMeta {
  id: string;
  number: number;
  title: string;
  summary: string;
  icon: ComponentType<{ style?: CSSProperties; className?: string }>;
  minutes: number;
}

const LESSONS: LessonMeta[] = [
  { id: 'roofline', number: 1, title: 'The Roofline', summary: 'Why inference is bounded by math speed, bandwidth, and memory — and what the ridge is.', icon: Gauge, minutes: 5 },
  { id: 'intensity', number: 2, title: 'Arithmetic Intensity & the B Rule', summary: 'How much math you get per byte, and why a matmul is compute-bound iff B > ridge.', icon: Percent, minutes: 6 },
  { id: 'prefillgen', number: 3, title: 'Prefill vs Generation', summary: 'Why prefill is compute-bound but generation is almost always memory-bound.', icon: Layers, minutes: 5 },
  { id: 'pareto', number: 4, title: 'Latency vs Throughput', summary: 'The critical batch tradeoff between speed per request and tokens per second.', icon: Timer, minutes: 6 },
  { id: 'kvcache', number: 5, title: 'The KV Cache', summary: 'Where inference memory actually goes — and how GQA shrinks it.', icon: Database, minutes: 5 },
  { id: 'cost', number: 6, title: 'Total Cost of Ownership', summary: 'Electricity + amortized hardware, and how batch and location move price.', icon: DollarSign, minutes: 6 },
  { id: 'quant', number: 7, title: 'Quantization & the β Rule', summary: 'How precision shifts the critical batch size.', icon: SlidersHorizontal, minutes: 5 },
  { id: 'memory', number: 8, title: 'Memory Hierarchy & the On-Chip Wall', summary: 'Why on-chip VMEM is ~22x faster than HBM, and how tiling reshapes intensity.', icon: MemoryStick, minutes: 6 },
  { id: 'sharding', number: 9, title: 'Distributing the Model', summary: 'Data, tensor, pipeline and expert parallelism — and when each becomes comms-bound.', icon: Network, minutes: 7 },
  { id: 'attention', number: 10, title: 'Attention Deep Dive', summary: 'Why attention flips from compute-bound prefill to always memory-bound generation.', icon: BrainCircuit, minutes: 6 },
  { id: 'serving', number: 11, title: 'Serving Systems', summary: 'TTFT, continuous batching, disaggregation, prefix caching and speculative decoding.', icon: Workflow, minutes: 7 },
  { id: 'insidechip', number: 12, title: 'Inside the Chip', summary: 'The MXU, the systolic array, and why the vector unit has a ridge of its own.', icon: Grid3x3, minutes: 7 },
  { id: 'networkroofline', number: 13, title: 'The Network Roofline', summary: 'When the fabric becomes the roof — and why batching cannot fix it.', icon: Cpu, minutes: 7 },
];

function loadProgress(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

export default function LearnJourney({ onLab }: { onLab: () => void }) {
  const [completed, setCompleted] = useState<string[]>(loadProgress);
  const [current, setCurrent] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(completed));
  }, [completed]);

  const currentMeta = LESSONS.find((l) => l.id === current);
  const idx = currentMeta ? LESSONS.indexOf(currentMeta) : -1;

  const markComplete = (id: string) => {
    setCompleted((c) => (c.includes(id) ? c : [...c, id]));
  };

  const renderLesson = (id: string) => {
    switch (id) {
      case 'roofline': return <LessonRoofline onComplete={() => markComplete('roofline')} />;
      case 'intensity': return <LessonIntensity onComplete={() => markComplete('intensity')} />;
      case 'prefillgen': return <LessonPrefillGen onComplete={() => markComplete('prefillgen')} />;
      case 'pareto': return <LessonPareto onComplete={() => markComplete('pareto')} />;
      case 'kvcache': return <LessonKVCache onComplete={() => markComplete('kvcache')} />;
      case 'cost': return <LessonCost onComplete={() => markComplete('cost')} />;
      case 'quant': return <LessonQuant onComplete={() => markComplete('quant')} />;
      case 'memory': return <LessonMemory onComplete={() => markComplete('memory')} />;
      case 'sharding': return <LessonSharding onComplete={() => markComplete('sharding')} />;
      case 'attention': return <LessonAttention onComplete={() => markComplete('attention')} />;
      case 'serving': return <LessonServing onComplete={() => markComplete('serving')} />;
      case 'insidechip': return <LessonInsideChip onComplete={() => markComplete('insidechip')} />;
      case 'networkroofline': return <LessonNetworkRoofline onComplete={() => markComplete('networkroofline')} />;
      default: return null;
    }
  };

  // ---- Detail view ----
  if (currentMeta) {
    return (
      <div className="max-w-4xl mx-auto pt-2 pb-16">
        <div className="flex items-center justify-between mb-4">
          <button
            type="button"
            onClick={() => setCurrent(null)}
            className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-700"
          >
            <ArrowLeft style={{ width: 16, height: 16 }} /> All lessons
          </button>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">
              {completed.includes(currentMeta.id) ? 'Completed' : 'Not yet'}
            </span>
            {completed.includes(currentMeta.id) && (
              <CheckCircle2 className="text-emerald-500" style={{ width: 16, height: 16 }} />
            )}
            <button
              type="button"
              onClick={onLab}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-[var(--color-accent)] hover:bg-[var(--color-accent-600)]"

            >
              <FlaskConical style={{ width: 14, height: 14 }} /> Open in Lab
            </button>
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={currentMeta.id}
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -24 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            {renderLesson(currentMeta.id)}
          </motion.div>
        </AnimatePresence>

        <div className="flex items-center justify-between mt-8">
          <button
            type="button"
            disabled={idx <= 0}
            onClick={() => setCurrent(LESSONS[idx - 1].id)}
            className="inline-flex items-center gap-1.5 text-sm font-medium text-slate-600 disabled:opacity-30"
          >
            <ArrowLeft style={{ width: 16, height: 16 }} /> {idx > 0 ? LESSONS[idx - 1].title : 'Start'}
          </button>
          {idx < LESSONS.length - 1 ? (
            <button
              type="button"
              onClick={() => setCurrent(LESSONS[idx + 1].id)}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-sm font-semibold text-white bg-[var(--color-accent)] hover:bg-[var(--color-accent-600)]"

            >
              Next: {LESSONS[idx + 1].title} <ArrowRight style={{ width: 16, height: 16 }} />
            </button>
          ) : (
            <button
              type="button"
              onClick={onLab}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-sm font-semibold text-white bg-[var(--color-accent-2)] hover:bg-[var(--color-accent-2-700)]"

            >
              <FlaskConical style={{ width: 15, height: 15 }} /> Put it together in the Lab
            </button>
          )}
        </div>
      </div>
    );
  }

  // ---- Overview ----
  const done = completed.length;
  const pct = Math.round((done / LESSONS.length) * 100);
  const readingMinutes = LESSONS.reduce((sum, l) => sum + l.minutes, 0);

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="max-w-5xl mx-auto pt-4 pb-16">
      {/* Masthead */}
      <section className="hero">
        <span className="section-kicker section-kicker--accent">A working model</span>
        <h1>
          Transformer inference,<br />priced by the byte.
        </h1>
        <p>
          Why a decoder spends its life waiting on memory, what a batch actually buys you, and
          where the money goes — worked through interactively, one lesson at a time, then left
          open for you to experiment in the Lab.
        </p>
      </section>

      <section className="pb-8">
        <div className="stats" />
        <p className="stats-row">
          <span>Roofline analysis</span>
          <span>Compiled by Denim Patel</span>
          <span>After &ldquo;How to Scale Your Model&rdquo;</span>
          <span>{readingMinutes} min read</span>
        </p>
        <div className="stats-rule" />
        <div className="stats-grid">
          <p className="stat">
            <span>Lessons</span><span className="stat-fill" />
            <span className="stat-value stat-value--accent">{LESSONS.length}</span>
          </p>
          <p className="stat">
            <span>Lessons completed</span><span className="stat-fill" />
            <span className="stat-value">{done}</span>
          </p>
          <p className="stat">
            <span>Hardware profiles</span><span className="stat-fill" />
            <span className="stat-value">{HARDWARE_PROFILES.length}</span>
          </p>
          <p className="stat">
            <span>Model profiles</span><span className="stat-fill" />
            <span className="stat-value">{MODEL_PROFILES.length}</span>
          </p>
          <p className="stat">
            <span>Concepts glossed</span><span className="stat-fill" />
            <span className="stat-value">{CONCEPTS.length}</span>
          </p>
          <p className="stat">
            <span>Progress</span><span className="stat-fill" />
            <span className="stat-value">{pct}%</span>
          </p>
        </div>
        <div className="stats-rule" />
        <div className="h-1 bg-slate-200 mt-4 overflow-hidden">
          <motion.div
            className="h-full"
            style={{ background: 'var(--color-accent)' }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          />
        </div>
      </section>

      <span className="section-kicker">The course</span>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {LESSONS.map((l, i) => {
          const Comp = l.icon;
          const isDone = completed.includes(l.id);
          return (
            <motion.button
              key={l.id}
              type="button"
              onClick={() => setCurrent(l.id)}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.4 }}
              className="glass-card glass-card-hover shimmer p-5 text-left relative overflow-hidden"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="flex items-center justify-center w-9 h-9 shrink-0 text-[var(--color-accent-700)] border border-[var(--color-divider)] bg-[var(--color-accent-100)]">
                  <Comp style={{ width: 18, height: 18 }} />
                </span>
                <div className="flex-1">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Lesson {l.number} · ~{l.minutes} min</div>
                  <div className="font-bold text-slate-800 leading-tight">{l.title}</div>
                </div>
                {isDone ? (
                  <CheckCircle2 className="text-emerald-500 shrink-0" style={{ width: 20, height: 20 }} />
                ) : (
                  <span className="flex items-center justify-center w-8 h-8 rounded-full bg-[var(--color-surface)] text-[var(--color-accent)] shrink-0">
                    <Play style={{ width: 16, height: 16 }} />
                  </span>
                )}
              </div>
              <p className="text-[13px] text-slate-500 leading-relaxed">{l.summary}</p>
              <div className={cn('mt-3 h-1 rounded-full', isDone ? 'bg-emerald-500' : 'bg-slate-200')}>
                {isDone && <div className="h-full rounded-full bg-emerald-500" style={{ width: '100%' }} />}
              </div>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
}
