import { useEffect, useState } from 'react';
import type { ComponentType, CSSProperties } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  GraduationCap, Gauge, Percent, Layers, Timer, Database, DollarSign, SlidersHorizontal, ArrowLeft, ArrowRight, Play, CheckCircle2, FlaskConical,
  MemoryStick, Network, BrainCircuit, Workflow, Cpu, Grid3x3,
} from 'lucide-react';
import GlassCard from '../ui/GlassCard';
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
              className="inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold text-white"
              style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}
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
              className="inline-flex items-center gap-1.5 rounded-xl px-4 py-2 text-sm font-semibold text-white"
              style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}
            >
              Next: {LESSONS[idx + 1].title} <ArrowRight style={{ width: 16, height: 16 }} />
            </button>
          ) : (
            <button
              type="button"
              onClick={onLab}
              className="inline-flex items-center gap-1.5 rounded-xl px-4 py-2 text-sm font-semibold text-white"
              style={{ background: 'linear-gradient(135deg, #22c48b, #149263)' }}
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

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="max-w-5xl mx-auto pt-4 pb-16">
      <GlassCard className="p-6 mb-6 shimmer" hover>
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <div className="flex items-center justify-center w-14 h-14 rounded-2xl text-white shrink-0"
            style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}>
            <GraduationCap style={{ width: 26, height: 26 }} />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Learn how transformer inference really works</h2>
            <p className="text-sm text-slate-500 mt-1">
              A guided, interactive intro to the roofline, arithmetic intensity, batch size, latency and cost — drawn from the{' '}
              <em>How to Scale Your Model</em> reference. Complete the checkpoints, then experiment in the Lab.
            </p>
          </div>
        </div>
        <div className="mt-5">
          <div className="flex justify-between text-xs mb-1.5">
            <span className="text-slate-500 font-medium">Progress</span>
            <span className="font-mono text-slate-600">{done}/{LESSONS.length} · {pct}%</span>
          </div>
          <div className="h-2.5 rounded-full bg-slate-200/60 overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ background: 'linear-gradient(90deg, #5b7cfa, #22c48b)' }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>
        </div>
      </GlassCard>

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
                <span className="flex items-center justify-center w-9 h-9 rounded-xl text-white shrink-0"
                  style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}>
                  <Comp style={{ width: 18, height: 18 }} />
                </span>
                <div className="flex-1">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Lesson {l.number} · ~{l.minutes} min</div>
                  <div className="font-bold text-slate-800 leading-tight">{l.title}</div>
                </div>
                {isDone ? (
                  <CheckCircle2 className="text-emerald-500 shrink-0" style={{ width: 20, height: 20 }} />
                ) : (
                  <span className="flex items-center justify-center w-8 h-8 rounded-full bg-white/70 text-[var(--color-accent)] shrink-0">
                    <Play style={{ width: 16, height: 16 }} />
                  </span>
                )}
              </div>
              <p className="text-[13px] text-slate-500 leading-relaxed">{l.summary}</p>
              <div className={cn('mt-3 h-1 rounded-full', isDone ? 'bg-emerald-400' : 'bg-slate-200/70')}>
                {isDone && <div className="h-full rounded-full bg-emerald-400" style={{ width: '100%' }} />}
              </div>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
}
