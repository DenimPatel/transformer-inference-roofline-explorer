import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { CheckCircle2, XCircle, ChevronRight } from 'lucide-react';
import { cn } from '../../lib/utils';
import GlassCard from '../ui/GlassCard';

export interface CheckpointQuestion {
  q: string;
  options: string[];
  answer: number; // index of correct option
  explain: string;
}

interface CheckpointProps {
  questions: CheckpointQuestion[];
  onComplete: () => void;
  congrats?: string;
}

export default function Checkpoint({ questions, onComplete, congrats = 'Lesson complete — great work!' }: CheckpointProps) {
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState(false);

  const allAnswered = questions.every((_, i) => answers[i] !== undefined);
  const allCorrect = submitted && questions.every((q, i) => answers[i] === q.answer);
  const score = submitted ? questions.filter((q, i) => answers[i] === q.answer).length : 0;

  return (
    <GlassCard className="p-5 mt-4" shimmer>
      <div className="flex items-center justify-between mb-4">
        <h4 className="font-bold text-slate-700 flex items-center gap-2">
          <span className="text-[var(--color-accent)]">
            <CheckCircle2 style={{ width: 16, height: 16 }} />
          </span>
          Check your understanding
        </h4>
        {submitted && (
          <span className={cn('text-xs font-bold', score === questions.length ? 'text-emerald-600' : 'text-amber-600')}>
            {score}/{questions.length} correct
          </span>
        )}
      </div>

      <div className="space-y-4">
        {questions.map((q, qi) => {
          const picked = answers[qi];
          const revealed = submitted;
          return (
            <div key={qi} className="space-y-1.5">
              <p className="text-sm font-medium text-slate-700">{qi + 1}. {q.q}</p>
              <div className="grid gap-1.5 pl-2">
                {q.options.map((opt, oi) => {
                  const selected = picked === oi;
                  const isCorrect = q.answer === oi;
                  let cls = 'glass-chip px-3 py-2 text-[13px] text-slate-600 hover:bg-white/80';
                  if (!revealed) {
                    if (selected) cls = 'px-3 py-2 text-[13px] rounded-full bg-[var(--color-accent)] text-white';
                  } else {
                    if (isCorrect) cls = 'px-3 py-2 text-[13px] rounded-full bg-emerald-100 text-emerald-700';
                    else if (selected) cls = 'px-3 py-2 text-[13px] rounded-full bg-rose-100 text-rose-700';
                  }
                  return (
                    <button
                      key={oi}
                      type="button"
                      disabled={submitted}
                      onClick={() => setAnswers((a) => ({ ...a, [qi]: oi }))}
                      className={cn('text-left border border-slate-200/60 transition-colors', cls)}
                    >
                      <span className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-current" />
                        {opt}
                        {revealed && isCorrect && <CheckCircle2 className="ml-auto" style={{ width: 14, height: 14 }} />}
                        {revealed && selected && !isCorrect && <XCircle className="ml-auto" style={{ width: 14, height: 14 }} />}
                      </span>
                    </button>
                  );
                })}
              </div>
              <AnimatePresence>
                {revealed && (
                  <motion.p
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className={cn('text-[12px] pl-2', answers[qi] === q.answer ? 'text-emerald-600' : 'text-rose-500')}
                  >
                    {q.explain}
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      <div className="mt-5 flex items-center gap-3 flex-wrap">
        {!submitted && (
          <button
            type="button"
            disabled={!allAnswered}
            onClick={() => setSubmitted(true)}
            className="inline-flex items-center gap-1.5 rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40 transition-opacity"
            style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}
          >
            Check answers
          </button>
        )}
        {allCorrect && (
          <motion.button
            type="button"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={onComplete}
            className="inline-flex items-center gap-1.5 rounded-xl px-4 py-2 text-sm font-semibold text-white"
            style={{ background: 'linear-gradient(135deg, #22c48b, #149263)' }}
          >
            {congrats} <ChevronRight style={{ width: 16, height: 16 }} />
          </motion.button>
        )}
        {submitted && !allCorrect && (
          <button
            type="button"
            onClick={() => {
              setSubmitted(false);
              setAnswers({});
            }}
            className="text-sm text-slate-500 hover:text-slate-700 underline-offset-2 hover:underline"
          >
            Try again
          </button>
        )}
      </div>
    </GlassCard>
  );
}
