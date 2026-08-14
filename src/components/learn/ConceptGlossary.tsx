import { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Search, X, BookOpen, ChevronDown } from 'lucide-react';
import { CONCEPTS } from '../../lib/concepts';
import { cn } from '../../lib/utils';

interface ConceptGlossaryProps {
  open: boolean;
  onClose: () => void;
}

export default function ConceptGlossary({ open, onClose }: ConceptGlossaryProps) {
  const [query, setQuery] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return CONCEPTS;
    return CONCEPTS.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        c.summary.toLowerCase().includes(q) ||
        c.id.toLowerCase().includes(q) ||
        c.sourceRef.toLowerCase().includes(q)
    );
  }, [query]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Concept glossary"
    >
      <div className="absolute inset-0 bg-slate-900/30 backdrop-blur-sm" onClick={onClose} />
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="glass-strong relative w-full max-w-2xl max-h-[82vh] flex flex-col rounded-3xl p-6"
      >
        <div className="flex items-start justify-between mb-4 gap-3">
          <div className="flex items-center gap-2.5">
            <span className="text-[var(--color-accent)]">
              <BookOpen style={{ width: 20, height: 20 }} />
            </span>
            <div>
              <h2 className="text-lg font-bold text-slate-800">Concept Glossary</h2>
              <p className="text-xs text-slate-400">Click any entry to expand. Anchored to the scaling-book reference.</p>
            </div>
          </div>
          <button type="button" onClick={onClose} aria-label="Close glossary" className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-white/70">
            <X style={{ width: 18, height: 18 }} />
          </button>
        </div>

        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" style={{ width: 16, height: 16 }} />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search concepts, formulas, sources…"
            className="glass-input w-full pl-9 py-2.5 text-sm text-slate-700 placeholder:text-slate-400"
          />
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar space-y-2 pr-1">
          {results.length === 0 && (
            <p className="text-sm text-slate-400 py-6 text-center">No concepts match “{query}”.</p>
          )}
          {results.map((c) => {
            const openCard = expandedId === c.id;
            return (
              <div key={c.id} className="glass rounded-xl overflow-hidden">
                <button
                  type="button"
                  onClick={() => setExpandedId(openCard ? null : c.id)}
                  className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-white/60 transition-colors"
                >
                  <span className="text-sm font-semibold text-slate-700 capitalize">{c.title}</span>
                  <span className="flex items-center gap-1.5">
                    <span className="hidden sm:block text-[10px] text-slate-400 max-w-[220px] truncate">{c.sourceRef.split('—')[0]}</span>
                    <ChevronDown
                      className={cn('text-slate-400 transition-transform', openCard && 'rotate-180')}
                      style={{ width: 16, height: 16 }}
                    />
                  </span>
                </button>
                <AnimatePresence initial={false}>
                  {openCard && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                    >
                      <div className="px-4 pb-4 space-y-2">
                        <p className="text-[12.5px] text-slate-600">{c.summary}</p>
                        {c.body.map((b, i) => (
                          <p key={i} className="text-[11.5px] leading-relaxed text-slate-500">{b}</p>
                        ))}
                        {c.formula && (
                          <div className="rounded-lg bg-white/70 border border-slate-200/60 px-3 py-2">
                            <code className="text-[11px] font-mono text-slate-700">{c.formula}</code>
                          </div>
                        )}
                        <div className="text-[10px] text-slate-400 border-t border-slate-200/70 pt-2">
                          Source: <span className="font-mono text-slate-500">{c.sourceRef}</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
}
