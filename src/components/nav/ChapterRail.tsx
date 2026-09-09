import { Fragment } from 'react';
import { Check, BookOpen, FlaskConical, PenLine } from 'lucide-react';
import { CURRICULUM, FLAT_SECTIONS, TOTAL_MINUTES } from '../../lib/curriculum';

interface ChapterRailProps {
  current: string;
  completed: string[];
  onNavigate: (id: string) => void;
  onOpenGlossary: () => void;
}

/**
 * The spine, made visible. Replaces the seven-tab bar: one ordered list of
 * parts and sections, so there is exactly one reading order and the reader can
 * always see where they are in it.
 */
export default function ChapterRail({
  current, completed, onNavigate, onOpenGlossary,
}: ChapterRailProps) {
  const done = FLAT_SECTIONS.filter((e) => completed.includes(e.section.id)).length;
  const pct = Math.round((done / FLAT_SECTIONS.length) * 100);

  return (
    <nav aria-label="Contents" className="text-sm">
      <div className="pb-4 mb-4 border-b border-slate-200">
        <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Contents</p>
        <p className="mt-1 text-xs text-slate-500">
          {FLAT_SECTIONS.length} sections · about {TOTAL_MINUTES} min
        </p>
        <div className="mt-2 h-1 bg-slate-200 overflow-hidden">
          <div
            className="h-full transition-[width] duration-500"
            style={{ width: `${pct}%`, background: 'var(--color-accent)' }}
          />
        </div>
        <p className="mt-1 text-[11px] text-slate-400">{done} of {FLAT_SECTIONS.length} read</p>
      </div>

      <ol className="space-y-5">
        {CURRICULUM.map((part) => (
          <li key={part.id}>
            <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">{part.label}</p>
            <p className="text-sm font-semibold text-slate-700 leading-snug">{part.title}</p>
            <ol className="mt-1.5 space-y-0.5">
              {part.sections.map((s) => {
                const active = s.id === current;
                const isDone = completed.includes(s.id);
                return (
                  <li key={s.id}>
                    <button
                      type="button"
                      onClick={() => onNavigate(s.id)}
                      aria-current={active ? 'page' : undefined}
                      className={`w-full text-left flex items-baseline gap-1.5 py-1 pl-2 border-l-2 transition-colors cursor-pointer bg-transparent ${
                        active
                          ? 'border-[var(--color-accent)] text-slate-900 font-semibold'
                          : 'border-slate-200 text-slate-500 hover:text-slate-800 hover:border-slate-400'
                      }`}
                    >
                      <span className="shrink-0 w-3.5 self-center">
                        {isDone && !active && (
                          <Check className="text-emerald-500" style={{ width: 13, height: 13 }} />
                        )}
                      </span>
                      <span className="leading-snug">{s.title}</span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </li>
        ))}
      </ol>

      <div className="mt-6 pt-4 border-t border-slate-200 space-y-1">
        <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400 pb-1">The bench</p>
        {[
          { label: 'Glossary', icon: BookOpen, action: onOpenGlossary },
          { label: 'Problems', icon: PenLine, action: () => onNavigate('problems') },
          { label: 'microGPT playground', icon: FlaskConical, action: () => onNavigate('playground') },
        ].map(({ label, icon: Icon, action }) => (
          <Fragment key={label}>
            <button
              type="button"
              onClick={action}
              className="w-full text-left flex items-center gap-1.5 py-1 pl-2 text-slate-500 hover:text-slate-800 cursor-pointer bg-transparent"
            >
              <Icon style={{ width: 13, height: 13 }} />
              <span>{label}</span>
            </button>
          </Fragment>
        ))}
      </div>
    </nav>
  );
}
