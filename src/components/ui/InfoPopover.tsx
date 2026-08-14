import { useRef, useState, useEffect, useCallback, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { HelpCircle, X, BookOpen, Calculator } from 'lucide-react';
import { getConcept, type Concept } from '../../lib/concepts';
import { cn } from '../../lib/utils';

interface InfoPopoverProps {
  conceptId?: string;
  title?: string;
  summary?: string;
  body?: string[];
  formula?: string;
  sourceRef?: string;
  className?: string;
  iconSize?: number;
  label?: string;
  children?: ReactNode;
}

/**
 * Click-to-learn "?" popover. Accepts a conceptId (looks up the glossary) or
 * inline content so it can be reused anywhere.
 */
export default function InfoPopover({
  conceptId,
  title,
  summary,
  body,
  formula,
  sourceRef,
  className,
  iconSize = 15,
  label = 'Learn more',
  children,
}: InfoPopoverProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const concept: Concept | undefined = conceptId ? getConcept(conceptId) : undefined;

  const toggle = useCallback(() => {
    if (!open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setPos({ top: r.bottom + 8, left: r.left });
    }
    setOpen((o) => !o);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(e.target as Node) &&
        btnRef.current &&
        !btnRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  // keep within viewport — pos is always set before the panel opens
  const style = pos
    ? {
        top: pos.top,
        left: Math.max(8, Math.min(pos.left, (window.innerWidth || 0) - 340 - 12)),
        maxHeight: Math.max(160, Math.min(420, (window.innerHeight || 0) - pos.top - 12)),
      }
    : undefined;

  const t = concept?.title ?? title;
  const s = concept?.summary ?? summary;
  const b = concept?.body ?? body;
  const f = concept?.formula ?? formula;
  const sr = concept?.sourceRef ?? sourceRef;

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        onClick={toggle}
        aria-expanded={open}
        aria-label={label}
        title={label}
        className={cn(
          'inline-flex items-center justify-center gap-1.5 rounded-full text-slate-400 transition-colors hover:text-[var(--color-accent)] hover:bg-white/70 cursor-help shrink-0',
          className
        )}
      >
        <HelpCircle style={{ width: iconSize, height: iconSize }} />
        {children}
      </button>

      {open &&
        createPortal(
          <div
            ref={panelRef}
            className="info-popover fixed z-[9999] w-[320px] overflow-y-auto rounded-2xl p-4 text-left custom-scrollbar"
          style={style}
          role="dialog"
          aria-label={t}
        >
          <div className="flex items-start justify-between gap-2 mb-2">
            <div className="flex items-center gap-2">
              <span className="text-[var(--color-accent)]">
                <BookOpen style={{ width: 16, height: 16 }} />
              </span>
              <h4 className="text-sm font-bold text-slate-800">{t}</h4>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="text-slate-400 hover:text-slate-600"
              aria-label="Close"
            >
              <X style={{ width: 15, height: 15 }} />
            </button>
          </div>

          <p className="text-[12.5px] leading-relaxed text-slate-600 mb-3">{s}</p>

          {b && b.length > 0 && (
            <div className="space-y-2 mb-3">
              {b.map((para, i) => (
                <p key={i} className="text-[11.5px] leading-relaxed text-slate-500">
                  {para}
                </p>
              ))}
            </div>
          )}

          {f && (
            <div className="flex items-start gap-2 mb-3 rounded-lg bg-white/70 border border-slate-200/60 px-2.5 py-2">
              <span className="text-[var(--color-amber)] mt-0.5">
                <Calculator style={{ width: 13, height: 13 }} />
              </span>
              <code className="text-[11px] font-mono text-slate-700 leading-snug">{f}</code>
            </div>
          )}

          {sr && (
            <div className="text-[10px] text-slate-400 leading-snug border-t border-slate-200/70 pt-2">
              Source: <span className="font-mono text-slate-500">{sr}</span>
            </div>
          )}
          </div>,
          document.body
        )}
    </>
  );
}
