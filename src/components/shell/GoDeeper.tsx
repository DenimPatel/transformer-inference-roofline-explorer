import { useState, type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';

interface GoDeeperProps {
  /** Short label for what is behind the disclosure. */
  label?: string;
  /** One line telling the reader whether they need this. */
  hint?: string;
  children: ReactNode;
}

/**
 * Progressive disclosure. Every section states its claim in plain language and
 * shows a figure; the derivation lives in here, closed by default, attached to
 * the concept it belongs to rather than in a separate "deep dive" destination.
 */
export default function GoDeeper({ label = 'The math', hint, children }: GoDeeperProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="my-6 border-t border-slate-200">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full flex items-baseline gap-2 py-3 text-left group"
      >
        <ChevronRight
          aria-hidden
          className={`shrink-0 self-center transition-transform text-slate-400 ${open ? 'rotate-90' : ''}`}
          style={{ width: 14, height: 14 }}
        />
        <span className="text-sm font-semibold text-slate-700 group-hover:text-[var(--color-accent)]">
          {label}
        </span>
        {hint && <span className="text-xs text-slate-400">{hint}</span>}
      </button>
      {open && <div className="pb-4 space-y-4">{children}</div>}
    </div>
  );
}
