import { Fragment, type ReactNode } from 'react';
import { motion } from 'motion/react';
import { BookOpen } from 'lucide-react';
import ConceptTag from '../ui/ConceptTag';

interface SectionProps {
  part?: string;
  number?: string;
  title: string;
  /** One-sentence plain-English claim, read before any math. */
  standfirst?: string;
  sourceRef?: string;
  /** Concept ids surfaced as cross-links to their canonical home. */
  concepts?: string[];
  children: ReactNode;
}

/**
 * The one container every curriculum section uses. Replaces LessonShell — same
 * visual treatment, plus a standfirst and concept cross-links. Deliberately
 * does not import the curriculum: prev/next navigation is rendered by the app
 * shell around this, which keeps the module graph acyclic.
 */
export default function Section({
  part, number, title, standfirst, sourceRef, concepts, children,
}: SectionProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="space-y-4 pb-10"
    >
      <header className="space-y-2">
        {(part || number) && (
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">
            {[part, number].filter(Boolean).join(' · ')}
          </p>
        )}
        <h2 className="text-3xl font-bold text-slate-800 tracking-tight">{title}</h2>
        {standfirst && (
          <p className="text-lg text-slate-600 leading-relaxed max-w-[62ch]">{standfirst}</p>
        )}
        {sourceRef && (
          <p className="text-[11px] text-slate-400 flex items-center gap-1.5">
            <BookOpen style={{ width: 12, height: 12 }} /> {sourceRef}
          </p>
        )}
        {concepts && concepts.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {concepts.map((id) => (
              <Fragment key={id}><ConceptTag id={id} /></Fragment>
            ))}
          </div>
        )}
      </header>

      {children}
    </motion.article>
  );
}
