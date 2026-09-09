import type { ReactNode } from 'react';

interface SectionBodyProps {
  /** Citation into /reference/scaling-book/. */
  sourceRef?: string;
  /** Accepted and ignored: the section header is rendered by the app shell
   *  from the curriculum, so these no longer need to be repeated per lesson. */
  number?: number;
  title?: string;
  subtitle?: string;
  children: ReactNode;
}

/**
 * The body of a curriculum section. Was LessonShell, which drew its own title
 * block; now that <Section> renders one header from the curriculum metadata,
 * this keeps only the source citation so the two do not compete.
 */
export default function SectionBody({ sourceRef, children }: SectionBodyProps) {
  return (
    <div className="space-y-4 pb-6">
      {sourceRef && (
        <p className="text-[11px] text-slate-400 border-l-2 border-slate-200 pl-2">{sourceRef}</p>
      )}
      {children}
    </div>
  );
}
