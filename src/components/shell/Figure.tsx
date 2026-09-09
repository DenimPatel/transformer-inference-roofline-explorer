import type { ReactNode } from 'react';

interface FigureProps {
  /**
   * Plain-English statement of what the reader should take away. Required on
   * purpose: a reader who cannot yet parse a log-log plot still gets the point.
   */
  takeaway: string;
  caption?: string;
  children: ReactNode;
}

/** Wraps a chart with its "read this chart" takeaway. */
export default function Figure({ takeaway, caption, children }: FigureProps) {
  return (
    <figure className="my-6">
      {children}
      <figcaption className="mt-3 text-sm text-slate-600 leading-relaxed">
        <span className="font-semibold text-slate-700">Reading this chart. </span>
        {takeaway}
        {caption && <span className="block mt-1 text-xs text-slate-400">{caption}</span>}
      </figcaption>
    </figure>
  );
}
