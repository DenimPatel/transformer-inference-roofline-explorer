import { DeepProblems } from '../DeepDive';

/**
 * Off-spine reference. Worked problems from the scaling-book material, kept
 * out of the reading order so the curriculum stays a narrative.
 */
export default function Problems() {
  return (
    <article className="space-y-4">
      <header className="space-y-2">
        <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">The bench · off the main path</p>
        <h2 className="text-3xl font-bold text-slate-800 tracking-tight">Problems</h2>
        <p className="text-lg text-slate-600 leading-relaxed max-w-[62ch]">
          Work these once you have read Part III. They are the fastest way to find out whether
          the roofline has actually landed.
        </p>
      </header>
      <DeepProblems />
    </article>
  );
}
