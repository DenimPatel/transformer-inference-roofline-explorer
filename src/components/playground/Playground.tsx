import { FlaskConical } from 'lucide-react';
import { MicroGptPlayground } from '../TokenGeneration';

/**
 * Off-spine. A real port of reference/microgpt.py that trains and generates in
 * the browser — useful, but a different kind of thing from the roofline
 * curriculum, so it gets its own route rather than a place in the reading order.
 */
export default function Playground() {
  return (
    <article className="space-y-4">
      <header className="space-y-2">
        <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400 flex items-center gap-1.5">
          <FlaskConical style={{ width: 12, height: 12 }} /> The bench · off the main path
        </p>
        <h2 className="text-3xl font-bold text-slate-800 tracking-tight">microGPT playground</h2>
        <p className="text-lg text-slate-600 leading-relaxed max-w-[62ch]">
          A working transformer, trained from scratch in your browser. Nothing here is required
          for the curriculum — it is where you go to watch the thing the curriculum describes
          actually run.
        </p>
      </header>
      <MicroGptPlayground />
    </article>
  );
}
