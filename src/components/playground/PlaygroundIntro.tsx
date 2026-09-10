import GlassCard from '../ui/GlassCard';
import ConceptTag from '../ui/ConceptTag';

/**
 * Orientation block shown above the Lab/Train sections. Answers, in order:
 * what this page is, what it deliberately is not (heading off the "why
 * doesn't Configure do anything here" confusion before it happens), and a
 * suggested — not enforced — order of operations for a first-time visitor.
 */
export default function PlaygroundIntro() {
  return (
    <GlassCard className="p-5 sm:p-6 space-y-4">
      <p className="text-sm text-slate-600 leading-relaxed">
        This is a real, working transformer (<strong>microGPT</strong>) trained and run entirely in your
        browser. It's the mechanical companion to the curriculum's{' '}
        <a href="#one-token" className="text-accent underline underline-offset-2">One token, end to end</a>,{' '}
        <a href="#attention" className="text-accent underline underline-offset-2">Attention</a>, and{' '}
        <a href="#kv-cache" className="text-accent underline underline-offset-2">KV cache</a> sections — same
        math, same cache, small enough to watch every number.
      </p>
      <p className="text-sm text-slate-600 leading-relaxed">
        <strong>It's not about cost.</strong> This page is about <em>how</em> generation works, not{' '}
        <em>how much it costs</em> — for FLOPs, bytes, and dollars, that's the rest of the site. The
        "Configure" hardware/economics panel has no effect here, so it's hidden on this page.
      </p>
      <div>
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Suggested order</p>
        <ol className="list-decimal pl-5 space-y-1.5 text-sm text-slate-600">
          <li>Play in the <strong>Lab</strong> first with the untrained (random, seed 42) weights — watch prefill build the cache and decode consume it.</li>
          <li><strong>Train</strong> a model on a small corpus and watch the loss fall.</li>
          <li>Come back to the <strong>Lab</strong> — same prompt, same buttons — and see the trained model's attention and probabilities look different.</li>
        </ol>
      </div>
      <div className="flex flex-wrap gap-1.5 pt-1">
        <ConceptTag id="kv-cache" />
        <ConceptTag id="attention-intensity" />
        <ConceptTag id="prefill" />
        <ConceptTag id="generation" />
      </div>
    </GlassCard>
  );
}
