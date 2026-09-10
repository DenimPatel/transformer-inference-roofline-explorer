import {
  BookOpen, Type, Layers, Eye, MemoryStick, Gauge, Dices, Hash, Cpu,
} from 'lucide-react';
import ConceptTag from './ui/ConceptTag';
import KvUsageExplain from './ui/KvUsageExplain';
import { DEFAULT_CONFIG } from '../lib/microgpt';
import { CHART as C } from '../lib/theme';
import {
  modelStatic, TokenizerDemo, EmbeddingDemo, ForwardDiagram, PrefillVisual,
  CausalAttentionGrid, DecodeLoop, SamplingExplain,
} from './playground/explainers';

const SECTIONS = [
  { id: 'tokenizer', label: 'Tokenizer', icon: Hash },
  { id: 'embedding', label: 'Embedding', icon: Type },
  { id: 'forward', label: 'Forward Pass', icon: Cpu },
  { id: 'prefill', label: 'Prefill', icon: Layers },
  { id: 'kv-cache', label: 'KV Cache', icon: MemoryStick },
  { id: 'attention', label: 'Attention', icon: Eye },
  { id: 'decode', label: 'Generation', icon: Gauge },
  { id: 'sampling', label: 'Sampling', icon: Dices },
];

/**
 * The Part II opener: the full "characters in, one token out" pipeline,
 * explained end to end. The hands-on lab that runs and trains this same
 * model lives at its own route (`/playground`) — see
 * `src/components/playground/Playground.tsx`.
 */
export default function TokenGenerationTab() {
  const cfg = DEFAULT_CONFIG;

  return (
    <div className="pb-16 max-w-6xl mx-auto mt-6 px-4">
      {/* ---- Hero ---- */}
      <section className="text-center mb-10">
        <div className="inline-flex items-center gap-2 glass-chip px-3 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4">
          <BookOpen className="w-3.5 h-3.5 text-accent" /> The full generation pipeline
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
          Token Generation, <span className="text-accent">Step by Step</span>
        </h1>
        <p className="text-slate-500 max-w-3xl mx-auto leading-relaxed">
          From raw characters to sampled output. <strong>Prefill</strong> ingests the whole prompt
          and builds the <strong>KV cache</strong>; <strong>decode</strong> then produces tokens one at a time,
          reusing that cache. Everything mirrors{' '}
          <code className="text-xs bg-slate-100 px-1 py-0.5 rounded">reference/microgpt.py</code> — and the
          playground lab runs the <em>real</em> algorithm in your browser.
        </p>
        <div className="flex flex-wrap justify-center gap-2 mt-6">
          <ConceptTag id="prefill" />
          <ConceptTag id="generation" />
          <ConceptTag id="kv-cache" />
          <ConceptTag id="attention-intensity" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-8 max-w-3xl mx-auto">
          <HeroKpi icon={Layers} label="Layers" value={`${cfg.nLayer}`} sub="depth" />
          <HeroKpi icon={Type} label="Embedding" value={`${cfg.nEmbd}`} sub="hidden dim" />
          <HeroKpi icon={Gauge} label="Max ctx" value={`${cfg.blockSize}`} sub="tokens" />
          <HeroKpi icon={Eye} label="Heads" value={`${cfg.nHead}`} sub={`d_head=${cfg.nEmbd / cfg.nHead}`} />
          <HeroKpi icon={Hash} label="Vocab" value={`${modelStatic.vocab.length + 1}`} sub="chars + BOS" />
        </div>
      </section>

      {/* ---- Section nav ---- */}
      <nav className="sticky top-0 z-30 -mx-2 px-2 py-3 mb-8 bg-[#f3f2f2] border-b border-slate-200 rounded-2xl">
        <div className="flex gap-1.5 overflow-x-auto custom-scrollbar py-1">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#${s.id}`}
              className="shrink-0 inline-flex items-center gap-1.5 glass-chip px-3 py-1.5 text-[11px] font-semibold text-slate-600 hover:text-accent hover:border-accent/40 transition-colors">
              <s.icon className="w-3.5 h-3.5" /> {s.label}
            </a>
          ))}
        </div>
      </nav>

      {/* 01 Tokenizer */}
      <SectionCard id="tokenizer" icon={Hash} color={C.accent} number="01"
        title="Tokenizer &amp; Vocabulary">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            The model cannot see characters or words — it only understands <strong>token ids</strong> (integers).
            microgpt uses a character-level tokenizer: it collects every unique character in the corpus and
            assigns each an integer id <code>0..n-1</code>. A special <strong>BOS</strong> (begin-of-sequence)
            token gets the id <code>n</code>.
          </p>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-1">Tokenize the prompt</h3>
          <p className="text-sm text-slate-500 mb-4">Each character maps to one integer; BOS marks the start.</p>
          <TokenizerDemo />
        </div>
      </SectionCard>

      {/* 02 Embedding */}
      <SectionCard id="embedding" icon={Type} color={C.sky} number="02"
        title="Embedding &amp; Position">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Tokens are looked up in two learned tables: <code>wte</code> (token embedding, shape
            <code> vocab &times; n_embd</code>) and <code>wpe</code> (position embedding, shape
            <code> block_size &times; n_embd</code>). In <code>microgpt.py</code>:
          </p>
          <pre className="text-xs bg-slate-900 text-slate-100 rounded-lg p-4 overflow-x-auto font-mono leading-relaxed">
{`tok_emb = state_dict['wte'][token_id]   # token id -> vector
pos_emb = state_dict['wpe'][pos_id]     # position id -> vector
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # add them
x = rmsnorm(x)`}
          </pre>
          <p>
            The <strong>token vector</strong> encodes <em>which</em> token; the <strong>position vector</strong>
            encodes <em>where</em> it sits, since attention alone has no notion of order. Summing them and
            applying RMSNorm produces the hidden vector <code>x</code> that flows through the layers.
          </p>
        </div>
        <EmbeddingDemo />
      </SectionCard>

      {/* 03 Forward pass */}
      <SectionCard id="forward" icon={Cpu} color={C.compute} number="03"
        title="The Forward Pass (One Token)">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Inside <code>gpt(token_id, pos_id, keys, values)</code> each token walks the same path: one
            <strong> attention block</strong> then one <strong>MLP block</strong>, both wrapped in residual
            connections. This is the heart of the whole pipeline.
          </p>
          <ol className="list-decimal pl-5 space-y-2 text-sm">
            <li><strong>Project to Q, K, V:</strong> <code>x</code> is linearly projected into a query
              <code>q</code>, key <code>k</code>, and value <code>v</code> vector.</li>
            <li><strong>Cache K &amp; V:</strong> <code>keys[layer].append(k)</code> and
              <code> values[layer].append(v)</code> — this single line <em>is</em> how the KV cache is created.</li>
            <li><strong>Attention per head:</strong> score <code>q &middot; k / &radic;head_dim</code> against
              every cached key, softmax → weights, weighted sum of cached values.</li>
            <li><strong>Output projection</strong> <code>attn_wo</code>, then add back the residual.</li>
            <li><strong>MLP:</strong> RMSNorm → <code>fc1</code> → ReLU → <code>fc2</code> → residual.</li>
          </ol>
          <p>
            Finally <code>lm_head</code> projects the hidden vector to <code>logits</code>, one score per
            vocabulary token — the raw, unnormalized opinion of "what should come next".
          </p>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-3">The data flow</h3>
          <ForwardDiagram />
        </div>
      </SectionCard>

      {/* 04 Prefill */}
      <SectionCard id="prefill" icon={Layers} color={C.compute} number="04"
        title="Prefill: Ingesting the Prompt">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            <strong>Prefill</strong> runs every prompt token through the model <em>in one pass</em>. Because
            attention is causal (a token only sees itself and earlier tokens), the prompt positions can be
            processed as one big batch — weights are reused across all <code>B &middot; T</code> tokens, so
            prefill has high arithmetic intensity and is <strong>almost always compute-bound</strong>.
          </p>
          <p>
            Crucially, this is the first time the KV cache is populated: as each prompt position computes its
            K and V, they are appended to the per-layer cache. When prefill finishes, the cache holds the K/V
            for the <em>entire prompt</em>, and the final forward produces logits that predict the
            <strong> first generated token</strong>. The time to reach that first token is
            <strong> TTFT</strong> (time-to-first-token).
          </p>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-1">Every prompt token contributes to the cache</h3>
          <p className="text-sm text-slate-500 mb-4">Prefill = parallel compute (compute-bound) + cache build (memory growth).</p>
          <PrefillVisual />
        </div>
      </SectionCard>

      {/* 05 KV cache */}
      <SectionCard id="kv-cache" icon={MemoryStick} color={C.violet} number="05"
        title="The KV Cache, Mechanically">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            A token only ever attends to earlier tokens, so instead of reprocessing the prefix on every step we
            <em> cache each token&rsquo;s key and value projection</em>. The cache is shaped
            <code> [layer][position][hidden_dim]</code> for K and another for V.
          </p>
          <p>
            Size per position: <code>2 &middot; bytes &middot; n_layers &middot; n_heads &middot; head_dim</code>.
            Because heads can share K/V (GQA), and because the cache grows linearly with context &times; batch,
            it — not the weights — is typically what dominates inference memory.
          </p>
        </div>
        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-3">Which token is recomputed vs cached?</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="glass rounded-lg p-4 text-sm text-slate-600">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-emerald-500" />
                <span className="font-bold text-slate-800">Prefix (prompt tokens)</span>
              </div>
              K &amp; V computed <em>once</em> during prefill and stored. Never recomputed again.
            </div>
            <div className="glass rounded-lg p-4 text-sm text-slate-600">
              <div className="flex items-center gap-2 mb-2">
                <Gauge className="w-4 h-4 text-amber-500" />
                <span className="font-bold text-slate-800">New token (decode)</span>
              </div>
              Only <em>Q</em> is freshly computed; its K &amp; V are appended, and it reads the stored K/V of
              everything before it.
            </div>
          </div>
        </div>
      </SectionCard>

      {/* 06 Attention */}
      <SectionCard id="attention" icon={Eye} color={C.sky} number="06"
        title="Causal Attention in Detail">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            For each head: <code>score = q &middot; k&#7731;&#7732; / &radic;head_dim</code> for every cached
            key, softmax to get weights, then a weighted sum of cached values. The bottom-triangle shape below
            is the <strong>causal mask</strong> — token <code>row</code> may attend only to columns
            <code> ≤ row</code>.
          </p>
          <p>
            The math is <code>I = ST/(S+T)</code>: <strong>prefill</strong> (S = T) has intensity
            <code> ≈ T/2</code> (compute-bound); <strong>decode</strong> (T = 1) pins intensity to
            <code> ≈ 1</code> — a constant far below the ridge, hence always memory-bound.
          </p>
        </div>
        <CausalAttentionGrid />
        <div className="mt-6">
          <h3 className="font-bold text-slate-800 mb-1">How each K and V is actually used</h3>
          <p className="text-sm text-slate-500 mb-4">
            The cached keys pick <em>where</em> to attend; the cached values supply <em>what</em> gets carried
            forward. Both lines below mirror the live lab.
          </p>
          <KvUsageExplain />
        </div>
      </SectionCard>

      {/* 07 Decode / Generation */}
      <SectionCard id="decode" icon={Gauge} color={C.amber} number="07"
        title="Decode: Generating One Token at a Time">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Once the prompt is prefilled, generation runs <strong>one token at a time</strong> (T = 1). At each
            step only a single query vector is computed; the keys and values it attends to are read from the
            cache. Weights are streamed fresh every step and the cache only grows, so decode is
            <strong> almost always memory-bound</strong> — the roofline flip from prefill.
          </p>
          <ul className="list-disc pl-5 space-y-2 text-sm">
            <li>Feed the newest token through the model (its K/V get appended).</li>
            <li>Read <code>lm_head</code> logits, sample the next token.</li>
            <li>Append it, repeat — until a BOS/stop token or the max context length.</li>
          </ul>
        </div>
        <DecodeLoop />
      </SectionCard>

      {/* 08 Sampling */}
      <SectionCard id="sampling" icon={Dices} color={C.violet} number="08"
        title="Sampling &amp; Temperature">
        <SamplingExplain />
      </SectionCard>
      <p className="text-center text-xs text-slate-400 mt-12">
        Implemented from <code className="text-xs bg-slate-100 px-1 rounded">reference/microgpt.py</code> — the
        "atomic" GPT. Everything above is computed live; nothing is a static image. The lab that runs and
        trains this model lives in the{' '}
        <a href="#playground" className="text-accent underline underline-offset-2">playground</a>.
      </p>
    </div>
  );
}

/* ---------------- shared layout pieces ---------------- */

function SectionCard({ id, icon: IconCmp, color, number, title, children }: any) {
  return (
    <section id={id} className="glass-card p-6 sm:p-8 mb-8 scroll-mt-24">
      <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
        <span className="inline-flex items-center justify-center w-9 h-9 rounded-xl mr-3 text-white shrink-0"
          style={{ background: color }}>
          <IconCmp className="w-5 h-5" />
        </span>
        <span className="text-3xl font-black mr-3 opacity-20" style={{ color }}>{number}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function HeroKpi({ icon: I, label, value, sub }: any) {
  return (
    <div className="glass rounded-xl p-3 text-left">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 mb-1">
        <I className="w-3.5 h-3.5" /> {label}
      </div>
      <div className="font-mono text-lg font-bold text-slate-900 leading-none">{value}</div>
      <div className="text-[11px] text-slate-400 mt-1 truncate">{sub}</div>
    </div>
  );
}
