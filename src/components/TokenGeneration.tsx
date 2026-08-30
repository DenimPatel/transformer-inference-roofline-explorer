import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  Line, LineChart,
} from 'recharts';
import {
  BookOpen, Type, Layers, Zap, MemoryStick, Eye, Gauge, Dices, FlaskConical,
  ArrowRight, RotateCcw, Hash, Play, Sparkles, Cpu, GraduationCap, Loader2,
} from 'lucide-react';
import { cn } from '../lib/utils';
import ConceptTag from './ui/ConceptTag';
import KvUsageExplain from './ui/KvUsageExplain';
import { MicroGPT, DEFAULT_CONFIG, softmax, type KV, type ForwardResult } from '../lib/microgpt';
import { SAMPLE_CORPUS, rawNames, sampleNames } from '../lib/names_data';

const C = {
  compute: '#22c48b',
  memory: '#f25f7d',
  ridge: '#f43f5e',
  accent: '#5b7cfa',
  accentSoft: '#8aa0ff',
  sky: '#0ea5e9',
  amber: '#f59e0b',
  violet: '#8b5cf6',
  slate: '#94a3b8',
  ink: '#0b1220',
};

function fmtNum(v: number, digits = 0) {
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: digits });
}

const SECTIONS = [
  { id: 'tokenizer', label: 'Tokenizer', icon: Hash },
  { id: 'embedding', label: 'Embedding', icon: Type },
  { id: 'forward', label: 'Forward Pass', icon: Cpu },
  { id: 'prefill', label: 'Prefill', icon: Layers },
  { id: 'kv-cache', label: 'KV Cache', icon: MemoryStick },
  { id: 'attention', label: 'Attention', icon: Eye },
  { id: 'decode', label: 'Generation', icon: Gauge },
  { id: 'sampling', label: 'Sampling', icon: Dices },
  { id: 'lab', label: 'MicroGPT Lab', icon: FlaskConical },
  { id: 'train', label: 'Train MicroGPT', icon: GraduationCap },
];

const modelStatic = new MicroGPT();

export default function TokenGenerationTab() {
  const cfg = DEFAULT_CONFIG;
  const modelRef = useRef<MicroGPT | null>(null);
  if (!modelRef.current) modelRef.current = new MicroGPT();
  const sharedModel = modelRef.current;

  const [trained, setTrained] = useState(false);
  const [sessionKey, setSessionKey] = useState(0);
  const bumpSession = () => setSessionKey((k) => k + 1);
  const markTrained = () => { setTrained(true); bumpSession(); };
  const markUntrained = () => { setTrained(false); bumpSession(); };

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
          lab at the bottom runs the <em>real</em> algorithm in your browser.
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
      <nav className="sticky top-0 z-30 -mx-2 px-2 py-3 mb-8 blur-[1px] backdrop-blur-md bg-[#eef1fb]/70 rounded-2xl">
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
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Logits are not probabilities. We turn them into a distribution with
            <code> softmax(logits / temperature)</code>, then either take the argmax (greedy) or draw a random
            token weighted by the distribution — exactly microgpt&rsquo;s
            <code> random.choices(range(vocab), weights=probs)</code>.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-3">Greedy (T &rarr; 0)</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              Always pick the most likely token. Deterministic, safe, but repetitive — the model&rsquo;s top
              pick is often a generic continuation.
            </p>
          </div>
          <div className="glass rounded-xl p-5">
            <h3 className="font-bold text-slate-800 mb-3">Sampling with T &gt; 0</h3>
            <p className="text-sm text-slate-500 leading-relaxed">
              Divide logits by T to sharpen or flatten the distribution before sampling. Higher T = more
              randomness and "creativity"; lower T = closer to greedy.
            </p>
          </div>
        </div>
      </SectionCard>

      {/* 09 Lab */}
      <SectionCard id="lab" icon={FlaskConical} color={C.accent} number="09"
        title="The Interactive MicroGPT Lab">
        <p className="text-sm text-slate-500 mb-5 max-w-3xl">
          A real, working port of <code>microgpt.py</code> running here in your browser (deterministic weights,
          seed 42). Type a prompt, <strong>Prefill</strong> to ingest it and build the KV cache, then click
          <strong> Generate</strong> to emit one token at a time while watching the cache grow and attention spread.
        </p>
        <MicroGptLab model={sharedModel} trained={trained} sessionKey={sessionKey} />
      </SectionCard>

      {/* 10 Train MicroGPT */}
      <SectionCard id="train" icon={GraduationCap} color={C.violet} number="10"
        title="Train It, Then Run Inference">
        <p className="text-sm text-slate-500 mb-5 max-w-3xl">
          <code>reference/microgpt.py</code> doesn&rsquo;t just run inference — it <strong>trains</strong> the model
          from scratch with Adam (autograd + 1000 optimizer steps). The lab below does the same in pure JavaScript:
          choose a corpus, click <strong>Train</strong>, and watch the loss fall. When it finishes, the Prefill/Generate
          lab above shares the <em>same trained weights</em> — so the tokens it emits actually look like the corpus.
        </p>
        <TrainPanel model={sharedModel} trained={trained} onTrained={markTrained} onUntrained={markUntrained} />
      </SectionCard>

      <p className="text-center text-xs text-slate-400 mt-12">
        Implemented from <code className="text-xs bg-slate-100 px-1 rounded">reference/microgpt.py</code> — the
        "atomic" GPT. Everything above is computed live; nothing is a static image.
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

/* ---------------- 01 Tokenizer demo ---------------- */

function TokenizerDemo() {
  const sample = "the cat sat";
  const ids = [modelStatic.BOS, ...modelStatic.tokenize(sample)];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1.5">
        {ids.map((id, i) => (
          <div key={i} className="glass rounded-lg px-2.5 py-1.5 flex flex-col items-center min-w-[42px]">
            <span className={cn('font-mono text-sm font-bold', id === modelStatic.BOS ? 'text-violet-600' : 'text-slate-800')}>
              {id === modelStatic.BOS ? '⟨BOS⟩' : modelStatic.labelForToken(id)}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">id {id}</span>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-500">
        <div className="glass rounded-lg p-3">
          <span className="text-slate-700 font-semibold">Vocabulary</span>{' '}
          <span className="font-mono">size {modelStatic.vocab.length + 1} (incl. BOS)</span>
        </div>
        <div className="glass rounded-lg p-3">
          <span className="text-slate-700 font-semibold">"the cat sat"</span>{' '}
          <span className="font-mono">-&gt; {ids.join(', ')}</span>
        </div>
      </div>
    </div>
  );
}

/* ---------------- 02 Embedding demo ---------------- */

function EmbeddingDemo() {
  const dims = DEFAULT_CONFIG.nEmbd;
  const tokenId = modelStatic.tokenize('t')[0] ?? 0;
  const pos = 0;
  const tok = modelStatic.weights.wte[tokenId];
  const posEmb = modelStatic.weights.wpe[pos];
  return (
    <div className="space-y-3">
      <VectorRow label={`wte["t"] (id ${tokenId})`} values={tok} color={C.accent} />
      <VectorRow label={`wpe[${pos}]`} values={posEmb} color={C.sky} />
      <VectorRow label="x = tok + pos" values={tok.map((v, i) => v + posEmb[i])} color={C.compute} />
      <p className="text-[11px] text-slate-400">
        Each column is one of the {dims} embedding dimensions (rounded to 2 decimals for display).
      </p>
    </div>
  );
}

function VectorRow({ label, values, color }: { label: string; values: number[]; color: string }) {
  return (
    <div>
      <div className="text-xs font-semibold text-slate-600 mb-1 font-mono">{label}</div>
      <div className="flex flex-wrap gap-1">
        {values.map((v, i) => (
          <div key={i} className="rounded-md px-1.5 py-1 text-[10px] font-mono"
            style={{ backgroundColor: `${color}14`, color: `${color}` }}>
            {v.toFixed(2)}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------------- 03 Forward diagram ---------------- */

function ForwardDiagram() {
  const steps = [
    { label: 'Embed + pos', color: C.accent, w: 'wte + wpe' },
    { label: 'RMSNorm', color: C.sky, w: 'scale' },
    { label: 'Q / K / V', color: C.violet, w: '3 × lin' },
    { label: 'Cache K,V', color: C.memory, w: 'append' },
    { label: 'Attention', color: C.compute, w: 'softmax' },
    { label: 'Out proj', color: C.accent, w: 'attn_wo' },
    { label: 'MLP', color: C.sky, w: 'fc1·ReLU·fc2' },
    { label: 'Logits', color: C.amber, w: 'lm_head' },
  ];
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {steps.map((s, i) => (
        <React.Fragment key={s.label}>
          <div className="rounded-lg px-3 py-2 text-center min-w-[92px]"
            style={{ backgroundColor: `${s.color}12`, border: `1px solid ${s.color}40` }}>
            <div className="text-xs font-bold" style={{ color: s.color }}>{s.label}</div>
            <div className="text-[10px] font-mono text-slate-500">{s.w}</div>
          </div>
          {i < steps.length - 1 && <ArrowRight className="w-4 h-4 text-slate-300 shrink-0" />}
        </React.Fragment>
      ))}
    </div>
  );
}

/* ---------------- 04 Prefill visual ---------------- */

function PrefillVisual() {
  const tokens = ['BOS', 't', 'h', 'e', ' ', 'c', 'a', 't'];
  return (
    <div>
      <div className="flex flex-wrap gap-1.5 mb-4">
        {tokens.map((t, i) => (
          <div key={i}
            className="rounded-lg px-2.5 py-1.5 text-xs font-bold text-white"
            style={{ background: i === 0 ? '#8b5cf6' : 'linear-gradient(135deg,#5b7cfa,#7f6bf0)' }}>
            {t}
          </div>
        ))}
        <div className="rounded-lg px-2.5 py-1.5 text-xs font-bold bg-emerald-500 text-white">next?</div>
      </div>
      <p className="text-sm text-slate-600 mb-2">
        All <strong>{tokens.length}</strong> prompt positions run at once (parallel, compute-bound) and each
        appends its K/V:
      </p>
      <div className="space-y-1 font-mono text-[11px] text-slate-500">
        <div>layer0.keys  &larr; k[BOS]  k[t]  k[h]  k[e]  k[ ]  k[c]  k[a]  k[t]</div>
        <div>layer0.values &larr; v[BOS]  v[t]  v[h]  v[e]  v[ ]  v[c]  v[a]  v[t]</div>
      </div>
    </div>
  );
}

/* ---------------- 06 Causal attention grid ---------------- */

function CausalAttentionGrid() {
  const N = 8;
  const cells: boolean[][] = [];
  for (let r = 0; r < N; r++) {
    const row: boolean[] = [];
    for (let c = 0; c < N; c++) row.push(c <= r);
    cells.push(row);
  }
  return (
    <div className="glass rounded-xl p-5">
      <div className="grid gap-1" style={{ gridTemplateColumns: `auto repeat(${N}, 24px)` }}>
        <div />
        {cells[0].map((_, c) => (<div key={c} className="text-center text-[10px] text-slate-400 font-mono">{c}</div>))}
        {cells.map((row, r) => (
          <React.Fragment key={r}>
            <div className="text-[10px] text-slate-400 font-mono pr-1 leading-none flex items-center">q{r}</div>
            {row.map((on, c) => (
              <div key={c}
                className={cn('h-6 w-6 rounded', on ? (c === r ? 'bg-[var(--color-accent)]' : 'bg-[var(--color-accent)]/30') : 'bg-slate-200/50')}
                title={on ? `q${r} attends k${c}` : 'masked'} />
            ))}
          </React.Fragment>
        ))}
      </div>
      <p className="text-[11px] text-slate-400 mt-3">
        Rows = query positions, columns = key positions. A cell is filled only when the query index ≥ the key
        index — the causal mask guarantees no token peeks into the future.
      </p>
    </div>
  );
}

/* ---------------- 07 Decode loop ---------------- */

function DecodeLoop() {
  const steps = [
    { label: 'Feed token', body: 'newest token → model; its K/V appended to cache', color: C.accent },
    { label: 'Attend to cache', body: 'Q attends all stored K/V (one row of the triangle)', color: C.compute },
    { label: 'Logits', body: 'lm_head → scores for every vocab token', color: C.amber },
    { label: 'Sample', body: 'softmax(logits/T) → pick next token', color: C.violet },
  ];
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      {steps.map((s, i) => (
        <div key={s.label} className="glass rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className={cn('flex items-center justify-center w-6 h-6 rounded-lg text-white text-xs font-bold', i === 0 ? '' : '')} style={{ background: s.color }}>
              {i + 1}
            </span>
            <span className="font-bold text-slate-800 text-sm">{s.label}</span>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">{s.body}</p>
        </div>
      ))}
    </div>
  );
}

/* ---------------- 09 Interactive MicroGPT Lab ---------------- */

function cloneKV(kv: KV): KV {
  return {
    keys: kv.keys.map((layer) => layer.map((vec) => [...vec])),
    values: kv.values.map((layer) => layer.map((vec) => [...vec])),
  };
}

function MicroGptLab({ model, trained, sessionKey }: {
  model: MicroGPT;
  trained: boolean;
  sessionKey: number;
}) {
  const [prompt, setPrompt] = useState('the cat sat');
  const [temperature, setTemperature] = useState(0.5);
  const [greedy, setGreedy] = useState(false);

  const [tokens, setTokens] = useState<number[]>([]);
  const [kv, setKv] = useState<KV>(() => model.newKV());
  const [logits, setLogits] = useState<number[]>([]);
  const [attention, setAttention] = useState<ForwardResult['attention']>([]);
  const [newestPos, setNewestPos] = useState(-1);
  const [ready, setReady] = useState(false);
  const [done, setDone] = useState(false);

  // Drop the session whenever the weights were (re)trained, since the old KV no longer applies.
  React.useEffect(() => {
    setTokens([]);
    setKv(model.newKV());
    setLogits([]);
    setAttention([]);
    setNewestPos(-1);
    setReady(false);
    setDone(false);
  }, [sessionKey, model]);

  const promptIds = useMemo(() => model.tokenize(prompt), [model, prompt]);
  const promptCount = 1 + promptIds.length; // BOS + prompt tokens
  const sampledId = tokens.length > 0 ? tokens[tokens.length - 1] : -1;

  const probs = useMemo(
    () => (logits.length ? model.probsFor(logits, temperature) : []),
    [model, logits, temperature],
  );

  const kvLen = kv.keys[0].length;

  const handlePrefill = () => {
    const fresh = model.newKV(); // fresh cache, but keep the current (possibly trained) weights
    const ids = [model.BOS, ...model.tokenize(prompt)];
    if (ids.length === 1) ids.push(model.tokenize(' ')[0] ?? 0);
    const res = model.prefill(ids, fresh);
    const first = model.sample(res.logits, temperature, greedy);
    setTokens([...ids, first]);
    setKv(fresh);
    setLogits(res.logits);
    setAttention(res.attention);
    setNewestPos(ids.length - 1);
    setReady(true);
    setDone(first === model.BOS || ids.length + 1 >= model.config.blockSize);
  };

  const handleGenerate = () => {
    if (!ready || done) return;
    if (tokens.length >= model.config.blockSize) { setDone(true); return; }
    const next = cloneKV(kv);
    const lastToken = tokens[tokens.length - 1];
    const pos = tokens.length - 1;
    const res = model.forward(lastToken, pos, next);
    const sample = model.sample(res.logits, temperature, greedy);
    const newTokens = [...tokens, sample];
    setTokens(newTokens);
    setKv(next);
    setLogits(res.logits);
    setAttention(res.attention);
    setNewestPos(tokens.length - 1);
    setDone(sample === model.BOS || newTokens.length >= model.config.blockSize);
  };

  const handleReset = () => {
    const fresh = model.newKV(); // reset the session only; training state is untouched
    setTokens([]);
    setKv(fresh);
    setLogits([]);
    setAttention([]);
    setNewestPos(-1);
    setReady(false);
    setDone(false);
  };

  const topProbs = useMemo(() => {
    return probs
      .map((p, i) => ({ token: model.labelForToken(i), prob: p, id: i }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 10);
  }, [probs, model]);

  return (
    <div className="space-y-5">
      {/* controls */}
      <div className="glass rounded-xl p-4 space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[220px]">
            <label className="text-xs font-semibold text-slate-500 mb-1 block uppercase tracking-wider">Prompt</label>
            <input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="type a prompt…"
              className="glass-input w-full text-sm py-2 px-3 text-slate-800 font-medium"
            />
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handlePrefill}
              disabled={ready && !done}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
              style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}
            >
              <span className="inline-flex items-center gap-1.5"><Play className="w-3.5 h-3.5" /> Prefill</span>
            </button>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!ready || done}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
              style={{ background: 'linear-gradient(135deg, #22c48b, #149263)' }}
            >
              <span className="inline-flex items-center gap-1.5">Generate <ArrowRight className="w-3.5 h-3.5" /></span>
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-slate-600 bg-white/70 hover:bg-white"
            >
              <span className="inline-flex items-center gap-1.5"><RotateCcw className="w-3.5 h-3.5" /> Reset</span>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-5">
          <div className="flex-1 min-w-[200px]">
            <div className="flex justify-between text-xs mb-1">
              <label className="font-semibold text-slate-600">Temperature</label>
              <span className="font-mono text-slate-900">{temperature.toFixed(2)}</span>
            </div>
            <input type="range" min={0.05} max={2} step={0.05} value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))} className="glass-slider w-full" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-600">Sampling</span>
            <div className="flex glass rounded-lg p-0.5">
              <button type="button" onClick={() => setGreedy(true)}
                className={cn('px-3 py-1 text-xs rounded-md font-semibold transition-colors', greedy ? 'bg-white shadow text-slate-900' : 'text-slate-500')}>
                Greedy
              </button>
              <button type="button" onClick={() => setGreedy(false)}
                className={cn('px-3 py-1 text-xs rounded-md font-semibold transition-colors', !greedy ? 'bg-white shadow text-slate-900' : 'text-slate-500')}>
                Sample
              </button>
            </div>
          </div>
          <div className="flex items-center gap-2 my-0">
            <span className="text-xs font-semibold text-slate-600">Weights</span>
            <span
              className={cn('inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-bold',
                trained ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500')}>
              {trained
                ? (<><Sparkles className="w-3 h-3 text-emerald-500" /> Trained on this corpus</>)
                : 'Random (seed 42)'}
            </span>
          </div>
        </div>
      </div>

      {/* token stream */}
      <div className="glass rounded-xl p-4">
        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Token stream</h4>
        {tokens.length === 0 ? (
          <p className="text-sm text-slate-400">Press <strong>Prefill</strong> to begin.</p>
        ) : (
          <div className="flex flex-wrap items-center gap-1.5">
            {tokens.map((id, i) => {
              const isBOS = i === 0;
              const isPrompt = i < promptCount;
              const isNew = i === newestPos;
              const isSampled = i === tokens.length - 1 && i >= promptCount;
              return (
                <div key={i}
                  className={cn(
                    'relative rounded-lg px-2.5 py-1.5 text-sm font-bold transition-all',
                    isBOS && 'bg-violet-100 text-violet-700',
                    !isBOS && isPrompt && 'bg-white text-slate-800 border border-slate-200',
                    !isBOS && !isPrompt && 'text-white',
                    !isBOS && !isPrompt && (isSampled ? '' : 'opacity-85'),
                    isNew && 'ring-2 ring-[var(--color-accent)] ring-offset-1',
                  )}
                  style={!isBOS && !isPrompt ? { background: 'linear-gradient(135deg,#5b7cfa,#7f6bf0)' } : undefined}
                >
                  {isBOS ? '⟨BOS⟩' : model.labelForToken(id)}
                  {isNew && <span className="absolute -top-2 -right-1 text-[9px] font-black text-[var(--color-accent)]">†</span>}
                </div>
              );
            })}
            {!done && ready && (
              <span className="text-xs text-slate-400 flex items-center gap-1 ml-1">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" /> predicting…
              </span>
            )}
          </div>
        )}
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-[11px] text-slate-400">
          <span><span className="inline-block w-2.5 h-2.5 rounded-sm bg-violet-200 mr-1 align-middle" />BOS</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-sm bg-white border border-slate-300 mr-1 align-middle" />prompt</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-sm mr-1 align-middle" style={{ background: '#9aa5ff' }} />generated</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-sm ring-1 ring-[var(--color-accent)] mr-1 align-middle" />newest KV†</span>
        </div>
        {done && <p className="text-xs text-emerald-600 font-semibold mt-2">Generation ended (stop token or max context reached).</p>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* KV cache grid */}
        <div className="glass rounded-xl p-4">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">KV cache (per layer · per position)</h4>
          <p className="text-[11px] text-slate-400 mb-3">
            K stored in <span className="text-[var(--color-accent)] font-semibold">blue</span>, V in{' '}
            <span className="text-emerald-500 font-semibold">green</span>. At each step the newest query is scored
            against every stored <strong>K</strong> (blue = the match) and the stored <strong>V</strong> values are
            blended by those weights (green = the carry). {kvLen === 0 ? 'Empty until you prefill.' : `Filled for ${kvLen} positions.`}
          </p>
          <KvSizeGauge
            positions={kvLen}
            promptPositions={promptCount}
            maxPositions={model.config.blockSize}
            nEmbd={model.config.nEmbd}
            nLayer={model.config.nLayer}
            nHead={model.config.nHead}
          />
          {kv.keys.map((layer, li) => (
            <div key={li} className="mb-2">
              <div className="text-[11px] font-mono text-slate-500 mb-1">layer {li} · {kv.keys[li].length} positions</div>
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {kv.keys[li].map((kvec, p) => (
                  <div key={p}
                    className={cn('rounded-md p-1 border', p === newestPos ? 'border-[var(--color-accent)] ring-1 ring-[var(--color-accent)]' : 'border-slate-200')}>
                    <div className="text-center text-[9px] font-mono text-slate-400 mb-0.5">
                      {p < tokens.length ? (p === 0 ? 'BOS' : model.labelForToken(tokens[p])) : p}
                    </div>
                    <VecStack k={kvec} v={kv.values[li][p]} colorK={C.accent} colorV={C.compute} />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* attention */}
        <div className="glass rounded-xl p-4">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Attention this step (per head)</h4>
          <p className="text-[11px] text-slate-400 mb-3">
            Softmax weight of the newest query over each cached position, per head.
          </p>
          {attention.length === 0 ? (
            <p className="text-sm text-slate-400">No forward pass yet — run <strong>Prefill</strong>.</p>
          ) : (
            <div className="space-y-3">
              {attention.map((a, idx) => (
                <div key={idx}>
                  <div className="text-[11px] font-mono text-slate-500 mb-0.5">
                    layer {a.layer} · head {a.head}
                  </div>
                  <div className="flex h-4 rounded overflow-hidden">
                    {a.weights.map((w, p) => (
                      <div key={p} title={`pos ${p}: ${(w * 100).toFixed(1)}%`}
                        className="h-full"
                        style={{
                          width: `${w * 100}%`,
                          background: p === a.weights.length - 1
                            ? C.violet
                            : p === 0 ? '#c7b3ff' : `hsl(${220 - p * 8}, 70%, ${55 + p * 2}%)`,
                        }} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* next-token probabilities */}
      <div className="glass rounded-xl p-4">
        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
          Next-token probabilities <span className="normal-case font-normal">softmax(logits / T)</span>
        </h4>
        <p className="text-[11px] text-slate-400 mb-2">
          Top 10 vocabulary tokens. The highlighted bar is what was actually sampled
          {sampledId >= 0 ? ` (${model.labelForToken(sampledId)})` : ''}.
        </p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topProbs} layout="vertical" margin={{ top: 4, right: 16, left: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} fontSize={10} />
              <YAxis type="category" dataKey="token" width={54} tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(v: any) => [`${(Number(v) * 100).toFixed(1)}%`, 'prob']}
                labelFormatter={(l: any, p: any) => (p && p[0] ? `token "${p[0].payload.token}"` : l)}
              />
              <Bar dataKey="prob" radius={[0, 4, 4, 0]} barSize={16}>
                {topProbs.map((d, i) => (
                  <Cell key={i} fill={d.id === sampledId ? C.amber : C.accent} fillOpacity={d.id === sampledId ? 1 : 0.55} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ---------------- 10 Train MicroGPT Panel ---------------- */

function TrainPanel({ model, trained, onTrained, onUntrained }: {
  model: MicroGPT;
  trained: boolean;
  onTrained: () => void;
  onUntrained: () => void;
}) {
  const [corpus, setCorpus] = useState(SAMPLE_CORPUS.join('\n'));
  const [steps, setSteps] = useState(1000);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [error, setError] = useState('');

  const docs = useMemo(() => corpus.split('\n').map((s) => s.trim()).filter((s) => s.length > 0), [corpus]);

  const handleLoadSample = (which: 'inline' | 'names') => {
    setSamples([]);
    setLossHistory([]);
    setTraining(false);
    if (which === 'names') setCorpus(sampleNames(60, rawNames).join('\n'));
    else setCorpus(SAMPLE_CORPUS.join('\n'));
  };

  const handleTrain = async () => {
    if (docs.length === 0) { setError('Add at least one document (a word per line) to train on.'); return; }
    setError('');
    setTraining(true);
    setProgress(0);
    setLossHistory([]);
    setSamples([]);
    await model.train(docs, steps, {
      onStep: (step, loss) => {
        setProgress(step);
        if (step % 10 === 0 || step === steps) {
          setLossHistory((h) => { const next = [...h, { step, loss }]; return next.length > 400 ? next.slice(-400) : next; });
        }
      },
    });
    setTraining(false);
    setProgress(steps);
    onTrained();
  };

  const handleRandomize = () => {
    model.reset();
    setLossHistory([]);
    setSamples([]);
    setProgress(0);
    setTraining(false);
    onUntrained();
  };

  const handleSampleNames = () => {
    setSamples(model.generate(12, 0.6));
  };

  const lossData = lossHistory.length >= 2
    ? lossHistory
    : [{ step: 0, loss: lossHistory.length ? lossHistory[0].loss : 0 }, ...lossHistory];

  return (
    <div className="space-y-5">
      {/* corpus + controls */}
      <div className="glass rounded-xl p-4 space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[240px]">
            <label className="text-xs font-semibold text-slate-500 mb-1 block uppercase tracking-wider">
              Training corpus <span className="normal-case font-normal">(one document per line)</span>
            </label>
            <textarea
              value={corpus}
              onChange={(e) => setCorpus(e.target.value)}
              rows={6}
              spellCheck={false}
              className="glass-input w-full text-sm py-2 px-3 font-mono text-slate-800 resize-y"
            />
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-600">Steps</span>
              <input type="number" min={1} max={10000} value={steps}
                onChange={(e) => setSteps(Math.max(1, Math.min(10000, Number(e.target.value) || 1)))}
                className="glass-input w-24 py-1.5 px-2 text-sm font-mono" />
            </div>
            <button type="button" onClick={() => handleTrain()}
              disabled={training || docs.length === 0}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
              style={{ background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)' }}>
              <span className="inline-flex items-center gap-1.5">
                {training ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <GraduationCap className="w-3.5 h-3.5" />}
                {training ? `Training… ${progress}/${steps}` : 'Train'}
              </span>
            </button>
            <button type="button" onClick={() => handleSampleNames()}
              disabled={training || !trained}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-white bg-white/70 hover:bg-white">
              <span className="inline-flex items-center gap-1.5"><Sparkles className="w-3.5 h-3.5" /> Sample names</span>
            </button>
            <button type="button" onClick={() => handleRandomize()}
              disabled={training}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-slate-600 bg-white/70 hover:bg-white">
              <span className="inline-flex items-center gap-1.5"><RotateCcw className="w-3.5 h-3.5" /> Random weights</span>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-semibold text-slate-600">Dataset:</span>
          <button type="button" onClick={() => handleLoadSample('inline')} disabled={training}
            className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/70 hover:bg-white">30 inline names</button>
          <button type="button" onClick={() => handleLoadSample('names')} disabled={training}
            className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/70 hover:bg-white">
            Load names.txt subset
          </button>
          <span className="text-[11px] text-slate-400">{docs.length} documents ready</span>
        </div>
        {error && <p className="text-xs text-rose-500 font-semibold">{error}</p>}
      </div>

      {/* loss curve + samples */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="glass rounded-xl p-4">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
            Training loss <span className="normal-case font-normal">(cross-entropy / token)</span>
          </h4>
          <p className="text-[11px] text-slate-400 mb-3">
            {trained ? 'Adam ran for the full run — weights now produce corpus-like tokens.' : 'Run Train to see loss fall toward 0 (perfect next-char prediction).'}
          </p>
          {lossHistory.length < 2 ? (
            <div className="h-40 flex items-center justify-center text-sm text-slate-400">
              {training ? <span className="flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> optimizing…</span> : 'No training run yet.'}
            </div>
          ) : (
            <div className="h-40">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData} margin={{ top: 8, right: 10, left: -4, bottom: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis type="number" dataKey="step" tickFormatter={(v) => v.toLocaleString()} fontSize={10} />
                  <YAxis type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => Number(v).toFixed(2)} fontSize={10} />
                  <Tooltip labelFormatter={(v) => `step ${Number(v).toLocaleString()}`} formatter={(v: any) => [`loss ${(Number(v)).toFixed(3)}`]} />
                  <Line type="monotone" dataKey="loss" stroke={C.violet} strokeWidth={2.5} dot={false} name="loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="glass rounded-xl p-4">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
            Hallucinated names <span className="normal-case font-normal">· inference on the trained model</span>
          </h4>
          <p className="text-[11px] text-slate-400 mb-3">
            Autoregressive sampling (temperature ~0.6), mirroring the reference&rsquo;s inference loop. Click
            <strong>Sample names</strong> after training.
          </p>
          {samples.length === 0 ? (
            <p className="text-sm text-slate-400">
              {trained ? 'The model is trained — hit "Sample names" to hear what it learned.' : 'Train first, then this readout fills with fresh names.'}
            </p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {samples.map((s, i) => (
                <span key={i} className="rounded-lg px-2.5 py-1.5 text-sm font-bold text-white"
                  style={{ background: 'linear-gradient(135deg,#8b5cf6,#6d28d9)' }}>
                  {s}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function VecStack({ k, v, colorK, colorV }: { k: number[]; v: number[]; colorK: string; colorV: string }) {
  const render = (vec: number[], color: string) => (
    <div className="flex gap-0.5" style={{ width: `${Math.max(vec.length * 5, 20)}px` }}>
      {vec.map((cell, i) => (
        <div key={i} className="rounded-[2px]"
          style={{ width: 4, height: 8, backgroundColor: color, opacity: 0.35 + 0.5 * Math.min(1, Math.abs(cell)) }} />
      ))}
    </div>
  );
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-0.5">
        <span className="text-[7px] font-bold text-[var(--color-accent)]">K</span>{render(k, colorK)}
      </div>
      <div className="flex items-center gap-0.5">
        <span className="text-[7px] font-bold text-emerald-500">V</span>{render(v, colorV)}
      </div>
    </div>
  );
}

// Live KV-cache shape readout: shows the array dimensions and how many values
// are stored, growing through prefill (prompt positions) then decode (generated).
function KvSizeGauge({ positions, promptPositions, maxPositions, nEmbd, nLayer, nHead }: {
  positions: number;
  promptPositions: number;
  maxPositions: number;
  nEmbd: number;
  nLayer: number;
  nHead: number;
}) {
  const headDim = nEmbd / nHead;
  // Each position stores K + V for every head, for every layer.
  const valsPerPos = 2 /* K + V */ * nLayer * nEmbd;
  const totalValues = positions * valsPerPos;
  const prefillValues = Math.min(positions, promptPositions) * valsPerPos;
  const genValues = Math.max(0, totalValues - prefillValues);
  const genPositions = Math.max(0, positions - promptPositions);
  const maxValues = maxPositions * valsPerPos;

  const pct = maxPositions ? (positions / maxPositions) * 100 : 0;
  const pctPrompt = maxPositions ? (Math.min(positions, promptPositions) / maxPositions) * 100 : 0;
  const pctGen = Math.max(0, pct - pctPrompt);

  return (
    <div className="mt-3">
      <div className="flex items-center justify-between text-[11px] mb-1.5">
        <span className="font-semibold text-slate-700">KV cache shape</span>
        <span className="font-mono font-bold text-slate-900 text-xs">{fmtNum(totalValues)} values stored</span>
      </div>
      <div className="rounded-lg bg-slate-50 px-3 py-2 font-mono text-[11px] text-slate-600 mb-2 flex flex-wrap gap-x-2">
        <span>K+V</span>
        <span>[n_layer={nLayer}]</span>
        <span>[n_pos={positions}]</span>
        <span>[2 · n_head={nHead} · head_dim={headDim}]</span>
        <span className="text-slate-400">= {fmtNum(totalValues)}</span>
      </div>
      <div className="h-2.5 rounded-full bg-slate-200 overflow-hidden flex">
        <div className="h-full" style={{ width: `${pctPrompt}%`, background: C.accent }} />
        <div className="h-full" style={{ width: `${pctGen}%`, background: '#9aa5ff' }} />
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1.5 text-[10px] text-slate-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-sm align-middle" style={{ background: C.accent }} />
          prefill {Math.min(positions, promptPositions)} pos · {fmtNum(prefillValues)} values
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-sm align-middle" style={{ background: '#9aa5ff' }} />
          generated {genPositions} pos · {fmtNum(genValues)} values
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-sm bg-slate-200 align-middle" />
          max {maxPositions} pos · {fmtNum(maxValues)} values
        </span>
      </div>
      <p className="text-[10px] text-slate-400 mt-1">
        {positions === 0
          ? 'One position stores K + V = 2 · 1 layer · 16 dims = 32 values. Prefill adds a block per prompt token; each Generate step adds one more.'
          : `Each new position adds K + V = ${valsPerPos} values (2 · nLayer · nEmbd). Blue = prompt tokens from prefill; lighter = generated.`}
      </p>
    </div>
  );
}
