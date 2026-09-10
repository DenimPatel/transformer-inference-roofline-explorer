import React from 'react';
import { cn } from '../../lib/utils';
import { MicroGPT, DEFAULT_CONFIG } from '../../lib/microgpt';
import { CHART as C } from '../../lib/theme';
import { ArrowRight } from 'lucide-react';

/**
 * Small, self-contained demos of each pipeline stage (tokenizer, embedding,
 * forward pass, prefill, causal attention, decode). Shared between the
 * "One token, end to end" curriculum section and the microGPT Playground's
 * GoDeeper disclosures, so the mechanics are explained in exactly one place.
 */

export const modelStatic = new MicroGPT();

export function TokenizerDemo() {
  const sample = 'the cat sat';
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

export function VectorRow({ label, values, color }: { label: string; values: number[]; color: string }) {
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

export function EmbeddingDemo() {
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

export function ForwardDiagram() {
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

export function PrefillVisual() {
  const tokens = ['BOS', 't', 'h', 'e', ' ', 'c', 'a', 't'];
  return (
    <div>
      <div className="flex flex-wrap gap-1.5 mb-4">
        {tokens.map((t, i) => (
          <div key={i}
            className="rounded-lg px-2.5 py-1.5 text-xs font-bold text-white"
            style={{ background: i === 0 ? '#7b4b90' : 'var(--color-accent)' }}>
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

export function CausalAttentionGrid() {
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

export function DecodeLoop() {
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
            <span className="flex items-center justify-center w-6 h-6 rounded-lg text-white text-xs font-bold" style={{ background: s.color }}>
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

export function SamplingExplain() {
  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-600 leading-relaxed">
        Logits are not probabilities. We turn them into a distribution with
        <code className="mx-1 bg-slate-100 px-1 rounded">softmax(logits / temperature)</code>, then either take
        the argmax (greedy) or draw a random token weighted by the distribution — exactly microgpt&rsquo;s
        <code className="mx-1 bg-slate-100 px-1 rounded">random.choices(range(vocab), weights=probs)</code>.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-4">
          <h4 className="font-bold text-slate-800 mb-2 text-sm">Greedy (T &rarr; 0)</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Always pick the most likely token. Deterministic, safe, but repetitive — the model&rsquo;s top
            pick is often a generic continuation.
          </p>
        </div>
        <div className="glass rounded-xl p-4">
          <h4 className="font-bold text-slate-800 mb-2 text-sm">Sampling with T &gt; 0</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Divide logits by T to sharpen or flatten the distribution before sampling. Higher T = more
            randomness and "creativity"; lower T = closer to greedy.
          </p>
        </div>
      </div>
    </div>
  );
}
