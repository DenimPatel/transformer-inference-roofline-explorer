import { useState } from 'react';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { cn } from '../../lib/utils';

const FEATURES = [
  {
    id: 'ttft',
    tag: 'ttft' as const,
    title: 'Time-to-First-Token (TTFT)',
    body: 'A latency goal that training never had. Batching many prefills together delays TTFT because a user sees nothing until their prefill finishes. Maximizing MFU helps TTFT.',
  },
  {
    id: 'continuous',
    tag: 'continuous-batching' as const,
    title: 'Continuous Batching',
    body: 'An orchestrator prefills variable-length requests and generates all active ones together, so short generations are not blocked by long ones. Pairs with prefix caching.',
  },
  {
    id: 'disaggregated',
    tag: 'disaggregated-serving' as const,
    title: 'Disaggregated Prefill/Generate',
    body: 'Prefill (compute-bound) and generation (memory-bound) run on separate pools, with KV caches shipped over the network. Each pool is specialized; the cost is moving KV.',
  },
  {
    id: 'prefix',
    tag: 'prefix-caching' as const,
    title: 'Prefix Caching',
    body: 'Shared prompt prefixes produce identical KV, so cache them (as an LRU trie) and skip re-prefill for that prefix. Smaller KV caches let more prefixes fit.',
  },
  {
    id: 'speculative',
    tag: 'speculative-decoding' as const,
    title: 'Speculative Decoding',
    body: 'A small draft model proposes K tokens; the big model verifies them in one batched forward. Because generation is memory-bound, the extra math is almost free — more tokens per step.',
  },
  {
    id: 'paged',
    tag: 'paged-attention' as const,
    title: 'Paged Attention',
    body: 'Store KV in OS-style page tables and read it non-contiguously, so every request uses only the memory it needs — no padding waste on ragged sequences.',
  },
];

export default function LessonServing({ onComplete }: { onComplete: () => void }) {
  const [prefillMs, setPrefillMs] = useState(910);
  const [decodeMs, setDecodeMs] = useState(19);
  const [decodeBatch, setDecodeBatch] = useState(32);
  const [genServers, setGenServers] = useState(1);

  // Balance: prefill servers P = G · decodeBatch · (prefillMs / decodeMs)
  const requiredP = genServers * decodeBatch * (prefillMs / decodeMs);
  const pDisplay = Math.ceil(requiredP);

  return (
    <LessonShell
      number={11}
      title="Serving Systems"
      subtitle="TTFT, continuous batching, disaggregation, prefix caching and speculative decoding — how real engines turn the roofline into an SLA."
      sourceRef="reference/scaling-book/inference.md — Designing an Effective Inference Engine · Continuous batching · Prefix caching · Appendix D"
    >
      <GlassCard className="p-5 space-y-3">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="ttft" />
          <ConceptTag id="continuous-batching" />
          <ConceptTag id="disaggregated-serving" />
          <ConceptTag id="prefix-caching" />
          <ConceptTag id="speculative-decoding" />
          <ConceptTag id="paged-attention" />
        </div>
        <p className="text-sm leading-relaxed text-slate-600">
          The roofline tells you <em>where</em> time goes; serving systems decide what to do about it. Because generation is
          memory-bound and prefill is compute-bound, the winning move is often to treat them as different workloads —
          separate batches, separate servers, or verifiers — rather than one monolithic loop.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {FEATURES.map((f) => (
            <div key={f.id} className="glass rounded-xl p-4">
              <div className="flex items-center justify-between mb-1.5">
                <h4 className="font-bold text-slate-800 text-sm">{f.title}</h4>
                <ConceptTag id={f.tag} />
              </div>
              <p className="text-[12.5px] leading-relaxed text-slate-600">{f.body}</p>
            </div>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5 space-y-4">
        <h3 className="font-bold text-slate-800">How many prefill servers do you need? (disaggregation ratio)</h3>
        <p className="text-sm leading-relaxed text-slate-600">
          Prefill and generation have different throughputs, so a disaggregated fleet should be sized so neither pool idles.
          For a prefill time <code className="text-xs">P_ms</code>, decode time <code className="text-xs">D_ms</code> and decode batch{' '}
          <code className="text-xs">B</code>:
        </p>
        <div className="flex flex-wrap gap-2 text-[12px] text-slate-600 font-mono">
          <span className="glass-chip px-2.5 py-1">prefill servers ≈ G · B · (P_ms / D_ms)</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <SliderControl label="Prefill time" value={prefillMs} min={100} max={3000} step={10} onChange={setPrefillMs} unit="ms" conceptId="prefill" />
          <SliderControl label="Decode time" value={decodeMs} min={5} max={80} step={1} onChange={setDecodeMs} unit="ms" conceptId="generation" />
          <SliderControl label="Decode batch" value={decodeBatch} min={1} max={256} step={1} onChange={setDecodeBatch} unit="seqs" conceptId="critical-batch" />
          <SliderControl label="Generate servers (G)" value={genServers} min={1} max={16} step={1} onChange={setGenServers} unit="" conceptId="disaggregated-serving" />
        </div>

        <div className="glass rounded-xl p-4 flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] text-slate-400">Required prefill servers (P)</div>
            <div className="text-2xl font-mono font-bold text-slate-800">{pDisplay}</div>
            <div className="text-[11px] text-slate-400">for G = {genServers} generate server{genServers > 1 ? 's' : ''}</div>
          </div>
          <div className={cn('text-sm font-bold rounded-xl px-3 py-2', pDisplay <= 8 ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700')}>
            {pDisplay <= 8 ? 'Fleet roughly balanced' : 'Prefill-bound: add prefill capacity (or shrink KV/prefix caching)'}
          </div>
        </div>
        <p className="text-[12.5px] text-slate-500">
          For the textbook LLaMA-70B bf16 figures (910 ms prefill, 19 ms decode at BS=32) one generate server needs ≈3 prefill
          servers — the P = 3G rule. Prefix caching and smaller KV caches cut the prefill side of this balance.
        </p>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'Why does disaggregated serving put prefill and generation on separate servers?',
            options: [
              'To double the FLOPs of each',
              'Because prefill is compute-bound and generation is memory-bound, so each pool can be sized and specialized for its own roofline',
              'To avoid using KV caches',
              'To remove the need for continuous batching',
            ],
            answer: 1,
            explain: 'The two phases hit opposite sides of the ridge, so separating them lets each use dedicated hardware and reduces request blocking.',
          },
          {
            q: 'Speculative decoding increases throughput during generation because…',
            options: [
              'It makes generation compute-bound',
              'Generation is memory-bound, so verifying extra draft tokens is nearly free FLOPs — more tokens per step',
              'It removes the KV cache',
              'It shrinks the model',
            ],
            answer: 1,
            explain: 'The big model is not FLOPs-limited during decode, so scoring K draft tokens in one pass yields more tokens per step for the same math.',
          },
        ]}
      />
    </LessonShell>
  );
}
