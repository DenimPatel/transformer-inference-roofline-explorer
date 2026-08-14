import { useState } from 'react';
import { motion } from 'motion/react';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';

const PARAMS_BYTES = 7e9 * 2;            // 7B params at bf16
const H80 = 80e9;

function seg(bytes: number) {
  return Math.max(2, Math.min(100, (bytes / H80) * 100));
}

export default function LessonKVCache({ onComplete }: { onComplete: () => void }) {
  const [context, setContext] = useState(8192);
  const [batch, setBatch] = useState(16);
  const [kvFactor, setKvFactor] = useState(1); // 1 = MHA, 0.25 = GQA 4:1
  const bytesPerToken = 128 * 1024;

  const kvBytes = context * bytesPerToken * batch * kvFactor;
  const total = PARAMS_BYTES + kvBytes;

  return (
    <LessonShell
      number={5}
      title="The KV Cache: Where Inference Memory Goes"
      subtitle="The caches that accelerate generation can dwarf your parameters — and that is the memory/bandwidth cost that matters."
      sourceRef="reference/scaling-book/inference.md — What about memory?"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="kv-cache" />
          <ConceptTag id="gqa" />
          <ConceptTag id="memory-bound" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          KV cache size = <code className="text-xs">2 · bytes/float · H · K · L · T</code>. Shrink it (GQA, quantization,
          paging) and you can fit bigger batches and longer contexts — and every token becomes cheaper to stream.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-4">
            <SliderControl label="Context length" value={context} min={256} max={65536} step={256} onChange={setContext} unit="tkns" logScale conceptId="kv-cache" />
            <SliderControl label="Concurrent requests" value={batch} min={1} max={1024} step={1} onChange={setBatch} unit="seqs" logScale />
            <div>
              <div className="text-xs font-medium text-slate-600 mb-1.5">KV-head sharing</div>
              <div className="flex glass rounded-xl p-1">
                {[
                  { v: 1, l: 'MHA 1:1' },
                  { v: 0.5, l: '2:1' },
                  { v: 0.25, l: '4:1 (GQA)' },
                  { v: 0.125, l: '8:1 (GQA)' },
                ].map((o) => (
                  <button
                    key={o.v}
                    type="button"
                    onClick={() => setKvFactor(o.v)}
                    className={`flex-1 text-xs py-1.5 rounded-lg transition-colors ${kvFactor === o.v ? 'bg-white shadow text-slate-800 font-bold' : 'text-slate-500'}`}
                  >
                    {o.l}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="glass-strong rounded-2xl p-4 space-y-3">
            <div className="text-xs text-slate-500">HBM footprint vs 80 GB H100</div>
            <div className="space-y-2">
              <Bar label="Parameters (7B @ bf16)" bytes={PARAMS_BYTES} color="#5b7cfa" />
              <Bar label="KV cache" bytes={kvBytes} color="#f5a623" />
            </div>
            <div className="text-xs text-slate-500">
              Total:{' '}
              <span className="font-mono font-bold text-slate-700">{(total / 1e9).toFixed(1)} GB</span> ·{' '}
              {((total / H80) * 100).toFixed(0)}% of an H100
            </div>
            <p className="text-[11px] text-slate-400">
              KV now uses {(kvBytes / 1e9).toFixed(1)} GB — {kvFactor < 1 ? `shrunken ${(1 / kvFactor).toFixed(0)}× by head sharing` : 'full MHA size'}.
            </p>
          </div>
        </div>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'Which statement about the KV cache is true?',
            options: [
              'It stores optimizer states',
              'It grows with context length and batch, and can exceed parameter memory',
              'It is tiny and negligible',
              'It only exists during training',
            ],
            answer: 1,
            explain: 'KV size = 2·bytes·H·K·L·T — it grows with context and batch and can dwarf parameters.',
          },
          {
            q: 'Why does GQA help inference?',
            options: [
              'It adds parameters',
              'It shares KV heads to shrink the cache and raise attention intensity',
              'It doubles the FLOPs',
              'It removes the need for memory',
            ],
            answer: 1,
            explain: 'Fewer KV heads means a smaller cache — better batching, longer contexts, cheaper streaming.',
          },
        ]}
      />
    </LessonShell>
  );
}

function Bar({ label, bytes, color }: { label: string; bytes: number; color: string }) {
  const pct = seg(bytes);
  return (
    <div>
      <div className="flex justify-between text-[11px] mb-1">
        <span className="text-slate-600">{label}</span>
        <span className="font-mono text-slate-500">{(bytes / 1e9).toFixed(1)} GB</span>
      </div>
      <div className="h-3 rounded-full bg-slate-200/50 overflow-hidden">
        <motion.div
          key={pct}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  );
}
