import React, { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { ArrowRight, RotateCcw, Play, Sparkles } from 'lucide-react';
import { cn } from '../../lib/utils';
import KpiCard from '../ui/KpiCard';
import SliderControl from '../ui/SliderControl';
import SegmentedControl from '../ui/SegmentedControl';
import InfoPopover from '../ui/InfoPopover';
import KvUsageExplain from '../ui/KvUsageExplain';
import Figure from '../shell/Figure';
import GoDeeper from '../shell/GoDeeper';
import { MicroGPT, type KV, type ForwardResult } from '../../lib/microgpt';
import { CHART as C } from '../../lib/theme';
import { cloneKV, VecStack, KvSizeGauge } from './shared';
import { TokenizerDemo, PrefillVisual, CausalAttentionGrid, DecodeLoop, SamplingExplain } from './explainers';

export default function MicroGptLab({ model, trained, sessionKey }: {
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

  const headDim = model.config.nEmbd / model.config.nHead;

  return (
    <div className="space-y-5">
      {/* model shape */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <KpiCard label="Layers" value={model.config.nLayer} subValue="depth" />
        <KpiCard label="Embedding" value={model.config.nEmbd} subValue="hidden dim" />
        <KpiCard label="Max ctx" value={model.config.blockSize} unit="tokens" />
        <KpiCard label="Heads" value={model.config.nHead} subValue={`d_head=${headDim}`} />
        <KpiCard label="Vocab" value={model.vocab.length + 1} subValue="chars + BOS" />
      </div>

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
              style={{ background: 'var(--color-accent)' }}
            >
              <span className="inline-flex items-center gap-1.5"><Play className="w-3.5 h-3.5" /> Prefill</span>
            </button>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!ready || done}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
              style={{ background: 'var(--color-emerald-600)' }}
            >
              <span className="inline-flex items-center gap-1.5">Generate <ArrowRight className="w-3.5 h-3.5" /></span>
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-slate-600 bg-white hover:bg-white"
            >
              <span className="inline-flex items-center gap-1.5"><RotateCcw className="w-3.5 h-3.5" /> Reset</span>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-5">
          <div className="flex-1 min-w-[200px]">
            <SliderControl
              label="Temperature"
              value={temperature}
              min={0.05}
              max={2}
              step={0.05}
              onChange={setTemperature}
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-600">Sampling</span>
            <SegmentedControl
              size="sm"
              options={[{ value: 'greedy', label: 'Greedy' }, { value: 'sample', label: 'Sample' }]}
              value={greedy ? 'greedy' : 'sample'}
              onChange={(v) => setGreedy(v === 'greedy')}
              className="w-44"
            />
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

        <GoDeeper label="Greedy vs. sampling, and what temperature does">
          <SamplingExplain />
        </GoDeeper>
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
                  style={!isBOS && !isPrompt ? { background: 'var(--color-accent)' } : undefined}
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
          <span><span className="inline-block w-2.5 h-2.5 rounded-sm mr-1 align-middle" style={{ background: '#99e0ff' }} />generated</span>
          <span><span className="inline-block w-2.5 h-2.5 rounded-sm ring-1 ring-[var(--color-accent)] mr-1 align-middle" />newest KV†</span>
        </div>
        {done && <p className="text-xs text-emerald-600 font-semibold mt-2">Generation ended (stop token or max context reached).</p>}
        <GoDeeper label="How tokens are assigned">
          <TokenizerDemo />
        </GoDeeper>
        <GoDeeper label="Why prefill runs the whole prompt at once">
          <PrefillVisual />
        </GoDeeper>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* KV cache grid */}
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-1.5 mb-1">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">KV cache (per layer · per position)</h4>
            <InfoPopover conceptId="kv-cache" iconSize={13} />
          </div>
          <Figure takeaway="Each new position adds one block of stored numbers; the darker portion is the prompt (from prefill), the lighter portion is what generation has added since.">
            <KvSizeGauge
              positions={kvLen}
              promptPositions={promptCount}
              maxPositions={model.config.blockSize}
              nEmbd={model.config.nEmbd}
              nLayer={model.config.nLayer}
              nHead={model.config.nHead}
            />
          </Figure>
          <p className="text-[11px] text-slate-400 mb-3">
            K stored in <span className="text-[var(--color-accent)] font-semibold">blue</span>, V in{' '}
            <span className="text-emerald-500 font-semibold">green</span>. At each step the newest query is scored
            against every stored <strong>K</strong> (blue = the match) and the stored <strong>V</strong> values are
            blended by those weights (green = the carry). {kvLen === 0 ? 'Empty until you prefill.' : `Filled for ${kvLen} positions.`}
          </p>
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
          <GoDeeper label="What's actually being cached">
            <KvUsageExplain />
          </GoDeeper>
        </div>

        {/* attention */}
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-1.5 mb-1">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Attention this step (per head)</h4>
            <InfoPopover conceptId="attention-intensity" iconSize={13} />
          </div>
          {attention.length === 0 ? (
            <p className="text-sm text-slate-400">No forward pass yet — run <strong>Prefill</strong>.</p>
          ) : (
            <Figure takeaway="Each bar shows how much weight the newest token's query places on every cached position, per head; taller/brighter segments are getting more attention.">
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
                              : p === 0 ? '#d7bfe1' : `hsl(${220 - p * 8}, 70%, ${55 + p * 2}%)`,
                          }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Figure>
          )}
          <GoDeeper label="Reading the causal mask">
            <CausalAttentionGrid />
          </GoDeeper>
        </div>
      </div>

      {/* next-token probabilities */}
      <div className="glass rounded-xl p-4">
        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
          Next-token probabilities <span className="normal-case font-normal">softmax(logits / T)</span>
        </h4>
        <Figure takeaway={`The highlighted bar is the token that was actually sampled${sampledId >= 0 ? ` (${model.labelForToken(sampledId)})` : ''}; a tall single bar means the model is confident, a flat distribution means it's guessing.`}>
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
        </Figure>
        <GoDeeper label="What happens on each Generate click">
          <DecodeLoop />
        </GoDeeper>
      </div>

      <p className="text-xs text-slate-400 pt-2 border-t border-slate-200">
        Want the full mechanical walkthrough with diagrams for every stage? See{' '}
        <a href="#one-token" className="text-accent underline underline-offset-2">One token, end to end</a>,{' '}
        <a href="#attention" className="text-accent underline underline-offset-2">Attention</a>, and{' '}
        <a href="#kv-cache" className="text-accent underline underline-offset-2">The KV cache</a> in the curriculum.
      </p>
    </div>
  );
}
