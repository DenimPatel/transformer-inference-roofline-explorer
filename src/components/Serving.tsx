import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  Rocket, Cpu, Database, Gauge, Layers, Zap, Sparkles, MemoryStick,
  Server, Timer, Braces, Hash, Activity,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { HARDWARE_PROFILES } from '../lib/hardware';
import { findModel, type ModelProfile } from '../lib/models';
import ConceptTag from './ui/ConceptTag';
import InfoPopover from './ui/InfoPopover';

// ---------------------------------------------------------------------------
// Palette + formatters (mirror DeepDive / TokenGeneration)
// ---------------------------------------------------------------------------
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

function fmtBytes(b: number): string {
  if (b >= 1e12) return `${(b / 1e12).toFixed(1)} TB`;
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} KB`;
  return `${b.toFixed(0)} B`;
}

function fmtFlops(f: number): string {
  if (f >= 1e15) return `${(f / 1e15).toFixed(2)} PFLOP`;
  if (f >= 1e12) return `${(f / 1e12).toFixed(1)} TFLOP`;
  return `${(f / 1e9).toFixed(1)} GFLOP`;
}

function fmtMs(ms: number): string {
  if (ms <= 0) return '0 ms';
  if (ms < 1) return `${(ms * 1000).toFixed(1)} µs`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

// ---------------------------------------------------------------------------
// Facts from the Qwen/Qwen3.8-Flash-Next model card
// (profile JSON supplies active/total/context/kv; rest is static)
// ---------------------------------------------------------------------------
const MODEL_FACTS = {
  lmParamsB: 125,
  nGramParamsB: 51,
  mtpParamsB: 4,
  residentParamsB: 180,     // 125 + 51 + 4
  activeParamsB: 6,
  hiddenDim: 2560,
  nLayers: 48,
  nExperts: 512,
  activeExperts: 10,        // +1 shared
  sparseBudgetTokens: 2048, // QSA budget
};

const PRECISIONS = [
  { bytes: 2, label: 'BF16', note: 'native tensor type' },
  { bytes: 1, label: 'FP8/int8', note: 'roughly halves weight bytes' },
  { bytes: 0.5, label: 'FP4/int4', note: 'quarter the weight bytes' },
];

const GPU_IDS = ['B200 (Blackwell)', 'Rubin GPU (R100)'];

function ChartTip({ active, payload, label, unit = '' }: any) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="glass-tooltip px-3 py-2 text-xs space-y-1">
      {label !== undefined && label !== '' && <div className="font-semibold text-slate-800">{label}</div>}
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-slate-500 capitalize">{p.name}:</span>
          <span className="font-mono font-semibold text-slate-800">{fmtNum(p.value, 2)}{p.unit || unit}</span>
        </div>
      ))}
    </div>
  );
}

function Slider({ label, value, min, max, step = 1, onChange, format }: any) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <label className="font-medium text-slate-700">{label}</label>
        <span className="font-mono text-slate-900">{format ? format(value) : fmtNum(value)}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="glass-slider w-full"
      />
    </div>
  );
}

function BoundBadge({ compute }: { compute: boolean }) {
  return (
    <span className={cn(
      'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold',
      compute ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700',
    )}>
      <span className={cn('w-2 h-2 rounded-full', compute ? 'bg-emerald-500' : 'bg-rose-500')} />
      {compute ? 'Compute-bound' : 'Memory / bandwidth-bound'}
    </span>
  );
}

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

// ---------------------------------------------------------------------------
// Main tab — owns the shared GPU + precision state so every section agrees
// ---------------------------------------------------------------------------
export default function ServingTab() {
  const model = findModel('qwen3-8-flash-next');

  const [hwId, setHwId] = useState('B200 (Blackwell)');
  const hw = HARDWARE_PROFILES.find((h) => h.id === hwId) || HARDWARE_PROFILES[0];
  const [precision, setPrecision] = useState(hw.bytesPerParam);

  const sections = [
    { id: 'facts', label: 'Model Facts', icon: Hash },
    { id: 'gpu', label: 'GPU', icon: Server },
    { id: 'fit', label: 'Does It Fit?', icon: Database },
    { id: 'budget', label: 'Per-Request Budget', icon: Gauge },
    { id: 'where', label: 'Where Resources Go', icon: Activity },
    { id: 'thinking', label: 'How to Think', icon: Sparkles },
  ];

  return (
    <div className="pb-16 max-w-6xl mx-auto mt-6 px-4">
      {/* ---- Hero ---- */}
      <section className="text-center mb-10">
        <div className="inline-flex items-center gap-2 glass-chip px-3 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-4">
          <Rocket className="w-3.5 h-3.5 text-accent" /> Serving a real model on a real GPU
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
          One request, <span className="text-accent">from first token to last</span>
        </h1>
        <p className="text-slate-500 max-w-3xl mx-auto leading-relaxed">
          How many <strong>FLOPs</strong> and how much <strong>memory</strong> does a single inference
          request actually burn — and <em>where</em> does that compute and memory go? We take
          <strong> Qwen3.8 Flash-Next</strong> (125B params, 6B active) and serve it on NVIDIA Blackwell{' '}
          <strong>B200</strong> or the newest <strong>Rubin R100</strong>, breaking every number down by phase.
        </p>
        <div className="flex flex-wrap justify-center gap-2 mt-6">
          <ConceptTag id="prefill" />
          <ConceptTag id="generation" />
          <ConceptTag id="kv-cache" />
          <ConceptTag id="moe" />
          <ConceptTag id="linear-attention" />
          <ConceptTag id="sparse-attention" />
          <ConceptTag id="n-gram-embedding" />
        </div>
      </section>

      {/* ---- Section nav ---- */}
      <nav className="sticky top-0 z-30 -mx-2 px-2 py-3 mb-8 blur-[1px] backdrop-blur-md bg-[#eef1fb]/70 rounded-2xl">
        <div className="flex gap-1.5 overflow-x-auto custom-scrollbar py-1">
          {sections.map((s) => (
            <a key={s.id} href={`#${s.id}`}
              className="shrink-0 inline-flex items-center gap-1.5 glass-chip px-3 py-1.5 text-[11px] font-semibold text-slate-600 hover:text-accent hover:border-accent/40 transition-colors">
              <s.icon className="w-3.5 h-3.5" /> {s.label}
            </a>
          ))}
        </div>
      </nav>

      {/* 01 Model Facts */}
      <SectionCard id="facts" icon={Hash} color={C.accent} number="01" title="Model Facts">
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            The first thing to internalize about serving: <strong>activation is cheap, residency is
            expensive</strong>. Qwen3.8 Flash-Next runs <strong>only ~6B parameters per token</strong>.
            But it must keep <strong>~180B parameters resident</strong> on the accelerator while it serves
            (125B LM + 51B n-gram embedding + 4B MTP). That split — tiny active set, huge resident set — is
            the whole story of why this model is hard to <em>fit</em> but cheap to <em>run</em>.
          </p>
          <p>
            It is also <strong>hybrid attention</strong>: most layers are Gated DeltaNet (linear), with Qwen
            Sparse Attention (QSA) inserted every few layers. DeltaNet keeps a constant-size recurrent state
            instead of a growing per-token KV cache, and QSA only lets each query look at a ~2048-token
            budget. Together they collapse the long-context memory wall.
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <HeroKpi icon={Cpu} label="LM params" value={`${fmtNum(MODEL_FACTS.lmParamsB)}B`} sub="total, MoE" />
          <HeroKpi icon={Zap} label="Active / token" value={`${fmtNum(MODEL_FACTS.activeParamsB)}B`} sub="10 routed + 1 shared" />
          <HeroKpi icon={MemoryStick} label="Resident" value={`${fmtNum(MODEL_FACTS.residentParamsB)}B`} sub="+51B n-gram +4B MTP" />
          <HeroKpi icon={Layers} label="Layers" value={`${MODEL_FACTS.nLayers}`} sub={`${MODEL_FACTS.hiddenDim} hidden`} />
        </div>
        <p className="text-sm text-slate-500 mt-6">
          Read more: the <ConceptTag id="n-gram-embedding" /> family of parameters is what lets the model scale
          capacity without adding per-token FLOPs.
        </p>
      </SectionCard>

      {/* 02 GPU */}
      <SectionCard id="gpu" icon={Server} color={C.sky} number="02" title="Pick the GPU">
        <p className="text-sm text-slate-500 mb-4">
          Everything below is computed live from this accelerator&apos;s real numbers. Try switching between
          NVIDIA Blackwell (FP8) and the newest Rubin (FP4) to feel how the roofline moves.
        </p>
        <div className="flex flex-wrap gap-2 mb-5">
          {HARDWARE_PROFILES.filter((h) => GPU_IDS.includes(h.id)).map((h) => (
            <button key={h.id} type="button" onClick={() => setHwId(h.id)}
              className={cn('rounded-xl px-4 py-2 text-sm font-semibold transition-colors border',
                hwId === h.id ? 'bg-accent text-white border-accent' : 'bg-white/70 text-slate-600 border-slate-200 hover:border-accent/50')}>
              {h.id} <span className="text-[10px] opacity-80">({h.arch})</span>
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
          <HeroKpi icon={Cpu} label="Peak compute" value={fmtFlops((hw.tflops * 1e12 * (2 / hw.bytesPerParam))) + '/s'} sub={`${hw.id}`} />
          <HeroKpi icon={Database} label="HBM capacity" value={fmtBytes(hw.capacity * 1e9)} sub="per GPU" />
          <HeroKpi icon={Gauge} label="Bandwidth" value={fmtBytes(hw.memBw * 1e12) + '/s'} sub="HBM" />
          <HeroKpi icon={Zap} label="Precision" value={hw.bytesPerParam === 0.5 ? 'FP4' : 'FP8'} sub={`${hw.bytesPerParam} B/param`} />
        </div>
        <p className="text-xs text-slate-400 flex items-center gap-1.5">
          <InfoPopover conceptId="roofline" iconSize={12} /> These feeds go straight into the per-request budget below.
        </p>
      </SectionCard>

      {/* 03 Does it fit? */}
      <SectionCard id="fit" icon={Database} color={C.amber} number="03" title="Does It Fit in HBM?">
        <FitSection hw={hw} model={model} precision={precision} setPrecision={setPrecision} />
      </SectionCard>

      {/* 04 Budget */}
      <BudgetSection hw={hw} model={model} precision={precision} />

      {/* 05 Where resources go */}
      <WhereResources hw={hw} model={model} precision={precision} />

      {/* 06 How to think */}
      <ServThinking />
    </div>
  );
}

// ---------------------------------------------------------------------------
// 03 Fit
// ---------------------------------------------------------------------------
function FitSection({ hw, model, precision, setPrecision }: any) {
  const capacity = hw.capacity * 1e9;
  const footprintAt = (bytesNum: number, p: number) => bytesNum * 1e9 * p;
  const footprintBytes = footprintAt(MODEL_FACTS.residentParamsB, precision);
  const lmOnly = footprintAt(MODEL_FACTS.lmParamsB, precision);
  const fits = footprintBytes <= capacity;
  const pct = (footprintBytes / capacity) * 100;

  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <div className="glass rounded-xl p-5">
        <h3 className="font-bold text-slate-800 mb-1">Weight footprint vs {hw.id} HBM</h3>
        <p className="text-sm text-slate-500 mb-4">
          Pick a precision. The resident ~180B model has to live entirely in HBM, alongside the growing KV cache.
        </p>
        <div className="grid grid-cols-3 gap-1.5 mb-4">
          {PRECISIONS.map((p) => (
            <button key={p.bytes} type="button" onClick={() => setPrecision(p.bytes)}
              className={cn('glass rounded-md py-1.5 text-[11px] font-semibold transition-colors',
                precision === p.bytes ? 'bg-accent text-white' : 'text-slate-600 hover:border-accent/40')}>
              {p.label}
            </button>
          ))}
        </div>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-500">LM weights (125B)</span>
            <span className="font-mono font-semibold text-slate-800">{fmtBytes(lmOnly)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">+ n-gram &amp; MTP (55B)</span>
            <span className="font-mono font-semibold text-slate-800">{fmtBytes(footprintBytes - lmOnly)}</span>
          </div>
          <div className="flex justify-between border-t border-slate-200 pt-2">
            <span className="text-slate-700 font-semibold">Total resident</span>
            <span className="font-mono font-bold text-slate-900 text-lg">{fmtBytes(footprintBytes)}</span>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>vs {hw.id} HBM ({fmtBytes(capacity)})</span>
              <span className="font-mono">{pct.toFixed(0)}%</span>
            </div>
            <div className="h-2.5 rounded-full bg-slate-200 overflow-hidden">
              <div className={cn('h-full rounded-full transition-all', fits ? 'bg-emerald-500' : 'bg-rose-500')}
                style={{ width: `${Math.min(pct, 100)}%` }} />
            </div>
            <p className={cn('text-xs mt-2', fits ? 'text-slate-500' : 'text-rose-600 font-semibold')}>
              {fits
                ? `The ${PRECISIONS.find((p) => p.bytes === precision)?.label} model fits one GPU (${Math.floor((capacity - footprintBytes) / 1e9)} GB left for KV).`
                : `Does NOT fit at ${PRECISIONS.find((p) => p.bytes === precision)?.label}. Quantize, shard across GPUs, or offload the n-gram tables.`}
            </p>
          </div>
        </div>
      </div>

      <div className="glass rounded-xl p-5">
        <h3 className="font-bold text-slate-800 mb-1">Why precision is non-negotiable</h3>
        <p className="text-sm text-slate-500 mb-4 leading-relaxed">
          At BF16 the ~180B model is {fmtBytes(MODEL_FACTS.residentParamsB * 1e9 * 2)} — larger than both GPUs&apos;
          HBM. FP8 halves it and FP4 quarters it. Sharding across GPUs also multiplies total bandwidth (your decode
          clock), so fit and speed go together.
        </p>
        <ul className="space-y-2 text-sm text-slate-600">
          {PRECISIONS.map((p) => {
            const fitsP = footprintAt(MODEL_FACTS.residentParamsB, p.bytes) <= capacity;
            return (
              <li key={p.bytes} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-accent" />
                <span className="font-semibold">{p.label}:</span>
                <span className="font-mono">{fmtBytes(footprintAt(MODEL_FACTS.residentParamsB, p.bytes))}</span>
                <span className="text-xs text-slate-400">fits {hw.id} — {fitsP ? '✓' : '✗'}</span>
              </li>
            );
          })}
        </ul>
        <div className="text-xs text-slate-400 mt-4 leading-relaxed">
          The <ConceptTag id="n-gram-embedding" /> tables are more amenable to offloading than MoE expert weights,
          and the KV cache (next section) fills whatever HBM is left over.
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 04 Budget
// ---------------------------------------------------------------------------
function BudgetSection({ hw, model, precision }: any) {
  const [prefillTokens, setPrefillTokens] = useState(1024);
  const [outputTokens, setOutputTokens] = useState(256);
  const [batch, setBatch] = useState(1);

  const flops = hw.tflops * 1e12 * (2 / precision);
  const memBw = hw.memBw * 1e12;
  const activeParams = MODEL_FACTS.activeParamsB * 1e9;
  const residentBytes = MODEL_FACTS.residentParamsB * 1e9 * precision;
  const kvPerToken = (model?.kvPerTokenKb ?? 24) * 1024;

  // compute clocks (single request chain)
  const prefillFlops = 2 * prefillTokens * activeParams;
  const prefillComputeMs = (prefillFlops / flops) * 1000;
  const decodeFlopsPerToken = 2 * activeParams;
  const decodeComputeMs = (decodeFlopsPerToken / flops) * 1000;

  // memory clock, two models of decode
  const classicKvPerStep = prefillTokens * kvPerToken * batch;
  const classicMemMs = ((residentBytes + classicKvPerStep) / memBw) * 1000;
  const sparseKvPerStep = Math.min(MODEL_FACTS.sparseBudgetTokens, prefillTokens) * kvPerToken * batch;
  const moeMemMs = ((activeParams * precision + sparseKvPerStep) / memBw) * 1000;

  const classicDecodeMs = Math.max(decodeComputeMs, classicMemMs);
  const moeDecodeMs = Math.max(decodeComputeMs, moeMemMs);

  const totalFlops = prefillFlops + outputTokens * decodeFlopsPerToken;
  const ttftMs = prefillComputeMs * batch;
  const totalTimeMoe = moeDecodeMs * outputTokens + ttftMs;
  const memForOneReq = (kvPerToken * (prefillTokens + outputTokens) + residentBytes) / 1e9;

  return (
    <SectionCard id="budget" icon={Gauge} color={C.violet} number="04" title="Per-Request FLOPs &amp; Memory">
      <p className="text-sm text-slate-500 mb-4">
        A request = <strong>prefill</strong> of {fmtNum(prefillTokens)} prompt tokens, then{' '}
        <strong>{fmtNum(outputTokens)}</strong> decode steps. Because activated FLOPs/token are tiny (6B), the math
        is cheap; the question is always whether you are waiting on compute or on memory.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">
        <div className="glass rounded-xl p-5 lg:col-span-1">
          <div className="space-y-4">
            <Slider label="Prompt tokens (prefill)" value={prefillTokens} min={64} max={262144} step={64}
              onChange={setPrefillTokens} format={(v: number) => fmtNum(v)} />
            <Slider label="Output tokens (decode)" value={outputTokens} min={16} max={16384} step={16}
              onChange={setOutputTokens} format={(v: number) => fmtNum(v)} />
            <Slider label="Concurrent requests (batch)" value={batch} min={1} max={1024}
              onChange={setBatch} format={(v: number) => v.toLocaleString()} />
          </div>
        </div>

        <div className="glass rounded-xl p-5">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Per-request totals</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-slate-500">Total FLOPs (1 req)</span>
              <span className="font-mono font-bold text-slate-900">{fmtFlops(totalFlops)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Prefill FLOPs</span>
              <span className="font-mono">{fmtFlops(prefillFlops)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Decode FLOPs</span>
              <span className="font-mono">{fmtFlops(outputTokens * decodeFlopsPerToken)}</span></div>
            <div className="flex justify-between border-t border-slate-200 pt-2"><span className="text-slate-500">HBM: weights + KV</span>
              <span className="font-mono font-bold text-slate-900">{memForOneReq.toFixed(1)} GB</span></div>
          </div>
        </div>

        <div className="glass rounded-xl p-5">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Latency on {hw.id}</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-slate-500">TTFT (prefill)</span>
              <span className="font-mono font-bold text-slate-900">{fmtMs(ttftMs)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Decode / token (classic)</span>
              <span className="font-mono">{fmtMs(classicDecodeMs)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Decode / token (realistic)</span>
              <span className="font-mono font-bold text-emerald-600">{fmtMs(moeDecodeMs)}</span></div>
            <div className="flex justify-between border-t border-slate-200 pt-2"><span className="text-slate-500">End-to-end (realistic)</span>
              <span className="font-mono font-bold text-slate-900">{fmtMs(totalTimeMoe)}</span></div>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-1">Where the decode time goes</h3>
          <p className="text-sm text-slate-500 mb-4">
            Two views of the same step. The <span className="text-rose-500 font-semibold">classic dense roofline</span>{' '}
            streams all ~180B weights + full KV every token; the <span className="text-emerald-600 font-semibold">realistic</span>{' '}
            view streams only ~6B active weights + a sparse ~2k-token KV budget.
          </p>
          <DecodeBreakdown
            classicCompMs={decodeComputeMs} classicMemMs={classicMemMs}
            moeCompMs={decodeComputeMs} moeMemMs={moeMemMs}
            classicBound={classicMemMs > decodeComputeMs}
            moeBound={moeMemMs > decodeComputeMs}
          />
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-1">The prefill ≈ compute-bound pole</h3>
          <p className="text-sm text-slate-500 mb-4 leading-relaxed">
            Prefill reuses the weights across all {fmtNum(prefillTokens)} prompt tokens, so its intensity is huge and it
            rides the <span className="text-emerald-600 font-semibold">compute roof</span>. Generation (T = 1, small active
            set) collapses toward the <span className="text-rose-500 font-semibold">bandwidth slope</span>.
          </p>
          <div className="space-y-3 text-sm text-slate-600">
            <div className="flex justify-between p-3 glass rounded-lg"><span>Prefill compute time</span>
              <span className="font-mono font-bold text-emerald-600">{fmtMs(prefillComputeMs)}</span></div>
            <div className="flex justify-between p-3 glass rounded-lg"><span>Decode compute / token</span>
              <span className="font-mono">{fmtMs(decodeComputeMs)}</span></div>
            <div className="flex justify-between p-3 glass rounded-lg"><span>Simultaneous requests</span>
              <span className="font-mono font-bold text-slate-900">{fmtNum(batch)}</span></div>
            <p className="text-xs text-slate-400">
              Same request served to {fmtNum(batch)}{' '}{batch === 1 ? 'client' : 'clients'} via continuous batching:
              prefill TTFT scales with batch, but each decode step serves all {fmtNum(batch)} at once — that is how you
              keep decode near the ridge.
            </p>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}

function DecodeBreakdown({ classicCompMs, classicMemMs, moeCompMs, moeMemMs, classicBound }: any) {
  const data = [
    { name: 'Classic (dense roofline)', compute: +classicCompMs.toFixed(3), memory: +classicMemMs.toFixed(3) },
    { name: 'Qwen3.8-Flash-Next (MoE + sparse)', compute: +moeCompMs.toFixed(3), memory: +moeMemMs.toFixed(3) },
  ];
  const maxV = Math.max(classicMemMs, classicCompMs, moeMemMs, moeCompMs) || 1;
  return (
    <div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 6, right: 24, left: 8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
            <XAxis type="number" domain={[0, maxV]} tickFormatter={(v: number) => fmtMs(v)} fontSize={10} />
            <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 10 }} />
            <Tooltip content={<ChartTip unit=" ms" />} labelFormatter={(l: any, p: any) => p[0]?.payload?.name || l} />
            <Bar dataKey="memory" name="Memory time" stackId="a" fill={C.memory} barSize={28} />
            <Bar dataKey="compute" name="Compute time" stackId="a" fill={C.compute} radius={[6, 6, 6, 6]} barSize={28} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-3 text-xs text-slate-500 mt-2">
        <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: C.memory }} /> memory (bandwidth)</span>
        <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: C.compute }} /> compute</span>
        <BoundBadge compute={!classicBound} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 05 Where resources go
// ---------------------------------------------------------------------------
function WhereResources({ hw, model, precision }: any) {
  const flops = hw.tflops * 1e12 * (2 / precision);
  const memBw = hw.memBw * 1e12;
  const activeParams = MODEL_FACTS.activeParamsB * 1e9;
  const residentBytes = MODEL_FACTS.residentParamsB * 1e9 * precision;
  const kvPerToken = (model?.kvPerTokenKb ?? 24) * 1024;
  const P = 1024, O = 256;

  const prefillCompute = (2 * P * activeParams / flops) * 1000;
  const prefillMem = ((residentBytes + P * kvPerToken) / memBw) * 1000;
  const decodeCompute = (2 * activeParams / flops) * 1000;
  const decodeMem = ((activeParams * precision + Math.min(MODEL_FACTS.sparseBudgetTokens, P) * kvPerToken) / memBw) * 1000;

  const data = [
    { name: 'Prefill', compute: +prefillCompute.toFixed(3), memory: +prefillMem.toFixed(3) },
    { name: 'Decode (per token)', compute: +decodeCompute.toFixed(3), memory: +decodeMem.toFixed(3) },
  ];
  const maxV = Math.max(prefillCompute, prefillMem, decodeCompute, decodeMem) || 1;

  const phases = [
    {
      name: 'Prefill — ingest prompt',
      tag: <ConceptTag id="prefill" />,
      actions: [
        'Stream each weight matrix once, reused across all prompt tokens',
        'Run Q/K/V projections + a big causal matmul over the whole prompt',
        'Write the sparse KV budget into the cache',
      ],
      where: 'Compute-bound — FLOPs dominate',
      color: C.compute,
    },
    {
      name: 'Decode — one token at a time',
      tag: <ConceptTag id="generation" />,
      actions: [
        'Read ~6B active weights (MoE routers pick a few of 512 experts)',
        'Read the bounded sparse KV for the query to attend to',
        'Compute logits, sample, append the token — repeat',
      ],
      where: 'Memory-bound — bandwidth dominates',
      color: C.amber,
    },
  ];

  return (
    <SectionCard id="where" icon={Activity} color={C.compute} number="05" title="Where Compute &amp; Memory Go, Per Phase">
      <div className="grid lg:grid-cols-2 gap-5">
        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-1">Compute vs memory, live</h3>
          <p className="text-sm text-slate-500 mb-4">
            Stacked time for each phase on {hw.id} (compute = {fmtFlops(flops) + '/s'}, memory = {fmtBytes(memBw) + '/s'}).
          </p>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} layout="vertical" margin={{ top: 6, right: 24, left: 24, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
                <XAxis type="number" domain={[0, maxV]} tickFormatter={(v: number) => fmtMs(v)} fontSize={10} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
                <Tooltip content={<ChartTip unit=" ms" />} labelFormatter={(l: any, p: any) => p[0]?.payload?.name || l} />
                <Bar dataKey="memory" name="Memory time" stackId="a" fill={C.memory} barSize={30} />
                <Bar dataKey="compute" name="Compute time" stackId="a" fill={C.compute} radius={[6, 6, 6, 6]} barSize={30} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap gap-3 text-xs text-slate-500 mt-3">
            <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: C.memory }} /> memory (bandwidth)</span>
            <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: C.compute }} /> compute</span>
          </div>
          <p className="text-xs text-slate-400 mt-3 leading-relaxed">
            Prefill is dominated by the compute bar (high intensity from weight reuse); decode flips to a memory bar — the
            same roofline flip taught in the <em>Deep Dive</em> tab.
          </p>
        </div>

        <div className="space-y-4">
          {phases.map((ph) => (
            <div key={ph.name} className="glass rounded-xl p-5">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-bold text-slate-800" style={{ color: ph.color }}>{ph.name}</h4>
                {ph.tag}
              </div>
              <ul className="space-y-1.5 text-sm text-slate-600 mb-3">
                {ph.actions.map((a, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="w-2 h-2 rounded-full mt-1.5 shrink-0" style={{ background: ph.color }} />
                    <span>{a}</span>
                  </li>
                ))}
              </ul>
              <div className={cn('inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold',
                ph.color === C.compute ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700')}>
                {ph.where}
              </div>
            </div>
          ))}
          <div className="text-xs text-slate-400 leading-relaxed">
            Because <ConceptTag id="linear-attention" /> keeps a constant state and <ConceptTag id="sparse-attention" /> caps
            the KV read, the decode memory bar stays small — this model&apos;s superpower at long context.
          </div>
        </div>
      </div>
    </SectionCard>
  );
}

// ---------------------------------------------------------------------------
// 06 How to think
// ---------------------------------------------------------------------------
function ServThinking() {
  return (
    <SectionCard id="thinking" icon={Sparkles} color={C.violet} number="06" title="How to Think About Serving">
      <div className="grid md:grid-cols-2 gap-4">
        <ThinkCard icon={Timer} color={C.accent} title="Decide which clock you are on"
          body="Every phase is max(FLOPs / FLOPs·s⁻¹, Bytes / Bytes·s⁻¹). Ask: am I waiting on the silicon or on HBM? That one question tells you whether to optimize compute or memory.">
          <ConceptTag id="arithmetic-intensity" />
        </ThinkCard>
        <ThinkCard icon={Braces} color={C.compute} title="Batch to amortize"
          body="Decode is memory-bound because weights are streamed per token. Serving N requests at once reuses each weight load across N tokens, dividing the memory clock and pushing decode toward the ridge. This is why continuous batching exists.">
          <ConceptTag id="continuous-batching" />
        </ThinkCard>
        <ThinkCard icon={Layers} color={C.violet} title="MoE shifts the bottleneck"
          body="A 125B model with 6B active means fewer FLOPs, but you still hold ~180B resident (memory) and pay an all-to-all to gather routed experts (network). The real cost moves off pure weight-streaming.">
          <ConceptTag id="moe" />
        </ThinkCard>
        <ThinkCard icon={MemoryStick} color={C.memory} title="The KV cache is the silent killer"
          body="GQA, paged/kv-sharded cache, and quantization all shrink the KV clock. With DeltaNet + QSA the per-token KV is bounded, which is exactly why long-context agentic serving stays cheap.">
          <ConceptTag id="kv-cache" />
          <ConceptTag id="paged-attention" />
          <ConceptTag id="kv-sharding" />
        </ThinkCard>
      </div>
      <div className="glass rounded-xl p-5 mt-6">
        <h3 className="font-bold text-slate-800 mb-2">The 60-second mental model</h3>
        <ol className="space-y-2 text-sm text-slate-600 list-decimal pl-5">
          <li><strong>Count FLOPs/token</strong> ≈ 2 · active params (6B here → tiny).</li>
          <li><strong>Count bytes/stream</strong> ≈ resident weights + KV read (big, but shrinkable via precision &amp; sparsity).</li>
          <li><strong>Compare each</strong> to the chip&apos;s FLOPs/s and bandwidth → you know whether you are compute- or memory-bound.</li>
          <li><strong>Optimize the bottleneck</strong>: quantize/shard for memory, batch for bandwidth amortization, and exploit the model&apos;s hybrid sparsity to cap KV.</li>
        </ol>
      </div>
    </SectionCard>
  );
}

function ThinkCard({ icon: I, color, title, body, children }: any) {
  return (
    <div className="glass rounded-xl p-5">
      <div className="flex items-center gap-2 mb-2">
        <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg text-white" style={{ background: color }}>
          <I className="w-4 h-4" />
        </span>
        <h4 className="font-bold text-slate-800">{title}</h4>
      </div>
      <p className="text-sm text-slate-500 leading-relaxed mb-3">{body}</p>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}
