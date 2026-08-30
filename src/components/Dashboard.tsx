import React, { useState, useMemo, useEffect } from 'react';
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ComposedChart, Scatter,
} from 'recharts';
import {
  BarChart3, Cpu, Server, Info, Layers, DollarSign, HelpCircle, GraduationCap, Activity, Braces, Rocket,
} from 'lucide-react';
import { cn } from '../lib/utils';
import { HARDWARE_PROFILES } from '../lib/hardware';
import { MODEL_PROFILES, findModel } from '../lib/models';
import { physicalUnits, makeCurve, optimalBatch, drainTimeMs, contextCrossover, maxThroughput, currentIntensity, RowInputs } from '../lib/roofline';
import KpiCard from './ui/KpiCard';
import SliderControl from './ui/SliderControl';
import InfoPopover from './ui/InfoPopover';
import ConceptTag from './ui/ConceptTag';
import DeepDiveTab from './DeepDive';
import TokenGenerationTab from './TokenGeneration';
import LearnJourney from './learn/LearnJourney';
import ConceptGlossary from './learn/ConceptGlossary';
import CompetitiveAnalysis from './CompetitiveAnalysis';
import ServingTab from './Serving';

type Tab = 'learn' | 'simulator' | 'comparison' | 'deep_dive' | 'token_generation' | 'serving';

const ECONOMICS_REGIONS = [
  { id: 'US Hyperscale', priceKwh: 0.07, pue: 1.10, desc: 'Texas/Iowa' },
  { id: 'US Commercial', priceKwh: 0.12, pue: 1.40, desc: 'Standard DC' },
  { id: 'Europe Avg', priceKwh: 0.22, pue: 1.35, desc: 'Germany/UK' },
  { id: 'Green / Hydro', priceKwh: 0.04, pue: 1.08, desc: 'Quebec/Iceland' },
  { id: 'Custom', priceKwh: 0.10, pue: 1.25, desc: 'Custom Settings' },
];

const TABS: { key: Tab; label: string; icon: React.ComponentType<{ style?: React.CSSProperties; className?: string }> }[] = [
  { key: 'learn', label: 'Learn', icon: GraduationCap },
  { key: 'simulator', label: 'Interactive Lab', icon: Server },
  { key: 'comparison', label: 'Competitive Analysis', icon: BarChart3 },
  { key: 'deep_dive', label: 'Deep Dive', icon: Info },
  { key: 'token_generation', label: 'Token Generation', icon: Braces },
  { key: 'serving', label: 'Serving', icon: Rocket },
];

function initialTab(): Tab {
  try {
    const raw = localStorage.getItem('roofline-learn-progress');
    const arr = raw ? (JSON.parse(raw) as string[]) : [];
    return arr.length === 0 ? 'learn' : 'simulator';
  } catch {
    return 'learn';
  }
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<Tab>(initialTab);
  const [glossaryOpen, setGlossaryOpen] = useState(false);

  // Hardware (Simulator)
  const [activeProfileId, setActiveProfileId] = useState<string>('H100 SXM5');
  const [flopsTera, setFlopsTera] = useState(989);
  const [memBwTera, setMemBwTera] = useState(3.35);
  const [memCapGb, setMemCapGb] = useState(80);
  const [tdpWatts, setTdpWatts] = useState(700);
  const [hardwarePrice, setHardwarePrice] = useState(30000);

  // Economics
  const [regionId, setRegionId] = useState('US Hyperscale');
  const [priceKwh, setPriceKwh] = useState(0.07);
  const [pue, setPue] = useState(1.10);
  const [utilization, setUtilization] = useState(65);
  const [amortizationYears, setAmortizationYears] = useState(3);

  // Model
  const [activeModelId, setActiveModelId] = useState('deepseek-v3');
  const [totalParamsB, setTotalParamsB] = useState(671);
  const [activeParamsB, setActiveParamsB] = useState(37);
  const [bytesPerParam, setBytesPerParam] = useState(2);
  const [contextLen, setContextLen] = useState(16384);
  const [bytesPerTokenKb, setBytesPerTokenKb] = useState(96);

  // Current Operation
  const [currentBatchSize, setCurrentBatchSize] = useState(256);

  // Focus profiles for comparison
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>(['H100 SXM5', 'B200 (Blackwell)', 'Rubin GPU (R100)', 'TPU v8i (Inference)', 'Groq + Rubin Pipeline']);

  // Derived Physical Units
  const units = physicalUnits({ flopsTera, memBwTera, memCapGb, totalParamsB, activeParamsB, bytesPerParam, bytesPerTokenKb });

  const isLiquid = activeProfileId.toLowerCase().includes('rubin') || activeProfileId.toLowerCase().includes('liquid');
  const rowInputs: RowInputs = {
    totalParams: units.totalParams,
    activeParams: units.activeParams,
    bytesPerParam,
    contextLen,
    bytesPerToken: units.bytesPerToken,
    tdpWatts,
    hardwarePrice,
    priceKwh,
    pue,
    utilization,
    amortizationYears,
    isLiquid,
  };

  const handleModelSelect = (id: string) => {
    setActiveModelId(id);
    if (id === 'custom' || id === '') return;
    const m = MODEL_PROFILES.find((x) => x.id === id);
    if (m) {
      setTotalParamsB(m.totalParamsB);
      setActiveParamsB(m.activeParamsB);
      setBytesPerParam(m.bytesPerParam);
      setContextLen(m.contextLen);
      setBytesPerTokenKb(m.kvPerTokenKb);
    }
  };

  const handleProfileSelect = (id: string) => {
    setActiveProfileId(id);
    if (id === 'custom') return;
    const p = HARDWARE_PROFILES.find((x) => x.id === id);
    if (p) {
      setFlopsTera(p.tflops);
      setMemBwTera(p.memBw);
      setMemCapGb(p.capacity);
      setBytesPerParam(p.bytesPerParam);
      setTdpWatts(p.tdp);
      setHardwarePrice(p.price);
    }
  };

  const handleRegionSelect = (id: string) => {
    setRegionId(id);
    if (id === 'Custom') return;
    const r = ECONOMICS_REGIONS.find((x) => x.id === id);
    if (r) {
      setPriceKwh(r.priceKwh);
      setPue(r.pue);
    }
  };

  const chartData = useMemo(() => makeCurve(units, rowInputs), [units, rowInputs]);

  // Derived Metrics
  const hardwareRatio = units.hardwareRatio;
  const optimalBatchSize = optimalBatch(units);
  const drainTime = drainTimeMs(units);
  const contextLengthCrossover = contextCrossover(units);
  const maxTokensPerSec = maxThroughput(units);

  const currentStat = chartData.reduce((prev, curr) =>
    Math.abs(curr.batchSize - currentBatchSize) < Math.abs(prev.batchSize - currentBatchSize) ? curr : prev
  );

  const currentStatIntensity = currentIntensity(currentBatchSize, units, rowInputs);

  const toggleProfile = (id: string) => {
    setSelectedProfiles((s) => (s.includes(id) ? s.filter((p) => p !== id) : [...s, id]));
  };

  useEffect(() => {
    document.body.style.overflow = glossaryOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [glossaryOpen]);

  return (
    <div className="min-h-screen text-slate-900 font-sans flex flex-col selection:bg-[var(--color-accent)]/20">
      <div className="aurora" aria-hidden="true" />
      <ConceptGlossary open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />

      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong px-4 sm:px-6 h-16 flex items-center justify-between shrink-0 border-b border-white/50">
        <div className="flex items-center space-x-3 min-w-0">
          <div className="flex items-center justify-center w-9 h-9 rounded-xl text-white shrink-0"
            style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)', boxShadow: '0 6px 20px -6px rgba(91,124,250,0.7)' }}>
            <BarChart3 className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h1 className="text-lg sm:text-xl font-bold tracking-tight text-slate-800 truncate">Roofline Explorer</h1>
            <p className="text-[11px] text-slate-400 font-normal hidden sm:block">Interactive visualizer for Transformer Inference Economics</p>
          </div>
        </div>

        {/* Tabs */}
        <nav className="flex glass rounded-2xl p-1 gap-0.5">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = activeTab === t.key;
            return (
              <button
                key={t.key}
                type="button"
                onClick={() => setActiveTab(t.key)}
                className={cn('relative px-2.5 sm:px-3.5 py-1.5 text-xs sm:text-sm font-medium rounded-xl transition-colors', active ? 'text-slate-800' : 'text-slate-500 hover:text-slate-700')}
              >
                {active && (
                  <span className="absolute inset-0 rounded-xl bg-white shadow border border-slate-200/60" />
                )}
                <span className="relative z-10 flex items-center gap-1.5">
                  <Icon style={{ width: 15, height: 15 }} className="sm:hidden" />
                  <span className="hidden sm:inline">{t.label}</span>
                </span>
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setGlossaryOpen(true)}
            className="hidden lg:inline-flex items-center gap-1.5 text-xs font-semibold text-slate-600 glass-chip px-3 py-1.5 hover:text-[var(--color-accent)] hover:bg-white/80"
          >
            <HelpCircle style={{ width: 14, height: 14 }} /> Concepts
          </button>
          <span className="hidden md:flex items-center gap-1.5 text-[11px] font-mono text-slate-500">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> LIVE
          </span>
        </div>
      </header>

      <main className={cn("w-full max-w-[1600px] mx-auto p-4 sm:p-6 flex-1", (activeTab === 'learn' || activeTab === 'deep_dive' || activeTab === 'token_generation' || activeTab === 'serving') ? "block" : "grid grid-cols-1 xl:grid-cols-12 gap-6")}>

        {activeTab === 'learn' && <LearnJourney onLab={() => setActiveTab('simulator')} />}

        {activeTab !== 'learn' && activeTab !== 'deep_dive' && activeTab !== 'token_generation' && activeTab !== 'serving' && (
          <section className="xl:col-span-3 space-y-5">
            {activeTab === 'simulator' && (
              <ControlPanel title="Hardware Environment" icon={<Server className="w-4 h-4 text-[var(--color-accent)]" />} conceptId="roofline">
                <div className="mb-4">
                  <label className="text-xs font-semibold text-slate-500 mb-1.5 block uppercase tracking-wider flex items-center gap-1.5">
                    Hardware Preset <InfoPopover conceptId="ridge-point" iconSize={13} />
                  </label>
                  <select
                    className="glass-input w-full text-sm py-2 px-3 text-slate-800 font-medium"
                    value={activeProfileId}
                    onChange={(e) => handleProfileSelect(e.target.value)}
                  >
                    <option value="custom">Custom Profile</option>
                    {(['NVIDIA', 'Google', 'AMD', 'AWS', 'Groq', 'SambaNova', 'Hybrid'] as const).map((vendor) => (
                      <optgroup key={vendor} label={vendor}>
                        {HARDWARE_PROFILES.filter((p) => p.vendor === vendor).map((p) => (
                          <option key={p.id} value={p.id}>{p.id} ({p.arch})</option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </div>
                <SliderControl label="Compute" value={flopsTera} min={10} max={100000} step={1} onChange={setFlopsTera} unit="TFLOP/s" conceptId="flops" />
                <SliderControl label="Memory Bandwidth" value={memBwTera} min={0.5} max={200.0} step={0.1} onChange={setMemBwTera} unit="TB/s" conceptId="bandwidth" />
                <SliderControl label="Memory Capacity" value={memCapGb} min={0.5} max={512} step={0.5} onChange={setMemCapGb} unit="GB" />
                <SliderControl label="Thermal Design Power" value={tdpWatts} min={50} max={3000} step={10} onChange={setTdpWatts} unit="W" />
                <SliderControl label="Est. System Cost" value={hardwarePrice} min={1000} max={400000} step={1000} onChange={setHardwarePrice} unit="$" />
              </ControlPanel>
            )}

            <ControlPanel title="Datacenter Economics" icon={<DollarSign className="w-4 h-4 text-emerald-500" />} conceptId="tco">
              <div className="mb-4">
                <label className="text-xs font-semibold text-slate-500 mb-1.5 block uppercase tracking-wider">Geography / Power</label>
                <select className="glass-input w-full text-sm py-2 px-3 text-slate-800 font-medium" value={regionId} onChange={(e) => handleRegionSelect(e.target.value)}>
                  {ECONOMICS_REGIONS.map((r) => (
                    <option key={r.id} value={r.id}>{r.id} ({r.desc})</option>
                  ))}
                </select>
              </div>
              <SliderControl label="Electricity Price" value={priceKwh} min={0.01} max={0.50} step={0.01} onChange={setPriceKwh} unit="$/kWh" />
              <SliderControl label="PUE (Efficiency)" value={pue} min={1.0} max={2.0} step={0.01} onChange={setPue} unit=" " comment={isLiquid ? 'Liquid cooling applied automatically (max 1.05).' : ''} />
              <SliderControl label="Hardware Utilization" value={utilization} min={10} max={100} step={1} onChange={setUtilization} unit="%" />
              <SliderControl label="Amortization Period" value={amortizationYears} min={1} max={10} step={1} onChange={setAmortizationYears} unit="Yrs" />
            </ControlPanel>

            {activeTab === 'comparison' && (
              <ControlPanel title="Hardware Selection" icon={<Cpu className="w-4 h-4 text-[var(--color-accent)]" />}>
                <p className="text-xs text-slate-500 mb-3">Select profiles to compare latency and cost curves:</p>
                <div className="space-y-1 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                  {HARDWARE_PROFILES.map((profile) => (
                    <label key={profile.id} className="flex items-center space-x-2 py-1.5 px-2 hover:bg-white/60 cursor-pointer rounded-lg">
                      <input type="checkbox" className="rounded border-slate-300 text-[var(--color-accent)] focus:ring-[var(--color-accent)]" checked={selectedProfiles.includes(profile.id)} onChange={() => toggleProfile(profile.id)} />
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: profile.color }} />
                        <span className="text-sm font-medium text-slate-700">{profile.id}</span>
                      </div>
                    </label>
                  ))}
                </div>
              </ControlPanel>
            )}

            <ControlPanel title="Model Architecture" icon={<Layers className="w-4 h-4 text-blue-500" />} conceptId="matmul-intensity">
              <div className="mb-4">
                <label className="text-xs font-semibold text-slate-500 mb-1.5 block uppercase tracking-wider flex items-center gap-1.5">
                  Open-Source Model Preset <InfoPopover conceptId="kv-cache" iconSize={13} />
                </label>
                <select
                  className="glass-input w-full text-sm py-2 px-3 text-slate-800 font-medium"
                  value={activeModelId}
                  onChange={(e) => handleModelSelect(e.target.value)}
                >
                  <option value="custom">Custom Architecture</option>
                  {Array.from(new Set(MODEL_PROFILES.map((m) => m.family))).map((family) => (
                    <optgroup key={family} label={family}>
                      {MODEL_PROFILES.filter((m) => m.family === family).map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name} ({m.type === 'moe' ? `${m.totalParamsB}B / ${m.activeParamsB}B active` : `${m.totalParamsB}B`})
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                {findModel(activeModelId)?.description && (
                  <p className="mt-2 text-[11px] leading-snug text-slate-500">{findModel(activeModelId)?.description}</p>
                )}
              </div>
              <SliderControl label="Total Parameters" value={totalParamsB} min={1} max={10000} step={1} onChange={setTotalParamsB} unit="B" />
              <SliderControl label="Active Parameters" value={activeParamsB} min={1} max={totalParamsB} step={1} onChange={setActiveParamsB} unit="B" comment={`Sparsity: ${(totalParamsB / activeParamsB).toFixed(1)}x`} conceptId="kv-cache" />
              <div className="pt-2">
                <label className="text-xs text-slate-600 mb-1.5 block">Parameter Precision</label>
                <div className="flex glass p-1 rounded-xl">
                  {([
                    { v: 0.5, l: 'FP4' },
                    { v: 1, l: 'FP8' },
                    { v: 2, l: 'FP16/BF16' },
                  ] as const).map((o) => (
                    <button key={o.v} onClick={() => setBytesPerParam(o.v)}
                      className={cn('flex-1 text-xs py-1.5 rounded-lg transition-colors', bytesPerParam === o.v ? 'bg-white shadow text-slate-900 font-bold' : 'text-slate-500')}>
                      {o.l}
                    </button>
                  ))}
                </div>
              </div>
              <SliderControl label="Context Length" value={contextLen} min={128} max={5000000} step={128} onChange={setContextLen} unit="tkns" logScale conceptId="kv-cache" />
              <SliderControl label="KV Size / Token" value={bytesPerTokenKb} min={16} max={2048} step={16} onChange={setBytesPerTokenKb} unit="KB" />
            </ControlPanel>

            {activeTab === 'simulator' && (
              <ControlPanel title="Operating Point" icon={<Activity className="w-4 h-4 text-emerald-500" />} conceptId="latency-throughput">
                <SliderControl label="Current Batch Size" value={currentBatchSize} min={1} max={32768} step={1} onChange={setCurrentBatchSize} unit="seqs" logScale conceptId="critical-batch" />
              </ControlPanel>
            )}
          </section>
        )}

        {activeTab !== 'learn' && activeTab !== 'deep_dive' && activeTab !== 'token_generation' && activeTab !== 'serving' && (
          <section className="xl:col-span-9 space-y-5 flex flex-col min-h-0">
            {activeTab === 'simulator' && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  <KpiCard label="Limiting Factor" value={currentStat.isMemoryBound ? 'Memory Bound' : 'Compute Bound'}
                    colorClass={currentStat.isMemoryBound ? 'text-amber-500' : 'text-emerald-600'}
                    subValue={`Cost: $${currentStat.totalCost1M.toFixed(2)} / 1M tkns`}
                    conceptId={currentStat.isMemoryBound ? 'memory-bound' : 'compute-bound'} />
                  <KpiCard label="Energy per Token" value={currentStat.joulesPerToken.toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 })}
                    unit="Joules" subValue={`Pwr Draw: ${currentStat.powerDrawW.toFixed(0)}W`} conceptId="tco" />
                  <KpiCard label="Performance Density" value={(flopsTera * 1000 / tdpWatts).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
                    unit="TFLOP/W" conceptId="flops" />
                  <KpiCard label="Max Token Throughput" value={maxTokensPerSec.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    unit="tokens/s" conceptId="latency-throughput" />
                  <KpiCard label="Optimal Batch (Balance)" value={optimalBatchSize.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    unit="seqs" conceptId="critical-batch" />
                  <KpiCard label="Drain Time (Capacity/BW)" value={drainTime.toFixed(1)} unit="ms" conceptId="memory-bound" />
                </div>

                {/* Roofline Grounding */}
                <GlassPanelBlue>
                  <div className="flex items-start justify-between gap-3 mb-1">
                    <div>
                      <h3 className="text-lg font-bold text-slate-800 mb-1">Where Is the Current Operating Point?</h3>
                      <p className="text-sm text-slate-500">
                        Single decode/generation step for <strong>{activeProfileId}</strong> — it sits below the ridge: the memory-bound regime. See <em>Deep Dive → Prefill vs Generation</em> for why prefill flips.
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <ConceptTag id="roofline" />
                      <InfoPopover conceptId="roofline" label="Roofline concept" />
                    </div>
                  </div>
                  <RooflineGrounding peakFlops={units.flops} peakBw={units.memBw} ridge={hardwareRatio} opIntensity={currentStatIntensity} opName="Current decode step" />
                </GlassPanelBlue>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 flex-1 min-h-[400px]">
                  <GlassChart title="Latency vs. Batch Size" subtitle="Total latency = max(memory time, compute time). Batch on a log scale."
                    conceptId="latency-throughput">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
                        <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
                        <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                          tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} stroke="#cbd5e1" tick={{ fill: '#64748b', fontSize: 11 }}
                          label={{ value: 'Batch Size (log)', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }} />
                        <YAxis stroke="#cbd5e1" tick={{ fill: '#64748b', fontSize: 11 }}
                          label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(val) => val.toFixed(1)} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                        <Line type="monotone" dataKey="tCompute" name="Compute Time" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                        <Line type="monotone" dataKey="tMemory" name="Memory Time" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                        <Line type="monotone" dataKey="latency" name="Total Latency" stroke="#1e293b" strokeWidth={3} dot={false} />
                        <ReferenceLine x={currentStat.batchSize} stroke="#94a3b8" strokeWidth={2} label={{ position: 'top', value: 'Current', fill: '#64748b', fontSize: 10 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </GlassChart>

                  <GlassChart title="Total Cost of Ownership" subtitle={`Electricity (${regionId}) + hardware amortization over ${amortizationYears} yrs.`}
                    conceptId="tco">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
                        <CartesianGrid strokeDasharray="4 4" stroke="#e2e8f0" vertical={false} />
                        <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                          tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} stroke="#cbd5e1" tick={{ fill: '#64748b', fontSize: 11 }}
                          label={{ value: 'Batch Size (log)', position: 'insideBottom', offset: -12, fill: '#64748b', fontSize: 11 }} />
                        <YAxis stroke="#cbd5e1" tick={{ fill: '#64748b', fontSize: 11 }}
                          label={{ value: 'Cost (USD/1M tkns)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} tickFormatter={(v) => '$' + Number(v).toFixed(2)} />
                        <Tooltip content={<CustomCostTooltip />} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                        <Line type="monotone" dataKey="costElec1M" name="Power Cost" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                        <Line type="monotone" dataKey="costHardware1M" name="Hardware CapEx" stroke="#64748b" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                        <Line type="monotone" dataKey="totalCost1M" name="Total TCO" stroke="#10b981" strokeWidth={3} dot={false} />
                        <ReferenceLine x={currentStat.batchSize} stroke="#94a3b8" strokeWidth={2} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </GlassChart>
                </div>

                {/* Theory Section */}
                <div className="glass rounded-2xl p-5 text-[13px] text-slate-600 leading-relaxed">
                  <h3 className="font-bold mb-3 flex items-center gap-2 text-slate-800">
                    <Info className="w-4 h-4 text-[var(--color-accent)]" /> Key Principles
                    <span className="ml-auto flex gap-2"><ConceptTag id="roofline" /><ConceptTag id="tco" /><ConceptTag id="generation" /></span>
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
                    <div className="flex items-start gap-2"><InfoPopover conceptId="compute-bound" iconSize={13} className="mt-0.5" /><span><strong className="text-slate-800">Compute Time:</strong> grows linearly with batch (B × ActiveParams / TFLOPs). Lower precision multiplies effective throughput.</span></div>
                    <div className="flex items-start gap-2"><InfoPopover conceptId="memory-bound" iconSize={13} className="mt-0.5" /><span><strong className="text-slate-800">Memory Time:</strong> large base offset from loading all weights, plus linear KV fetch. Bandwidth is the wall.</span></div>
                    <div className="flex items-start gap-2"><InfoPopover conceptId="tco" iconSize={13} className="mt-0.5" /><span><strong className="text-slate-800">Cost Economics:</strong> TCO = Electricity (Joules × PUE × $/kWh) + amortized hardware. Liquid systems can reach PUE ≈ 1.05.</span></div>
                    <div className="flex items-start gap-2"><InfoPopover conceptId="network-roofline" iconSize={13} className="mt-0.5" /><span><strong className="text-slate-800">Hybrid Pipelines:</strong> LPUs for prefill (bandwidth) + dense GPUs for decode (compute/capacity) target TTFT and throughput together.</span></div>
                  </div>
                </div>
              </>
            )}

            {activeTab === 'comparison' && (
              <CompetitiveAnalysis units={units} rowInputs={rowInputs} selectedProfiles={selectedProfiles} />
            )}
          </section>
        )}

        {activeTab === 'deep_dive' && <DeepDiveTab hardwareProfileId={activeProfileId} />}
        {activeTab === 'token_generation' && <TokenGenerationTab />}
        {activeTab === 'serving' && <ServingTab />}
      </main>
    </div>
  );
}

/* ---------------- presentational ------------- */

function ControlPanel({ title, icon, children, conceptId }: { title: string; icon: React.ReactNode; children: React.ReactNode; conceptId?: string }) {
  return (
    <div className="glass-card p-5">
      <h2 className="flex items-center text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-4">
        <span className="mr-2 p-1.5 rounded-lg bg-white/70">{icon}</span> {title}
        {conceptId && <span className="ml-auto normal-case"><InfoPopover conceptId={conceptId} iconSize={14} /></span>}
      </h2>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function GlassPanelBlue({ children }: { children: React.ReactNode }) {
  return <div className="glass rounded-2xl p-5">{children}</div>;
}

function GlassChart({ title, subtitle, children, conceptId }: { title: string; subtitle: string; children: React.ReactNode; conceptId?: string }) {
  return (
    <div className="glass-card p-5 flex flex-col">
      <div className="mb-3 shrink-0 flex items-start justify-between gap-2">
        <div>
          <h3 className="text-lg font-bold text-slate-800 mb-0.5">{title}</h3>
          <p className="text-sm text-slate-500">{subtitle}</p>
        </div>
        {conceptId && <InfoPopover conceptId={conceptId} className="text-slate-300" />}
      </div>
      <div className="flex-1 w-full min-h-[300px]">{children}</div>
    </div>
  );
}

function RooflineGrounding({ peakFlops, peakBw, ridge, opIntensity, opName }: any) {
  const data = useMemo(() => {
    const pts = [];
    const minI = 0.05;
    const maxI = 400000;
    for (let i = Math.log10(minI); i <= Math.log10(maxI); i += 0.08) {
      const intensity = Math.pow(10, i);
      pts.push({ intensity, achievable: Math.min(peakBw * intensity, peakFlops) });
    }
    return pts;
  }, [peakFlops, peakBw]);

  const achieved = Math.min(peakBw * opIntensity, peakFlops);
  const opData = [{ intensity: opIntensity, achieved }];
  const isMemBound = opIntensity < ridge;

  return (
    <div>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 20, right: 24, left: 0, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.4} />
            <XAxis dataKey="intensity" scale="log" domain={['dataMin', 'dataMax']} type="number" tickFormatter={(v) => (Number(v) < 1 ? Number(v).toFixed(1) : Number(v).toFixed(0))} label={{ value: 'Arithmetic Intensity (FLOPs/B) — log', position: 'bottom', offset: -10, fontSize: 11 }} />
            <YAxis scale="log" domain={['dataMin', 'dataMax']} type="number" tickFormatter={(v) => (v / 1e12).toFixed(1)} label={{ value: 'Throughput (TFLOPs/s) — log', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip labelFormatter={(v: any) => `Intensity: ${Number(v).toFixed(2)} FLOPs/B`} formatter={(v: any) => [`${(Number(v) / 1e12).toFixed(2)} TFLOP/s`]} />
            <Line type="monotone" dataKey="achievable" name="Roofline" stroke="#5b7cfa" strokeWidth={3} dot={false} />
            <ReferenceLine x={ridge} stroke="#f43f5e" strokeDasharray="4 4" label={{ position: 'top', value: `ridge ≈ ${ridge.toFixed(0)}`, fill: '#f43f5e', fontSize: 10 }} />
            <Scatter data={opData} dataKey="achieved" fill={isMemBound ? '#f59e0b' : '#10b981'} name={`${opName}: ${isMemBound ? 'memory-bound' : 'compute-bound'}`} shape="circle" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-2 mt-3 text-xs">
        <span className="glass-chip px-2.5 py-1 text-slate-600">Operating intensity: <span className="font-mono font-bold">{opIntensity.toFixed(2)}</span> FLOPs/B</span>
        <span className="glass-chip px-2.5 py-1 text-slate-600">Hardware ridge: <span className="font-mono font-bold">{ridge.toFixed(1)}</span> FLOPs/B</span>
        <span className={cn('px-2.5 py-1 rounded-full font-bold', isMemBound ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700')}>
          {isMemBound ? 'Memory-bound (below ridge)' : 'Compute-bound (at ridge/peak)'}
        </span>
      </div>
    </div>
  );
}

const TooltipShell = ({ children }: { children: React.ReactNode }) => (
  <div className="glass-tooltip p-3 rounded-xl text-xs space-y-1 text-slate-700 min-w-[200px] z-50">{children}</div>
);

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const p = payload[0].payload;
    return (
      <TooltipShell>
        <div className="font-bold text-slate-900 mb-2 border-b border-slate-200 pb-1">Batch Size: {Number(label).toLocaleString()} seqs</div>
        <div className="text-emerald-600 flex justify-between gap-4"><span>Compute Time:</span><span>{p.tCompute.toFixed(2)} ms</span></div>
        <div className="text-amber-600 flex justify-between gap-4"><span>Memory Time:</span><span>{p.tMemory.toFixed(2)} ms</span></div>
        <div className="text-slate-500 pl-4 flex justify-between gap-4"><span>Weight Fetch:</span><span>{p.tWeightFetch.toFixed(2)} ms</span></div>
        <div className="text-slate-500 pl-4 flex justify-between mb-1 pb-1 border-b border-slate-200 gap-4"><span>KV Fetch:</span><span>{p.tKvFetch.toFixed(2)} ms</span></div>
        <div className="text-slate-900 font-bold flex justify-between pt-1 gap-4"><span>Total Latency:</span><span>{p.latency.toFixed(2)} ms</span></div>
        <div className={cn('mt-2 font-bold', p.isMemoryBound ? 'text-amber-600' : 'text-emerald-600')}>{p.isMemoryBound ? 'Memory Bound' : 'Compute Bound'}</div>
      </TooltipShell>
    );
  }
  return null;
};

const CustomCostTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const p = payload[0].payload;
    return (
      <TooltipShell>
        <div className="font-bold text-slate-900 mb-2 border-b border-slate-200 pb-1">Batch Size: {Number(label).toLocaleString()} seqs</div>
        <div className="text-amber-600 flex justify-between gap-4"><span>Power Cost:</span><span>${p.costElec1M.toFixed(2)}</span></div>
        <div className="text-slate-600 flex justify-between gap-4 pb-1 border-b border-slate-200"><span>Hardware Cost:</span><span>${p.costHardware1M.toFixed(2)}</span></div>
        <div className="text-emerald-600 font-bold flex justify-between gap-4 pt-1"><span>Total TCO:</span><span>${p.totalCost1M.toFixed(2)}</span></div>
      </TooltipShell>
    );
  }
  return null;
};
