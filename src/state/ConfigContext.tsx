import React, {
  createContext, useContext, useState, useMemo, useEffect, useCallback, type ReactNode,
} from 'react';
import { HARDWARE_PROFILES } from '../lib/hardware';
import { MODEL_PROFILES } from '../lib/models';
import {
  physicalUnits, makeCurve, optimalBatch, drainTimeMs, contextCrossover,
  maxThroughput, currentIntensity, type RowInputs, type PhysicalUnits,
} from '../lib/roofline';
import { parseHash, buildHash } from '../lib/useRoute';

export const ECONOMICS_REGIONS = [
  { id: 'US Hyperscale', priceKwh: 0.07, pue: 1.10, desc: 'Texas/Iowa' },
  { id: 'US Commercial', priceKwh: 0.12, pue: 1.40, desc: 'Standard DC' },
  { id: 'Europe Avg', priceKwh: 0.22, pue: 1.35, desc: 'Germany/UK' },
  { id: 'Green / Hydro', priceKwh: 0.04, pue: 1.08, desc: 'Quebec/Iceland' },
  { id: 'Custom', priceKwh: 0.10, pue: 1.25, desc: 'Custom Settings' },
];

const DEFAULTS = {
  hw: 'H100 SXM5',
  flops: 989,
  membw: 3.35,
  memcap: 80,
  tdp: 700,
  price: 30000,
  region: 'US Hyperscale',
  kwh: 0.07,
  pue: 1.10,
  util: 65,
  years: 3,
  model: 'deepseek-v3',
  ptotal: 671,
  pactive: 37,
  bpp: 2,
  ctx: 16384,
  kvkb: 96,
  batch: 256,
};

/** Query-string keys, kept short so a shared URL stays readable. */
type NumKey = 'flops' | 'membw' | 'memcap' | 'tdp' | 'price' | 'kwh' | 'pue'
  | 'util' | 'years' | 'ptotal' | 'pactive' | 'bpp' | 'ctx' | 'kvkb' | 'batch';

function readNum(q: URLSearchParams, key: NumKey, fallback: number): number {
  const raw = q.get(key);
  if (raw === null) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

export interface ConfigValue {
  // hardware
  activeProfileId: string; setActiveProfileId: (id: string) => void;
  flopsTera: number; setFlopsTera: (n: number) => void;
  memBwTera: number; setMemBwTera: (n: number) => void;
  memCapGb: number; setMemCapGb: (n: number) => void;
  tdpWatts: number; setTdpWatts: (n: number) => void;
  hardwarePrice: number; setHardwarePrice: (n: number) => void;
  // economics
  regionId: string; setRegionId: (id: string) => void;
  priceKwh: number; setPriceKwh: (n: number) => void;
  pue: number; setPue: (n: number) => void;
  utilization: number; setUtilization: (n: number) => void;
  amortizationYears: number; setAmortizationYears: (n: number) => void;
  // model
  activeModelId: string; setActiveModelId: (id: string) => void;
  totalParamsB: number; setTotalParamsB: (n: number) => void;
  activeParamsB: number; setActiveParamsB: (n: number) => void;
  bytesPerParam: number; setBytesPerParam: (n: number) => void;
  contextLen: number; setContextLen: (n: number) => void;
  bytesPerTokenKb: number; setBytesPerTokenKb: (n: number) => void;
  // operating point
  currentBatchSize: number; setCurrentBatchSize: (n: number) => void;
  selectedProfiles: string[]; toggleProfile: (id: string) => void;
  // actions
  selectModel: (id: string) => void;
  selectProfile: (id: string) => void;
  selectRegion: (id: string) => void;
  // derived
  units: PhysicalUnits;
  rowInputs: RowInputs;
  chartData: ReturnType<typeof makeCurve>;
  optimalBatchSize: number;
  drainTime: number;
  contextLengthCrossover: number;
  maxTokensPerSec: number;
  currentStat: ReturnType<typeof makeCurve>[number];
  currentStatIntensity: number;
  isLiquid: boolean;
}

const Ctx = createContext<ConfigValue | null>(null);

/**
 * Global model + hardware + economics configuration.
 *
 * This used to be ~20 useState hooks private to Dashboard, which meant the
 * lessons could not see the reader's chosen hardware. Lifting it here is what
 * lets prose anywhere on the site quote live numbers, and what makes a
 * configuration shareable: it round-trips through the URL query string.
 */
export function ConfigProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(() => parseHash(window.location.hash).query, []);

  const [activeProfileId, setActiveProfileId] = useState(initial.get('hw') ?? DEFAULTS.hw);
  const [flopsTera, setFlopsTera] = useState(() => readNum(initial, 'flops', DEFAULTS.flops));
  const [memBwTera, setMemBwTera] = useState(() => readNum(initial, 'membw', DEFAULTS.membw));
  const [memCapGb, setMemCapGb] = useState(() => readNum(initial, 'memcap', DEFAULTS.memcap));
  const [tdpWatts, setTdpWatts] = useState(() => readNum(initial, 'tdp', DEFAULTS.tdp));
  const [hardwarePrice, setHardwarePrice] = useState(() => readNum(initial, 'price', DEFAULTS.price));

  const [regionId, setRegionId] = useState(initial.get('region') ?? DEFAULTS.region);
  const [priceKwh, setPriceKwh] = useState(() => readNum(initial, 'kwh', DEFAULTS.kwh));
  const [pue, setPue] = useState(() => readNum(initial, 'pue', DEFAULTS.pue));
  const [utilization, setUtilization] = useState(() => readNum(initial, 'util', DEFAULTS.util));
  const [amortizationYears, setAmortizationYears] = useState(() => readNum(initial, 'years', DEFAULTS.years));

  const [activeModelId, setActiveModelId] = useState(initial.get('model') ?? DEFAULTS.model);
  const [totalParamsB, setTotalParamsB] = useState(() => readNum(initial, 'ptotal', DEFAULTS.ptotal));
  const [activeParamsB, setActiveParamsB] = useState(() => readNum(initial, 'pactive', DEFAULTS.pactive));
  const [bytesPerParam, setBytesPerParam] = useState(() => readNum(initial, 'bpp', DEFAULTS.bpp));
  const [contextLen, setContextLen] = useState(() => readNum(initial, 'ctx', DEFAULTS.ctx));
  const [bytesPerTokenKb, setBytesPerTokenKb] = useState(() => readNum(initial, 'kvkb', DEFAULTS.kvkb));

  const [currentBatchSize, setCurrentBatchSize] = useState(() => readNum(initial, 'batch', DEFAULTS.batch));
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>(
    initial.get('cmp')?.split('~').filter(Boolean) ?? [
      'H100 SXM5', 'B200 (Blackwell)', 'Rubin GPU (R100)', 'TPU v8i (Inference)', 'Groq + Rubin Pipeline',
    ],
  );

  const selectModel = useCallback((id: string) => {
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
  }, []);

  const selectProfile = useCallback((id: string) => {
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
  }, []);

  const selectRegion = useCallback((id: string) => {
    setRegionId(id);
    if (id === 'Custom') return;
    const r = ECONOMICS_REGIONS.find((x) => x.id === id);
    if (r) {
      setPriceKwh(r.priceKwh);
      setPue(r.pue);
    }
  }, []);

  const toggleProfile = useCallback((id: string) => {
    setSelectedProfiles((s) => (s.includes(id) ? s.filter((p) => p !== id) : [...s, id]));
  }, []);

  const units = physicalUnits({
    flopsTera, memBwTera, memCapGb, totalParamsB, activeParamsB, bytesPerParam, bytesPerTokenKb,
  });

  const isLiquid =
    activeProfileId.toLowerCase().includes('rubin') || activeProfileId.toLowerCase().includes('liquid');

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

  const chartData = useMemo(() => makeCurve(units, rowInputs), [units, rowInputs]);

  const currentStat = chartData.reduce((prev, curr) =>
    Math.abs(curr.batchSize - currentBatchSize) < Math.abs(prev.batchSize - currentBatchSize) ? curr : prev,
  );

  // Mirror the configuration into the URL so a setup can be shared, without
  // adding a history entry per keystroke.
  useEffect(() => {
    const id = window.setTimeout(() => {
      const { section, query } = parseHash(window.location.hash);
      const next = new URLSearchParams(query);
      const put = (k: string, v: string | number, d: string | number) => {
        if (String(v) === String(d)) next.delete(k);
        else next.set(k, String(v));
      };
      put('hw', activeProfileId, DEFAULTS.hw);
      put('flops', flopsTera, DEFAULTS.flops);
      put('membw', memBwTera, DEFAULTS.membw);
      put('memcap', memCapGb, DEFAULTS.memcap);
      put('tdp', tdpWatts, DEFAULTS.tdp);
      put('price', hardwarePrice, DEFAULTS.price);
      put('region', regionId, DEFAULTS.region);
      put('kwh', priceKwh, DEFAULTS.kwh);
      put('pue', pue, DEFAULTS.pue);
      put('util', utilization, DEFAULTS.util);
      put('years', amortizationYears, DEFAULTS.years);
      put('model', activeModelId, DEFAULTS.model);
      put('ptotal', totalParamsB, DEFAULTS.ptotal);
      put('pactive', activeParamsB, DEFAULTS.pactive);
      put('bpp', bytesPerParam, DEFAULTS.bpp);
      put('ctx', contextLen, DEFAULTS.ctx);
      put('kvkb', bytesPerTokenKb, DEFAULTS.kvkb);
      put('batch', currentBatchSize, DEFAULTS.batch);
      window.history.replaceState(null, '', buildHash(section, next));
    }, 250);
    return () => window.clearTimeout(id);
  }, [
    activeProfileId, flopsTera, memBwTera, memCapGb, tdpWatts, hardwarePrice,
    regionId, priceKwh, pue, utilization, amortizationYears,
    activeModelId, totalParamsB, activeParamsB, bytesPerParam, contextLen,
    bytesPerTokenKb, currentBatchSize,
  ]);

  const value: ConfigValue = {
    activeProfileId, setActiveProfileId,
    flopsTera, setFlopsTera,
    memBwTera, setMemBwTera,
    memCapGb, setMemCapGb,
    tdpWatts, setTdpWatts,
    hardwarePrice, setHardwarePrice,
    regionId, setRegionId,
    priceKwh, setPriceKwh,
    pue, setPue,
    utilization, setUtilization,
    amortizationYears, setAmortizationYears,
    activeModelId, setActiveModelId,
    totalParamsB, setTotalParamsB,
    activeParamsB, setActiveParamsB,
    bytesPerParam, setBytesPerParam,
    contextLen, setContextLen,
    bytesPerTokenKb, setBytesPerTokenKb,
    currentBatchSize, setCurrentBatchSize,
    selectedProfiles, toggleProfile,
    selectModel, selectProfile, selectRegion,
    units, rowInputs, chartData,
    optimalBatchSize: optimalBatch(units),
    drainTime: drainTimeMs(units),
    contextLengthCrossover: contextCrossover(units),
    maxTokensPerSec: maxThroughput(units),
    currentStat,
    currentStatIntensity: currentIntensity(currentBatchSize, units, rowInputs),
    isLiquid,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useConfig(): ConfigValue {
  const v = useContext(Ctx);
  if (!v) throw new Error('useConfig must be used inside <ConfigProvider>');
  return v;
}
