import type { KV } from '../../lib/microgpt';
import { CHART as C } from '../../lib/theme';

export function fmtNum(v: number, digits = 0) {
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: digits });
}

export function cloneKV(kv: KV): KV {
  return {
    keys: kv.keys.map((layer) => layer.map((vec) => [...vec])),
    values: kv.values.map((layer) => layer.map((vec) => [...vec])),
  };
}

export function VecStack({ k, v, colorK, colorV }: { k: number[]; v: number[]; colorK: string; colorV: string }) {
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

/**
 * Live KV-cache shape readout: shows the array dimensions and how many values
 * are stored, growing through prefill (prompt positions) then decode (generated).
 */
export function KvSizeGauge({ positions, promptPositions, maxPositions, nEmbd, nLayer, nHead }: {
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
    <div>
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
        <div className="h-full" style={{ width: `${pctGen}%`, background: '#99e0ff' }} />
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1.5 text-[10px] text-slate-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-sm align-middle" style={{ background: C.accent }} />
          prefill {Math.min(positions, promptPositions)} pos · {fmtNum(prefillValues)} values
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-sm align-middle" style={{ background: '#99e0ff' }} />
          generated {genPositions} pos · {fmtNum(genValues)} values
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-sm bg-slate-200 align-middle" />
          max {maxPositions} pos · {fmtNum(maxValues)} values
        </span>
      </div>
    </div>
  );
}
