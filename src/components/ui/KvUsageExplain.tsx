import React from 'react';
import { cn } from '../../lib/utils';

const COLOR_K = '#5b7cfa';
const COLOR_V = '#22c48b';

// Round an element of a K/V vector for the mini display.
function fmtCell(v: number): string {
  return v.toFixed(2);
}

// Turn a vector into a row of cells, colour-coded like the Lab's VecStack.
function VecRow({ label, values, color }: { label: string; values: number[]; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[9px] font-black w-3" style={{ color }}>{label}</span>
      <div className="flex gap-1">
        {values.map((v, i) => (
          <span key={i} className="rounded px-1 py-0.5 text-[9px] font-mono"
            style={{ backgroundColor: `${color}16`, color }}>
            {fmtCell(v)}
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * Explains how each cached K and V value is actually used inside the attention head.
 * The formula mirrors microgpt.ts `forward()` and reference/microgpt.py `gpt()`:
 *   score(k) = q · k / √head_dim   -- K is the MATCH / gate
 *   out[j]   = Σ_t weights[t]·v_t[j] -- V is the CONTENT / carry
 */
export default function KvUsageExplain() {
  // Toy numbers mirroring a real head's 4-dim projection (deterministic, for display only).
  const q = [0.24, -0.61, 0.80, 0.13];
  const k0 = [0.31, -0.42, 0.55, 0.09];   // cached K at position 0 (older token)
  const k1 = [0.17, -0.28, 0.63, 0.22];   // cached K at position 1 (newest token)
  const v0 = [0.44, -0.39, 0.52, 0.08];   // cached V at position 0
  const v1 = [0.27, -0.18, 0.71, 0.35];   // cached V at position 1

  const score0 = q.reduce((s, vi, j) => s + vi * k0[j], 0);
  const score1 = q.reduce((s, vi, j) => s + vi * k1[j], 0);
  const w0 = Math.exp(score0) / (Math.exp(score0) + Math.exp(score1));
  const w1 = 1 - w0;

  // Output = weighted sum of cached V.
  const out = v1.map((_, j) => w0 * v0[j] + w1 * v1[j]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div className="glass rounded-xl p-5">
        <div className="flex items-center gap-2 mb-2">
          <span className="inline-block w-4 h-4 rounded-sm" style={{ background: COLOR_K }} />
          <h4 className="font-bold text-slate-800">K — the <em>match</em></h4>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed mb-3">
          Every cached key <code className="bg-slate-100 px-1 rounded">k</code> is dotted against the new token&rsquo;s
          query: <code>score = Σ q·k / √head_dim</code>. Passing those raw scores through a
          softmax turns them into <strong>attention weights</strong>. So K decides <em>how much</em> each earlier
          position gets to influence the output — it is the <strong>gate</strong> that selects where to look.
        </p>
        <div className="font-mono text-[10px] text-slate-500 space-y-1.5">
          <div className="flex justify-between"><span>score(k₀)</span><span>{score0.toFixed(2)} → w = {w0.toFixed(2)}</span></div>
          <div className="flex justify-between"><span>score(k₁)</span><span>{score1.toFixed(2)} → w = {w1.toFixed(2)}</span></div>
        </div>
      </div>

      <div className="glass rounded-xl p-5">
        <div className="flex items-center gap-2 mb-2">
          <span className="inline-block w-4 h-4 rounded-sm" style={{ background: COLOR_V }} />
          <h4 className="font-bold text-slate-800">V — the <em>content</em></h4>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed mb-3">
          The attention weights are applied to the cached <strong>values</strong>, not the keys:
          <code>out[j] = Σ<sub>t</sub> weights[t] · v<sub>t</sub>[j]</code>. Each cached <code>v</code> is multiplied by
          its softmax weight and summed — a weighted blend. So V is the <em>content</em> that actually flows forward
          into the next token&rsquo;s hidden state.
        </p>
        <div className="space-y-1.5 mb-3">
          <div className="text-[10px] text-slate-400 font-mono">weights × cached V → output</div>
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-500">
            <span className="text-[9px] font-black w-3 text-white">{w0.toFixed(2)}</span>
            <span className="text-slate-400">→</span>
            <VecRow label="V₀" values={v0} color={COLOR_V} />
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-500">
            <span className="text-[9px] font-black w-3 text-white">{w1.toFixed(2)}</span>
            <span className="text-slate-400">→</span>
            <VecRow label="V₁" values={v1} color={COLOR_V} />
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-700 mt-1.5">
            <span className="text-[9px] font-black w-3 text-slate-700">out</span>
            <span className="text-slate-400">=</span>
            <VecRow label="Σ" values={out} color={COLOR_V} />
          </div>
        </div>
      </div>

      <div className="glass rounded-xl p-5">
        <h4 className="font-bold text-slate-800 mb-2">The two lines behind the cache</h4>
        <p className="text-sm text-slate-600 leading-relaxed mb-3">
          This is precisely the microgpt forward pass: the cache stores <code>k</code> and <code>v</code> per
          position so a later step can <em>score against every stored K</em> and <em>blend every stored V</em> without
          recomputing the prefix.
        </p>
        <pre className="text-[11px] bg-slate-900 text-slate-100 rounded-lg p-3 overflow-x-auto font-mono leading-relaxed">
{`score = softmax(q · kᵢ / √d)   # K selects (match)
out   = Σᵢ score[i] · vᵢ        # V supplies (content)`}
        </pre>
        <p className="text-[10px] text-slate-400 mt-3">
          See <code className="bg-slate-100 px-1 rounded">microgpt.ts</code> forward() and{" "}
          <code className="bg-slate-100 px-1 rounded">reference/microgpt.py</code> (lines 126-130).
        </p>
      </div>
    </div>
  );
}