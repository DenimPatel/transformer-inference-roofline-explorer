import React, { useState, useMemo } from 'react';
import {
  ComposedChart, Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  Scatter, AreaChart, Area, Bar, Cell
} from 'recharts';
import { Info, Calculator, Cpu, Network, Zap, Activity, BookOpen } from 'lucide-react';
import { cn } from '../lib/utils';
import { HARDWARE_PROFILES } from '../lib/hardware';
import ConceptTag from './ui/ConceptTag';

export default function DeepDiveTab({ hardwareProfileId }: { hardwareProfileId: string }) {
  const hw = HARDWARE_PROFILES.find(h => h.id === hardwareProfileId) || HARDWARE_PROFILES[0];
  const peakFlops = hw.tflops * 1e12;
  const peakBw = hw.memBw * 1e12;
  const hardwareIntensity = peakFlops / peakBw;

  return (
    <div className="space-y-12 pb-12 max-w-5xl mx-auto mt-8">
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <Calculator className="w-6 h-6 mr-3 text-blue-500" />
          1. Formalized Mathematical Framework
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          <ConceptTag id="overlap" /><ConceptTag id="roofline" /><ConceptTag id="arithmetic-intensity" />
        </div>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            The core framework for determining the runtime of an algorithm on hardware is the interaction
            between <strong>computation time</strong> (<i>T<sub>math</sub></i>) and
            <strong>communication time</strong> (<i>T<sub>comms</sub></i>).
          </p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Time Equations:</strong> <i>T<sub>math</sub> = (Computation FLOPs) / (Accelerator FLOPs/s)</i>, and <i>T<sub>comms</sub> = (Communication Bytes) / (Bandwidth Bytes/s)</i>.</li>
            <li><strong>Lower vs Upper Bound:</strong> A <strong>lower bound</strong> (<i>T<sub>lower</sub> = max(T<sub>math</sub>, T<sub>comms</sub>)</i>) assumes perfect overlap of math and communication. An <strong>upper bound</strong> (<i>T<sub>upper</sub> = T<sub>math</sub> + T<sub>comms</sub></i>) assumes no overlap. Since <i>T<sub>math</sub> + T<sub>comms</sub> ≤ 2·max(...)</i>, the two bounds differ by at most 2× — which is exactly the room available for overlap optimizations (like collective matmuls).</li>
            <li><strong>Arithmetic (Operational) Intensity:</strong> <i>I = (Total FLOPs) / (Communication Bytes)</i>. An algorithm is <strong>compute-bound</strong> when its intensity exceeds the hardware's peak intensity (<i>I<sub>hw</sub> = peak FLOPs/s ÷ bandwidth</i>), and <strong>memory/communication-bound</strong> otherwise.</li>
          </ul>
        </div>

        {/* Overlap visual: lower vs upper bound */}
        <div className="glass rounded-xl p-5 mb-6">
          <h3 className="font-bold text-slate-800 mb-3 flex items-center">
            <Activity className="w-4 h-4 mr-2 text-blue-500" /> How Much Can Overlap Help?
          </h3>
          <OverlapBoundsChart />
          <p className="text-sm text-slate-500 mt-3">
            When <i>T<sub>math</sub> = T<sub>comms</sub></i> the gap between the no-overlap upper bound and
            the perfect-overlap lower bound is largest, so this is where scheduling (e.g. overlapping one
            matmul's communication with the next block's compute) buys you the most.
          </p>
        </div>

        <div className="glass rounded-xl p-5">
          <h3 className="font-bold text-slate-800 mb-4 flex items-center">
            Roofline Plot: {hw.id}
          </h3>
          <p className="text-sm text-slate-500 mb-6">
            Log–log throughput (FLOPs/s) vs arithmetic intensity. The "ridge point" is the hardware
            intensity threshold (≈{hardwareIntensity.toFixed(1)} FLOPs/Byte). Left of the ridge you are
            bandwidth-bound; right of it you saturate peak FLOPs.
          </p>
          <div className="h-80 w-full">
            <RooflineChart peakFlops={peakFlops} peakBw={peakBw} ridge={hardwareIntensity} showOps />
          </div>
        </div>
      </section>

      {/* 2. Arithmetic Intensity & The Ridge Point (NEW) */}
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <Info className="w-6 h-6 mr-3 text-cyan-500" />
          2. Arithmetic Intensity &amp; The Ridge Point
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          <ConceptTag id="arithmetic-intensity" /><ConceptTag id="ridge-point" /><ConceptTag id="critical-batch" />
        </div>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            The single most useful number in roofline analysis is the <strong>arithmetic intensity</strong>,
            <i>I = FLOPs ÷ bytes</i>. It measures how much math you get out of every byte you drag through
            the memory system. The crossover where an operation stops being limited by bandwidth and starts
            saturating the FLOPs/s is the hardware's <strong>ridge point</strong>, <i>I<sub>hw</sub> = peak
            FLOPs/s ÷ bandwidth</i>.
          </p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>If <i>I &lt; I<sub>hw</sub></i>:</strong> the operation is <em>bandwidth-bound</em> — bytes limit you before FLOPs do, and you leave arithmetic throughput on the table.</li>
            <li><strong>If <i>I &gt; I<sub>hw</sub></i>:</strong> the operation is <em>compute-bound</em> — you use essentially all the accelerator's FLOPs/s.</li>
            <li><strong>For a matmul</strong> (see Section 3) the intensity is ≈ the per-replica <strong>token batch size B</strong>. So the familiar rule falls out: <em>a bf16 matmul is compute-bound iff <i>B &gt; I<sub>hw</sub></i></em> (≈240 tokens on most TPUs, ≈295 on an H100).</li>
          </ul>
        </div>
        <ArithmeticIntensitySection hardwareIntensity={hardwareIntensity} />
      </section>

      {/* 3. Detailed Matrix Multiplication (Matmul) Math */}
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <Cpu className="w-6 h-6 mr-3 text-emerald-500" />
          3. Detailed Matrix Multiplication (Matmul) Math
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          <ConceptTag id="matmul-intensity" /><ConceptTag id="critical-batch" />
        </div>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>Matrix multiplication is the fundamental operation of deep learning, and its arithmetic intensity decides how efficiently a model uses hardware.</p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Variables:</strong> For <i>X[B, D] × Y[D, F] → Z[B, F]</i>, the total FLOPs are <i>2BDF</i>.</li>
            <li><strong>Memory Movement:</strong> Load the two inputs (<i>2BD + 2DF</i>) and write the result (<i>2BF</i>).</li>
            <li><strong>Intensity:</strong> <i>I = 2BDF / (2BD + 2DF + 2BF)</i>. When <i>B</i> is small relative to <i>D, F</i> (typical: <i>B &lt; 1024</i> tokens, <i>D, F &gt; 8000</i>), this simplifies to <i>I ≈ B</i>.</li>
            <li><strong>Critical Threshold:</strong> Compute-bound iff <i>B &gt; I<sub>hw</sub></i> — ~240 tokens for TPUs, ~295 for an H100 (per-replica token batch, not sequences).</li>
            <li><strong>Tiling caveat:</strong> Large matmuls are decomposed into tiles that fit high-bandwidth on-chip memory (VMEM/SMEM/TMEM) and are re-loaded multiple times, so bytes are <em>not</em> exactly <i>O(N²)</i>. For tile sizes <i>bm × bn</i> the effective intensity becomes ≈ <i>bm·bn/(bm+bn)</i> — so tuning tiles also tunes intensity.</li>
          </ul>
        </div>
        <MatmulInteractiveSection hardwareIntensity={hardwareIntensity} />
      </section>

      {/* 4. Prefill vs Generation (NEW) */}
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <Activity className="w-6 h-6 mr-3 text-sky-500" />
          4. Prefill vs Generation: Why Inference Flips the Roofline
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          <ConceptTag id="prefill" /><ConceptTag id="generation" /><ConceptTag id="kv-cache" />
        </div>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>
            Transformer inference is really <em>two different workloads</em>. Understanding which side of the
            ridge each sits on explains why the Interactive Lab's decode (generation) model looks the way it
            does.
          </p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Prefill</strong> processes a long prompt of <i>T</i> tokens at once. Linear ops reuse their weights across <i>B·T</i> tokens, and attention intensity is <i>≈ T/2</i>. So prefill is essentially <strong>always compute-bound</strong> (for real prompts) — maximizing MFU directly improves TTFT.</li>
            <li><strong>Generation</strong> runs one token at a time (<i>T = 1</i>). Weights are not amortized; each token streams the full parameter set plus its own KV cache. Attention intensity <i>ST/(S+T) ≈ 1</i> (constant). So generation is <strong>basically always memory-bandwidth-bound</strong>, and batching many requests together is what pushes per-token intensity back toward the ridge.</li>
            <li><strong>Step-time bound:</strong> for small generate batches, <i>T<sub>step</sub> ≥ (B · KV-cache + parameters) / bandwidth</i>. This is the theoretical minimum you should aim for — and the exact equation the Interactive Lab simulates.</li>
          </ul>
        </div>
        <PrefillGenerationSection hw={hw} />
      </section>

      {/* 5. Inter-Chip Network Rooflines */}
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <Network className="w-6 h-6 mr-3 text-purple-500" />
          5. Inter-Chip Network Rooflines
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          <ConceptTag id="network-roofline" /><ConceptTag id="model-parallelism" />
        </div>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>When models are distributed across multiple accelerators, the communication bottleneck shifts from on-chip HBM bandwidth to inter-chip network bandwidth.</p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Sharding Math:</strong> For two chips splitting the contracting <i>D</i> dimension, each chip does <i>BDF</i> FLOPs and must exchange <i>2BF</i> bytes of partial sums.</li>
            <li><strong>Change in Bottlenecks:</strong> Unlike single-chip operations (limited by <i>B</i>), inter-chip rooflines depend on the <strong>model dimension D</strong>: <i>BDF/(2BF) = D/2</i>.</li>
            <li><strong>Threshold Example:</strong> With a network bandwidth of 4.5e10 bytes/s and 1.97e14 FLOPs/s per chip, the op becomes compute-bound when <i>D/2 &gt; 4377</i>, i.e. <i>D &gt; 8755</i>, regardless of batch size.</li>
          </ul>
        </div>
        <NetworkRooflineInteractiveSection hw={hw} />
      </section>

      {/* 6. Quantization and Mixed Precision Analysis */}
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <Zap className="w-6 h-6 mr-3 text-amber-500" />
          6. Quantization and Mixed Precision Analysis
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          <ConceptTag id="quantization" /><ConceptTag id="critical-batch" />
        </div>
        <div className="prose prose-slate max-w-none text-slate-600 mb-6 space-y-4">
          <p>Quantization changes arithmetic intensity by changing the bytes-per-parameter — and precision changes how much effective FLOPs/s the accelerator can do. The two interact:</p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>bf16 weights + bf16 math:</strong> critical batch <i>B<sub>crit</sub> ≈ I<sub>hw</sub></i> (~240 on TPU v5e).</li>
            <li><strong>int8/fp8 weights, bf16 compute (mixed):</strong> weight bytes halve while FLOPs stay bf16, so <i>B<sub>crit</sub> ≈ I<sub>hw</sub>/2 ≈ 120</i>. This is the "easy win": lower <i>B<sub>crit</sub></i> means you become compute-bound sooner.</li>
            <li><strong>int8 weights + int8 math:</strong> FLOPs/s roughly doubles, so <i>B<sub>crit</sub></i> returns to ~240 — the two effects cancel.</li>
            <li><strong>Batch-specific weights (e.g. per-token MoE paths):</strong> comms explode to <i>BD + BDF + BF</i>; since <i>BDF</i> dominates, intensity collapses to a constant <i>≈ 2</i>, so the op is <strong>always communication-bound</strong>.</li>
          </ul>
        </div>
        <QuantizationInteractiveSection hardwareIntensity={hardwareIntensity} />
      </section>

      {/* 7. Low-Level "Tiling" and Vector Math */}
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <Calculator className="w-6 h-6 mr-3 text-slate-600" />
          7. Low-Level "Tiling" and Vector Math
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          <ConceptTag id="matmul-intensity" /><ConceptTag id="arithmetic-intensity" />
        </div>
        <div className="prose prose-slate max-w-none text-slate-600 space-y-4">
          <p>The matrix multiply unit (MXU / Tensor Core) and the vector processing unit (VPU / SIMT core) have very different rooflines, so you must compute them separately and take the max.</p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Tiling:</strong> Large matmuls are broken into tiles that fit on-chip (VMEM/SMEM/TMEM) and re-loaded from HBM repeatedly — so actual bytes exceed the naïve <i>O(N²)</i> estimate and intensity drops to ≈ <i>bm·bn/(bm+bn)</i>.</li>
            <li><strong>Vector Operations (Dot Products):</strong> A dot product loads <i>2N</i> input bytes for ≈ <i>2N</i> FLOPs → intensity <i>1/2</i>. Low <em>and constant</em>, so it is essentially always bandwidth-bound.</li>
            <li><strong>Hardware Mapping:</strong> Matmuls run on the MXU (high ridge); softmax/relu run on the VPU (lower ridge). To bound a model's time, build the roofline for <em>both</em> units and take the maximum at each intensity.</li>
          </ul>
        </div>
      </section>

      {/* 8. Worked Problems (NEW) */}
      <section className="glass-card p-6">
        <h2 className="text-2xl font-bold flex items-center text-slate-900 mb-6">
          <BookOpen className="w-6 h-6 mr-3 text-rose-500" />
          8. Worked Problems (from the scaling-book, Part 1)
        </h2>
        <WorkedProblemsSection hw={hw} peakFlops={peakFlops} peakBw={peakBw} hardwareIntensity={hardwareIntensity} />
      </section>
    </div>
  );
}

// -------------------------------------------------------------
// 1. Overlap: lower vs upper bound
// -------------------------------------------------------------
function OverlapBoundsChart() {
  const data = useMemo(() => {
    // Score is math/comms ratio from 0.1 (comms-dominated) to 10 (compute-dominated)
    const rows = [];
    for (let r = 0.15; r <= 6.5; r += 0.2) {
      // T_math + T_comms vs 2 * max  -> normalize
      const lower = 2 * Math.max(r, 1); // 2*max(normalized) ; math=r, comms=1
      const upper = r + 1;              // no overlap = sum
      const overlapWin = ((upper - lower * 0.5) > 0 ? (upper - lower * 0.5) : 0);
      rows.push({
        ratio: r,
        noOverlap: r + 1,
        perfectOverlap: Math.max(r, 1),
        // percentage saved by overlapping
        saved: Math.round(((r + 1 - Math.max(r, 1)) / (r + 1)) * 100)
      });
    }
    return rows;
  }, []);

  return (
    <div>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <defs>
              <linearGradient id="overlapFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="upperFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
            <XAxis
              dataKey="ratio"
              type="number"
              scale="log"
              domain={[0.1, 10]}
              tickFormatter={(v) => `${Number(v).toFixed(1)}`}
              label={{ value: 'T_math / T_comms ratio', position: 'bottom', fontSize: 11 }}
            />
            <YAxis label={{ value: 'Relative time', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip formatter={(v: any, name: any) => [Number(v).toFixed(2), name]} labelFormatter={(v: any) => `ratio ${Number(v).toFixed(2)}`} />
            <Area type="monotone" dataKey="noOverlap" name="Upper bound (no overlap)" stroke="#f43f5e" strokeWidth={2} fill="url(#upperFill)" />
            <Area type="monotone" dataKey="perfectOverlap" name="Lower bound (perfect overlap)" stroke="#2563eb" strokeWidth={2} fill="url(#overlapFill)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
        {data.filter(d => [0.5, 1, 2, 5].includes(Math.round(d.ratio))).map(d => (
          <div key={d.ratio} className="glass rounded-md p-2 text-center">
            <div className="text-slate-400">ratio {d.ratio}</div>
            <div className="font-mono font-bold text-emerald-600">{d.saved}% saved</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Section 1 & shared: proper teaching Roofline
// -------------------------------------------------------------
function RooflineChart({ peakFlops, peakBw, ridge, showOps = false }: { peakFlops: number, peakBw: number, ridge: number, showOps?: boolean }) {
  const data = useMemo(() => {
    const pts = [];
    const minI = 0.05;
    const maxI = 200000;
    for (let i = Math.log10(minI); i <= Math.log10(maxI); i += 0.05) {
      const intensity = Math.pow(10, i);
      const achievable = Math.min(peakBw * intensity, peakFlops);
      pts.push({ intensity, achievable });
    }
    return pts;
  }, [peakFlops, peakBw]);

  // Per-operation markers (intensity on the theoretical achieved line)
  const ops = useMemo(() => {
    if (!showOps) return [];
    const mk = (name: string, intensity: number, color: string) => ({
      name,
      intensity,
      achieved: Math.min(peakBw * intensity, peakFlops),
      color,
      bound: intensity >= ridge ? 'Compute-bound' : 'Bandwidth-bound'
    });
    return [
      mk('Dot product', 0.5, '#ef4444'),
      mk('Attention (generate, ~1)', 1, '#f59e0b'),
      mk('Matmul, B=64', 64, '#3b82f6'),
      mk('Attention (prefill, T=1024/2)', 512, '#10b981'),
      mk('Matmul, B=1024', 1024, '#8b5cf6'),
    ];
  }, [showOps, peakBw, peakFlops, ridge]);

  return (
    <ComposedChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
      <CartesianGrid strokeDasharray="3 3" opacity={0.4} />
      <XAxis
        dataKey="intensity"
        scale="log"
        domain={['dataMin', 'dataMax']}
        type="number"
        tickFormatter={(v) => Number(v) < 1 ? Number(v).toFixed(1) : Number(v).toFixed(0)}
        label={{ value: 'Arithmetic Intensity (FLOPs / Byte) — log', position: 'bottom', fontSize: 11 }}
      />
      <YAxis
        scale="log"
        domain={['dataMin', 'dataMax']}
        type="number"
        tickFormatter={(v) => (v / 1e12).toFixed(0)}
        label={{ value: 'Throughput (TFLOPs / s) — log', angle: -90, position: 'insideLeft', fontSize: 11 }}
      />
      <Tooltip
        labelFormatter={(v: any) => `Intensity: ${Number(v).toFixed(1)} FLOPs/B`}
        formatter={(v: any) => [`${(Number(v) / 1e12).toFixed(2)} TFLOP/s`]}
      />
      {showOps && <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />}
      {/* the roof */}
      <Line type="monotone" dataKey="achievable" name="Achievable performance (roof)" stroke="#0ea5e9" strokeWidth={3} dot={false} />
      {/* ridge line */}
      <ReferenceLine x={ridge} stroke="#f43f5e" strokeDasharray="4 4" label={{ position: 'top', value: `ridge ≈ ${ridge.toFixed(0)}`, fill: '#f43f5e', fontSize: 10 }} />
      {/* per-op markers */}
      {showOps && (
        <Scatter data={ops} dataKey="achieved" name="Operations"
          fill="#0ea5e9" shape="circle" legendType="none" />
      )}
    </ComposedChart>
  );
}

// -------------------------------------------------------------
// Section 2: Arithmetic intensity interactive
// -------------------------------------------------------------
function ArithmeticIntensitySection({ hardwareIntensity }: { hardwareIntensity: number }) {
  const [B, setB] = useState(64);
  const [D, setD] = useState(4096);
  const [F, setF] = useState(4096);

  // exact intensity of matmul
  const intensity = (2 * B * D * F) / (2 * B * D + 2 * D * F + 2 * B * F);
  const approx = B; // B << D, F simplification
  const isComputeBound = intensity > hardwareIntensity;
  const criticalB = hardwareIntensity;

  const data = useMemo(() => {
    const pts = [];
    for (let batch = 1; batch <= 1024; batch *= 1.15) {
      const b = Math.round(batch);
      const i = (2 * b * D * F) / (2 * b * D + 2 * D * F + 2 * b * F);
      pts.push({ batch: b, intensity: i, approx: b });
    }
    return pts;
  }, [D, F]);

  return (
    <div className="glass rounded-xl p-5">
      <h3 className="font-bold text-slate-800 mb-4">Interactive: Where Does This Matmul Sit?</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Per-replica token batch (B)</label>
              <span className="font-mono text-slate-900">{B}</span>
            </div>
            <input type="range" min={1} max={1024} value={B} onChange={e => setB(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Dimension D / F</label>
              <span className="font-mono text-slate-900">{D} / {F}</span>
            </div>
            <input type="range" min={1024} max={16384} step={256} value={D} onChange={e => { setD(Number(e.target.value)); setF(Number(e.target.value)); }} className="glass-slider w-full glass-slider w-full" />
          </div>

          <div className="glass rounded-lg p-4 mt-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Exact intensity</span>
              <span className="font-mono font-bold text-slate-900">{intensity.toFixed(1)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">≈ B simplification</span>
              <span className="font-mono text-slate-600">{approx}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Hardware ridge</span>
              <span className="font-mono text-slate-600">{hardwareIntensity.toFixed(1)}</span>
            </div>
            <div className={cn("text-xs font-bold mt-2", isComputeBound ? "text-emerald-500" : "text-amber-500")}>
              {isComputeBound ? "✅ Compute-bound" : "⚠️ Bandwidth-bound"}
            </div>
            <div className="text-[10px] text-slate-400">
              This hardware's critical batch size: <span className="font-mono">{criticalB.toFixed(0)}</span> tokens. Cross it and you stop wasting FLOPs.
            </div>
          </div>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="batch" type="number" scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => `${v}`} label={{ value: 'Token batch (log)', position: 'bottom', fontSize: 10 }} />
              <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip />
              <Line type="monotone" dataKey="approx" name="≈ B" stroke="#94a3b8" strokeWidth={2} strokeDasharray="4 4" dot={false} />
              <Line type="monotone" dataKey="intensity" name="Exact" stroke="#06b6d4" strokeWidth={3} dot={false} />
              <ReferenceLine y={hardwareIntensity} stroke="#f43f5e" strokeDasharray="3 3" label={{ position: 'top', value: 'Ridge', fill: '#f43f5e', fontSize: 10 }} />
              <ReferenceLine x={B} stroke="#475569" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Section 3: matmul interactive
// -------------------------------------------------------------
function MatmulInteractiveSection({ hardwareIntensity }: { hardwareIntensity: number }) {
  const [B, setB] = useState(128);
  const [D, setD] = useState(4096);
  const [F, setF] = useState(4096);

  const data = useMemo(() => {
    const pts = [];
    for (let batch = 1; batch <= 1024; batch *= 1.2) {
      const b = Math.round(batch);
      const intensity = (2 * b * D * F) / (2 * b * D + 2 * D * F + 2 * b * F);
      pts.push({ batch: b, intensity });
    }
    return pts;
  }, [D, F]);

  const currentIntensity = (2 * B * D * F) / (2 * B * D + 2 * D * F + 2 * B * F);

  return (
    <div className="glass rounded-xl p-5">
      <h3 className="font-bold text-slate-800 mb-4">Interactive Matmul Scaling</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Batch Size (B)</label>
              <span className="font-mono text-slate-900">{B}</span>
            </div>
            <input type="range" min={1} max={1024} value={B} onChange={e => setB(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Dimension D</label>
              <span className="font-mono text-slate-900">{D}</span>
            </div>
            <input type="range" min={128} max={16384} step={128} value={D} onChange={e => setD(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Dimension F</label>
              <span className="font-mono text-slate-900">{F}</span>
            </div>
            <input type="range" min={128} max={16384} step={128} value={F} onChange={e => setF(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>

          <div className="glass rounded-lg p-4 mt-4">
            <div className="text-sm text-slate-500 mb-1">Current Arithmetic Intensity</div>
            <div className="text-2xl font-mono font-bold text-slate-900">{currentIntensity.toFixed(1)} <span className="text-sm font-normal text-slate-500">FLOPs/B</span></div>
            <div className={cn("text-xs font-bold mt-2", currentIntensity > hardwareIntensity ? "text-emerald-500" : "text-amber-500")}>
              {currentIntensity > hardwareIntensity ? "✅ Compute Bound" : "⚠️ Memory Bound"}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              (Requires &gt; {hardwareIntensity.toFixed(1)} FLOPs/B — critical batch ≈ {hardwareIntensity.toFixed(0)})
            </div>
          </div>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="batch" label={{ value: 'Batch Size (B)', position: 'bottom', fontSize: 10 }} />
              <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip />
              <ReferenceLine y={hardwareIntensity} stroke="#f43f5e" strokeDasharray="3 3" label={{ position: 'top', value: 'Hardware Limit', fill: '#f43f5e', fontSize: 10 }} />
              <Line type="monotone" dataKey="intensity" stroke="#2563eb" strokeWidth={3} dot={false} />
              <ReferenceLine x={B} stroke="#475569" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Section 4: Prefill vs Generation
// -------------------------------------------------------------
function PrefillGenerationSection() {
  const [batch, setBatch] = useState(32);
  const [context, setContext] = useState(4096);
  const [kvKb, setKvKb] = useState(128);

  const kvBytes = context * kvKb * 1024;
  // per-replica token batch given batch sequences (for prefill, tokens = batch*context)
  const prefillTokens = batch * context;

  // intensities
  const prefillAttention = context / 2;        // T/2
  const generateAttention = 1;                 // ST/(S+T) ~ 1
  const matmulPrefill = prefillTokens;

  const data = [
    { name: `Prefill attention (T=${context})`, intensity: prefillAttention, class: 'compute', desc: `≈ T/2 = ${(prefillAttention).toFixed(0)} FLOPs/B — way above the ridge, compute-bound.` },
    { name: `Prefill matmul sum`, intensity: matmulPrefill, class: 'compute', desc: `linear ops reuse weights over ~${prefillTokens} tokens — compute-bound.` },
    { name: `Generate attention (T=1)`, intensity: generateAttention, class: 'memory', desc: `≈ ST/(S+T) ≈ 1 — constant & far below ridge: always memory-bound.` },
    { name: `Generate with batch ${batch}`, intensity: (batch * context) / (batch * context + context), class: 'middle', desc: `Batching joins many KV reads so weights amortize a little; still usually below the ridge.` },
  ];

  const kvTotalGb = (kvBytes * batch) / 1e9;

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
        <div className="space-y-4">
          <p className="text-sm text-slate-500">How generation batching + KV-cache size move the attention roofline:</p>
          <div>
            <div className="flex justify-between text-sm mb-1"><label className="font-medium text-slate-700">Concurrent requests (batch)</label><span className="font-mono text-slate-900">{batch}</span></div>
            <input type="range" min={1} max={1024} value={batch} onChange={e => setBatch(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1"><label className="font-medium text-slate-700">Context length</label><span className="font-mono text-slate-900">{context.toLocaleString()}</span></div>
            <input type="range" min={256} max={65536} step={256} value={context} onChange={e => setContext(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1"><label className="font-medium text-slate-700">KV size (kB/token)</label><span className="font-mono text-slate-900">{kvKb}</span></div>
            <input type="range" min={16} max={1024} step={16} value={kvKb} onChange={e => setKvKb(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>
          <div className="glass rounded-lg p-4">
            <div className="text-sm text-slate-500 mb-1">Total KV-cache memory (batch × context)</div>
            <div className="text-2xl font-mono font-bold text-slate-900">{kvTotalGb.toFixed(1)} <span className="text-sm font-normal text-slate-500">GB</span></div>
            <div className="text-[11px] text-slate-400 mt-1">
              This is the dominant memory/bandwidth cost during generation. Per-token step time ≥ (KV + params)/bandwidth.
            </div>
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.4} />
              <XAxis type="number" scale="log" domain={[0.5, 'dataMax']} label={{ value: 'Arithmetic intensity (log)', position: 'bottom', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={190} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [`${Number(v).toFixed(1)} FLOPs/B`, 'Intensity']} labelFormatter={(l: any, payload: any) => payload[0]?.payload?.desc || ''} />
              <Bar dataKey="intensity" name="Intensity" radius={[0, 4, 4, 0]}>
                {data.map((d, i) => (
                  <Cell key={i} fill={d.class === 'compute' ? '#10b981' : d.class === 'middle' ? '#f59e0b' : '#ef4444'} />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function NetworkRooflineInteractiveSection({ hw }: { hw: any }) {
  const [D, setD] = useState(4096);
  const [B, setB] = useState(256);
  const [networkBwGbps, setNetworkBwGbps] = useState(400);
  const F = 4096;

  const netBwBytes = (networkBwGbps * 1e9) / 8;
  const chipFlops = hw.tflops * 1e12;
  const interChipIntensityThreshold = chipFlops / netBwBytes;

  const currentIntensity = (B * D * F) / (B * F * 2);

  return (
    <div className="glass rounded-xl p-5">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Model Dimension (D)</label>
              <span className="font-mono text-slate-900">{D}</span>
            </div>
            <input type="range" min={1024} max={32768} step={1024} value={D} onChange={e => setD(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <label className="font-medium text-slate-700">Network Bandwidth</label>
              <span className="font-mono text-slate-900">{networkBwGbps} Gbps</span>
            </div>
            <input type="range" min={10} max={3200} step={100} value={networkBwGbps} onChange={e => setNetworkBwGbps(Number(e.target.value))} className="glass-slider w-full glass-slider w-full" />
          </div>

          <div className="glass rounded-lg p-4 mt-4">
            <div className="text-sm text-slate-500 mb-1">Inter-Chip Math vs Comm</div>
            <div className="text-2xl font-mono font-bold text-slate-900">{currentIntensity.toFixed(0)} <span className="text-sm font-normal text-slate-500">FLOPs/B</span></div>
            <div className={cn("text-xs font-bold mt-2", currentIntensity > interChipIntensityThreshold ? "text-emerald-500" : "text-amber-500")}>
              {currentIntensity > interChipIntensityThreshold ? "✅ Compute Bound (Network is fast enough)" : "⚠️ Network Bound"}
            </div>
            <div className="text-[10px] text-slate-400 mt-1">
              (Requires &gt; {interChipIntensityThreshold.toFixed(0)} FLOPs/B. Threshold depends on D, not B.)
            </div>
          </div>
        </div>
        <div className="flex items-center justify-center p-4">
          <div className="text-center space-y-2">
            <div className="p-3 bg-purple-100 text-purple-800 rounded-lg font-bold">Chip 1 computes half</div>
            <div className="animate-pulse font-mono text-purple-400 text-sm">↓ {netBwBytes.toExponential(1)} Bytes/s ↑</div>
            <div className="p-3 bg-purple-100 text-purple-800 rounded-lg font-bold">Chip 2 computes half</div>
            <p className="text-xs text-slate-500 mt-4 max-w-xs">
              With a sharded D dimension, partial sums of size B × F must cross the network. The larger D is, the more math per byte of communication — so the roofline here is set by D, not by B.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function QuantizationInteractiveSection({ hardwareIntensity }: { hardwareIntensity: number }) {
  const B = 256;
  const D = 4096;
  const F = 4096;

  const denom = (wb: number, wa: number) => (wb * B * D + wb * D * F + wa * B * F);
  const int1 = (2 * B * D * F) / (2 * B * D + 2 * D * F + 2 * B * F);
  const int2 = (2 * B * D * F) / (1 * B * D + 1 * D * F + 1 * B * F);
  const int3 = (2 * B * D * F) / (2 * B * D + 1 * D * F + 2 * B * F);
  const int4 = (2 * B * D * F) / (2 * B * D + 2 * B * D * F + 2 * B * F);

  const data = [
    { name: 'BF16', intensity: int1, bCrit: hardwareIntensity, note: 'weights & compute bf16' },
    { name: 'Mixed int8 w', intensity: int3, bCrit: hardwareIntensity / 2, note: 'int8 weights, bf16 compute → B_crit halved' },
    { name: 'Pure int8', intensity: int2, bCrit: hardwareIntensity, note: 'int8 both; compute 2× → B_crit back to ~base' },
    { name: 'Batch-Specific', intensity: int4, bCrit: 2, note: 'unique weights per token → always bound' },
  ];

  return (
    <div className="glass rounded-xl p-5">
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} />
            <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="intensity" name="Intensity" fill="#d97706" radius={[4, 4, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.intensity > hardwareIntensity ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
            <ReferenceLine y={hardwareIntensity} stroke="#0ea5e9" strokeDasharray="3 3" label={{ position: 'top', value: 'Ridge', fill: '#0ea5e9', fontSize: 10 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="overflow-x-auto">
        <table className="glass-slider w-full text-sm text-slate-600">
          <thead>
            <tr className="text-left text-slate-400 text-xs uppercase tracking-wider border-b border-slate-200">
              <th className="py-2 pr-3">Scheme</th>
              <th className="py-2 pr-3">Intensity (B=256)</th>
              <th className="py-2 pr-3">Critical batch B_crit</th>
              <th className="py-2">Effect</th>
            </tr>
          </thead>
          <tbody>
            {data.map(d => (
              <tr key={d.name} className="border-b border-slate-100">
                <td className="py-2 pr-3 font-medium">{d.name}</td>
                <td className="py-2 pr-3 font-mono">{d.intensity.toFixed(1)}</td>
                <td className="py-2 pr-3 font-mono">≈{d.bCrit.toFixed(0)}</td>
                <td className="py-2 text-slate-500">{d.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Section 8: Worked problems
// -------------------------------------------------------------
function WorkedProblemsSection({ hw, peakFlops, peakBw, hardwareIntensity }: any) {
  const hbmBw = 8.2e11;          // textbook TPU value
  const bf16Flops = 1.97e14;     // textbook TPU v5e value
  const int8Flops = 3.94e14;
  const netBw = 4.5e10;

  const [revealed, setRevealed] = useState<Record<string, boolean>>({});

  const toggle = (key: string) => setRevealed(r => ({ ...r, [key]: !r[key] }));

  const Answer = ({ id, children }: any) => revealed[id] ? (
    <div className="text-xs bg-emerald-50 border border-emerald-200 text-emerald-900 rounded-lg p-3 mt-2 space-y-1">{children}</div>
  ) : (
    <button onClick={() => toggle(id)} className="text-xs text-blue-600 hover:underline mt-2">Show answer</button>
  );

  const Q = ({ children }: any) => (
    <div className="glass rounded-lg p-4">
      <div className="text-sm text-slate-700 leading-relaxed">{children}</div>
    </div>
  );

  // Q3 roofline-vs-B curve
  const q3Data = useMemo(() => {
    const rows = [];
    const roofline = (B: number, DD: number, FF: number) => {
      const tf = 2 * B * DD * FF;
      const ft = tf / bf16Flops;
      const ct = (2 * B * DD + DD * FF + 2 * B * FF) / hbmBw;
      const tt = Math.max(ft, ct);
      return tf / tt;
    };
    for (let b = 1; b <= 512; b += 8) {
      rows.push({ B: b, big: roofline(b, 4096, 4096) / 1e12, small: roofline(b, 1024, 1024) / 1e12 });
    }
    return rows;
  }, []);

  return (
    <div className="space-y-4">
      <Q>
        <strong>Q1 — int8 matmul:</strong> <i>X[B,D] ·<sub>D</sub> Y[D,F] → Z[B,F]</i> in int8 (1 byte/param) vs bf16 (2 bytes/param).
        <ol className="list-decimal pl-5 mt-2 text-xs text-slate-500">
          <li>Bytes loaded / written?</li>
          <li>Total OPs?</li>
          <li>Arithmetic intensity?</li>
          <li>T_math, T_comms, and reasonable lower/upper runtime bounds?</li>
        </ol>
        <Answer id="q1">
          <p><strong>1.</strong> Load <i>BD + DF</i> bytes (1 byte/param), write <i>BF</i>.</p>
          <p><strong>2.</strong> Still <i>2BDF</i> OPs (int8 just runs faster).</p>
          <p><strong>3.</strong> <i>I = 2BDF/(BD+DF+BF) ≈ 2B</i>. int8 ridge = <i>3.94e14/8.2e11 = 480</i>, so rule <i>B &gt; 240</i> — basically unchanged from bf16.</p>
          <p><strong>4.</strong> <i>T_math = 2BDF/3.94e14</i>, <i>T_comms = (BD+DF+BF)/8.2e11</i>. Lower bound = max, upper = sum.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q2 — int8 weights + bf16 math:</strong> weights int8, activations/compute bf16,
        <i>1.97e14</i> bf16 FLOPs/s. At what batch size do we become compute-bound?
        <Answer id="q2">
          <p><i>2BDF</i> bf16 FLOPs but only <i>DF</i> weight bytes. Compute-bound when
          <i>2B &gt; 240 → B<sub>crit</sub> &gt; 120</i> — half of bf16. Easy quantization win.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q3 — roofline vs B:</strong> peak FLOPs/s vs B for <i>D=F=4096</i> and <i>D=F=1024</i>
        (exact bytes). Larger D/F reaches peak sooner; D=F=1024 roughly doubles the critical batch size.
        <div className="h-56 mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={q3Data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="B" label={{ value: 'Batch', position: 'bottom', fontSize: 10 }} />
              <YAxis tickFormatter={(v: any) => `${Number(v).toFixed(0)}`} label={{ value: 'TFLOP/s', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [`${Number(v).toFixed(1)} TFLOP/s`]} />
              <Line type="monotone" dataKey="big" name="D=F=4096" stroke="#2563eb" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="small" name="D=F=1024" stroke="#f59e0b" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Answer id="q3">
          <p>Both curves saturate at the hardware peak (~{ (bf16Flops/1e12).toFixed(0) } TFLOP/s on the textbook TPU),
          but the bigger model crosses the ridge at a smaller batch. Small matmuls need ~2× the batch to become compute-bound.</p>
          <p className="text-[10px] text-slate-400">This mirrors your Interactive Lab: small batches are memory-bound, and the crossover batch is set by the hardware ridge.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q4 — batch-specific weights:</strong> <i>int8[B,D] ·<sub>D</sub> int8[B,D,F] → int8[B,F]</i>, a different matrix per batch element. Arithmetic intensity?
        <Answer id="q4">
          <p>FLOPs = <i>2BDF</i>. Comms = <i>BD + BDF + BF</i>. Since <i>BDF</i> dominates the denominator,
          <i>I ≈ 2</i>, constant. The operation is essentially <strong>always communication-bound</strong> regardless of B.</p>
        </Answer>
      </Q>

      <Q>
        <strong>Q5 — H100 memory roofline:</strong> using the H100 spec sheet, find the batch at which a bf16
        matmul becomes compute-bound. (Tensor-core bf16 figure is ~2× the true value due to sparsity.)
        <Answer id="q5">
          <p>Reported bf16 = <i>1.979e15</i> "with sparsity"; true value = <i>9.89e14</i>. With
          <i>3.35e12</i> bytes/s bandwidth: <i>B<sub>crit</sub> = 9.89e14 / 3.35e12 ≈ 295</i> tokens — similar to TPUs.</p>
        </Answer>
      </Q>

      <p className="text-sm text-slate-500">
        Adapted from the scaling-book (Part 1, "A Few Problems to Work"). Reference copy stored in{" "}
        <code className="text-xs bg-slate-100 px-1 rounded">reference/scaling-book/roofline.md</code>.
      </p>
    </div>
  );
}
