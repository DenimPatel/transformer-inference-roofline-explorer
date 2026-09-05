import { useMemo, useState } from 'react';
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { cn } from '../../lib/utils';

// TPU v5e, the scaling book's worked example.
const CHIP_FLOPS = 1.97e14;   // FLOPs/s per chip
const TIERS = [
  { id: 'hbm', label: 'HBM (in-chip)', bytes: 8.2e11, note: 'weights ↔ compute cores' },
  { id: 'ici', label: 'ICI / NVLink', bytes: 4.5e10, note: 'chip ↔ chip, same domain' },
  { id: 'pcie', label: 'PCIe (host)', bytes: 1.6e10, note: 'chip ↔ CPU' },
  { id: 'dcn', label: 'DCN / InfiniBand', bytes: 6.25e9, note: 'across pods' },
] as const;

export default function LessonNetworkRoofline({ onComplete }: { onComplete: () => void }) {
  const [D, setD] = useState(8192);
  const [tierId, setTierId] = useState<string>('ici');

  const tier = TIERS.find((t) => t.id === tierId) ?? TIERS[1];
  const threshold = CHIP_FLOPS / tier.bytes;   // intensity the link demands
  const intensity = D / 2;                      // 2-chip sharded matmul
  const criticalD = 2 * threshold;

  const curve = useMemo(() => {
    const rows: any[] = [];
    for (let d = 512; d <= 131072; d *= 1.15) {
      const dd = Math.round(d);
      rows.push({
        D: dd,
        ici: Math.min((dd / 2) / (CHIP_FLOPS / 4.5e10), 1) * 100,
        dcn: Math.min((dd / 2) / (CHIP_FLOPS / 6.25e9), 1) * 100,
      });
    }
    return rows;
  }, []);

  return (
    <LessonShell
      number={13}
      title="The Network Roofline"
      subtitle="Once a model spans chips, the fabric becomes the roof — and the threshold stops depending on batch size."
      sourceRef="reference/scaling-book/roofline.md — Network communication rooflines · tpus.md — TPU Networking · gpus.md — Rooflines for LLM Scaling on GPUs"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="network-roofline" />
          <ConceptTag id="model-parallelism" />
          <ConceptTag id="tensor-parallelism" />
          <ConceptTag id="collectives" />
          <ConceptTag id="ici-topology" />
          <ConceptTag id="nvlink-domain" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          Every roofline so far has compared FLOPs against <strong>HBM</strong> bandwidth. But &ldquo;bandwidth&rdquo; is
          not one number. A tensor may cross three very different links, each two orders of magnitude apart, and each
          with its own ridge:
        </p>

        <div className="grid sm:grid-cols-2 gap-2">
          {TIERS.map((t) => (
            <button key={t.id} onClick={() => setTierId(t.id)}
              className={cn('glass rounded-xl p-3 text-left transition-colors',
                tierId === t.id ? 'border-accent ring-1 ring-accent/40' : 'hover:border-accent/40')}>
              <div className="font-semibold text-slate-800 text-sm">{t.label}</div>
              <div className="text-[11px] text-slate-400">{t.note}</div>
              <div className="font-mono text-xs text-slate-600 mt-1">
                {(t.bytes / 1e9).toFixed(t.bytes < 1e10 ? 2 : 0)} GB/s · ridge {(CHIP_FLOPS / t.bytes).toFixed(0)}
              </div>
            </button>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5 space-y-4">
        <h3 className="font-bold text-slate-800">Two chips, one matmul</h3>
        <p className="text-sm leading-relaxed text-slate-600">
          Split <code className="text-xs">X[B,D] · Y[D,F]</code> across two chips along the contracted dimension{' '}
          <i>D</i>. Each chip multiplies its half and then ships its <strong>partial sums</strong> to the other so they
          can be added. Each chip does <i>BDF</i> FLOPs and sends <i>2BF</i> bytes, so:
        </p>
        <div className="glass rounded-xl p-4 font-mono text-sm text-center text-slate-700">
          I = BDF / 2BF = <strong className="text-accent">D / 2</strong>
        </div>
        <p className="text-sm leading-relaxed text-slate-600">
          Look at what cancelled: <strong>B is gone</strong>. On the HBM roofline, intensity was <i>≈ B</i> and batching
          was the fix. Here the threshold depends only on the <strong>model dimension</strong>. You cannot batch your way
          out of a network bottleneck — you need a wider model, a faster link, or less sharding.
        </p>

        <SliderControl label="Model dimension (D)" value={D} min={512} max={65536} step={512}
          onChange={setD} unit="" logScale conceptId="network-roofline" />

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">Intensity (D/2)</div>
            <div className="font-mono font-bold text-slate-800">{intensity.toFixed(0)}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">{tier.label} demands</div>
            <div className="font-mono font-bold text-slate-800">{threshold.toFixed(0)}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">Compute-bound when</div>
            <div className="font-mono font-bold text-slate-800">D &gt; {criticalD.toFixed(0)}</div>
          </div>
        </div>

        <div className={cn('text-sm font-semibold rounded-xl px-3 py-2',
          intensity > threshold ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700')}>
          {intensity > threshold
            ? `At D = ${D} you are compute-bound over the ${tier.label} link — the fabric keeps up.`
            : `At D = ${D} you are comms-bound over the ${tier.label} link. You need D > ${criticalD.toFixed(0)}, a faster link, or fewer shards.`}
        </div>
        <p className="text-xs text-slate-400">
          Over ICI at 4.5e10 bytes/s the book&rsquo;s numbers give <i>D/2 &gt; 4377</i>, i.e. <strong>D &gt; 8755</strong>.
          Most frontier models clear that comfortably, which is exactly why in-domain tensor parallelism works and
          cross-datacenter tensor parallelism does not.
        </p>
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="font-bold text-slate-800 mb-3">Same model, two fabrics</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={curve} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="D" scale="log" type="number" domain={['dataMin', 'dataMax']}
                tickFormatter={(v: any) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${v}`)}
                label={{ value: 'Model dimension D (log)', position: 'bottom', fontSize: 10 }} />
              <YAxis domain={[0, 100]} tickFormatter={(v: any) => `${v}%`}
                label={{ value: '% of peak FLOPs', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [`${Number(v).toFixed(0)}%`, 'of peak']}
                labelFormatter={(v: any) => `D = ${Number(v).toLocaleString()}`} />
              <ReferenceLine x={8755} stroke="#f43f5e" strokeDasharray="3 3"
                label={{ position: 'top', value: 'ICI needs D > 8755', fill: '#f43f5e', fontSize: 10 }} />
              <ReferenceLine x={D} stroke="#475569" />
              <Line type="monotone" dataKey="ici" name="ICI / NVLink" stroke="#5b7cfa" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="dcn" name="DCN / InfiniBand" stroke="#f59e0b" strokeWidth={3} strokeDasharray="6 4" dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-slate-400 mt-3">
          The scale-out curve needs a model roughly 7x wider to reach the same utilization. That single gap is why tensor
          parallelism stops at the edge of an NVLink node or ICI slice, and data or pipeline parallelism takes over
          beyond it.
        </p>
      </GlassCard>

      <GlassCard className="p-5 space-y-3">
        <h3 className="font-bold text-slate-800">The thresholds worth memorizing</h3>
        {[
          { k: 'Tensor parallelism', r: 'Y < F · W / C  →  ≈ 8-way', w: 'No B in the rule. For F = 28,672 the arithmetic gives ~11-way in-node, i.e. about 8 on real hardware — which is precisely how big an NVLink domain is.' },
          { k: 'Data parallelism / FSDP', r: 'B / X > C / W  →  ≈ 2,500 tokens/GPU', w: 'Gradients are a fixed size, so tokens amortize them. This rule does contain B, so batching genuinely helps.' },
          { k: '2-chip sharded matmul', r: 'D / 2 > C / W  →  D > 8,755 over ICI', w: 'The derivation on this page. Depends on model width, not batch.' },
        ].map((x) => (
          <div key={x.k} className="glass rounded-xl p-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="font-semibold text-slate-800 text-sm">{x.k}</span>
              <span className="font-mono text-[11px] font-bold text-accent">{x.r}</span>
            </div>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">{x.w}</p>
          </div>
        ))}
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'For a matmul sharded across two chips along D, the arithmetic intensity is D/2. What does that tell you?',
            options: [
              'Raising the batch size will make it compute-bound',
              'The threshold depends on the model dimension, not the batch — batching cannot fix a network bottleneck',
              'It is always compute-bound because D is large',
              'The intensity is the same as the HBM roofline',
            ],
            answer: 1,
            explain: 'Each chip does BDF FLOPs and sends 2BF bytes, so B cancels: I = D/2. Unlike the HBM roofline (I ≈ B), no amount of batching moves this. Only a wider model, a faster link, or fewer shards will.',
          },
          {
            q: 'Why does tensor parallelism in practice stop at about 8 ways?',
            options: [
              'Because 8 is the maximum number of GPUs a server chassis can hold',
              'Because the rule Y < F·W/C gives ~11-way inside an NVLink domain for a typical F, and bandwidth collapses past the domain boundary',
              'Because past 8 ways the model no longer fits in HBM',
              'Because collectives only support powers of two up to 8',
            ],
            answer: 1,
            explain: 'The TP roofline gives Y < F·W/C, roughly F/2200 within a node. For LLaMA-3’s F = 28,672 that is ~11-way, or about 8 once rounded to real hardware — and crossing the node boundary drops W by an order of magnitude, so the rule tightens sharply.',
          },
          {
            q: 'You are comms-bound on the DCN / InfiniBand tier. Which change helps most?',
            options: [
              'Increase the global batch size',
              'Restructure so the collective happens inside one ICI/NVLink domain instead',
              'Reduce the model dimension D',
              'Switch the KV cache to int8',
            ],
            answer: 1,
            explain: 'The scale-out fabric is roughly two orders of magnitude slower than the in-domain link, so its critical D is proportionally larger. Keeping the traffic inside a domain changes the ridge you are measured against; batching does not, and shrinking D makes it worse.',
          },
        ]}
      />
    </LessonShell>
  );
}
