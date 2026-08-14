import { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ComposedChart, Bar, Cell } from 'recharts';
import GlassCard from '../ui/GlassCard';
import { ConceptTag } from '../ui/ConceptTag';
import LessonShell from './LessonShell';
import Checkpoint from './Checkpoint';
import SliderControl from '../ui/SliderControl';
import { cn } from '../../lib/utils';

const HBM_RIDGE = 295; // H100 bf16
const VMEM_INTENSITY = 20; // ~10-20 saturates the MXU from on-chip memory

export default function LessonMemory({ onComplete }: { onComplete: () => void }) {
  const [tile, setTile] = useState(128);

  // Effective intensity of a tiled matmul when the tile is bm x bn, reloaded from HBM
  const tileIntensity = (tile * tile) / (tile + tile); // bm*bn/(bm+bn) with bm=bn=tile

  const curve = useMemo(() => {
    const rows: any[] = [];
    for (let t = 16; t <= 4096; t *= 1.15) {
      const tt = Math.round(t);
      rows.push({ tile: tt, intensity: (tt * tt) / (tt + tt) });
    }
    return rows;
  }, []);

  const ops = [
    { name: 'Dot product (vector)', intensity: 0.5, color: '#ef4444' },
    { name: 'Attention (gate)', intensity: 1, color: '#f59e0b' },
    { name: 'Tiled matmul (tile=128)', intensity: 64, color: '#3b82f6' },
    { name: 'Matmul, B=1024', intensity: 1024, color: '#8b5cf6' },
  ];

  return (
    <LessonShell
      number={8}
      title="Memory Hierarchy & the On-Chip Wall"
      subtitle="HBM is big and slow; on-chip VMEM is tiny but ~22x faster — and it decides whether you can saturate the math units."
      sourceRef="reference/scaling-book/tpus.md — VMEM and arithmetic intensity · reference/scaling-book/gpus.md — Memory"
    >
      <GlassCard className="p-5 space-y-4">
        <div className="flex flex-wrap gap-2">
          <ConceptTag id="memory-hierarchy" />
          <ConceptTag id="tiling" />
          <ConceptTag id="bandwidth" />
          <ConceptTag id="arithmetic-intensity" />
        </div>

        <p className="text-sm leading-relaxed text-slate-600">
          There are two memory tiers that matter for the roofline: <strong>HBM</strong> (large, off-chip, ~1-2 TB/s) and{' '}
          <strong>on-chip memory</strong> (VMEM on TPUs, SMEM/TMEM on GPUs) that sits next to the math units. On-chip memory
          has roughly <strong>22x the bandwidth</strong> of HBM — an MXU op needs only <code className="text-xs">10-20</code>{' '}
          FLOPs/byte to saturate from VMEM, versus <code className="text-xs">~240</code> from HBM.
        </p>

        <p className="text-sm leading-relaxed text-slate-600">
          That is why <strong>tiling</strong> matters: a big matmul is cut into chunks that fit on-chip memory and re-loaded,
          so the effective intensity becomes <code className="text-xs">bm·bn/(bm+bn)</code> — usually far below the HBM ridge.
        </p>

        <SliderControl
          label="On-chip tile size (bm = bn)"
          value={tile}
          min={16}
          max={2048}
          step={16}
          onChange={setTile}
          unit=""
          logScale
          conceptId="tiling"
        />

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">Tiled intensity</div>
            <div className="font-mono font-bold text-slate-800">{tileIntensity.toFixed(0)} FLOPs/B</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">HBM ridge</div>
            <div className="font-mono font-bold text-slate-800">{HBM_RIDGE}</div>
          </div>
          <div className="glass rounded-xl px-3 py-2">
            <div className="text-[11px] text-slate-400">On-chip threshold</div>
            <div className="font-mono font-bold text-slate-800">~{VMEM_INTENSITY}</div>
          </div>
        </div>

        <div className={cn('text-sm font-semibold rounded-xl px-3 py-2', tileIntensity >= HBM_RIDGE ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700')}>
          {tileIntensity >= HBM_RIDGE
            ? 'This tile saturates the MXU from HBM directly (above the ridge).'
            : 'This tile is below the HBM ridge — it must be re-loaded from HBM, wasting bandwidth. A bigger tile (or keeping data in VMEM) fixes this.'}
        </div>

        <div className="flex flex-wrap gap-2 text-[11px] text-slate-500">
          {[
            'Dot product: intensity ≈ 1/2 — effectively always bandwidth-bound',
            'Vector (softmax/relu) ops run on a much slower unit with a lower ridge',
            'Tile size is really an arithmetic-intensity knob',
          ].map((tip) => (
            <span key={tip} className="glass-chip px-2.5 py-1">{tip}</span>
          ))}
        </div>
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="font-bold text-slate-800 mb-3">Where ops land vs the on-chip & HBM ridges</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={ops} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis scale="log" domain={[0.3, 4000]} type="number" tickFormatter={(v: any) => (v < 1 ? v.toFixed(1) : v.toFixed(0))} label={{ value: 'Intensity (FLOPs/B, log)', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [`${Number(v).toFixed(1)} FLOPs/B`, 'Intensity']} />
              <ReferenceLine y={VMEM_INTENSITY} stroke="#10b981" strokeDasharray="3 3" label={{ position: 'top', value: 'VMEM ~20', fill: '#10b981', fontSize: 10 }} />
              <ReferenceLine y={HBM_RIDGE} stroke="#f43f5e" strokeDasharray="3 3" label={{ position: 'top', value: `HBM ${HBM_RIDGE}`, fill: '#f43f5e', fontSize: 10 }} />
              <Bar dataKey="intensity" radius={[6, 6, 0, 0]}>
                {ops.map((o) => (
                  <Cell key={o.name} fill={o.color} />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="h-56 mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={curve} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="tile" scale="log" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => `${v}`} label={{ value: 'Tile size (log)', position: 'bottom', fontSize: 10 }} />
              <YAxis label={{ value: 'Effective intensity', angle: -90, position: 'insideLeft', fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [`${Number(v).toFixed(0)}`, 'Intensity']} labelFormatter={(v: any) => `tile ${v}`} />
              <ReferenceLine y={HBM_RIDGE} stroke="#f43f5e" strokeDasharray="3 3" label={{ position: 'top', value: 'HBM ridge', fill: '#f43f5e', fontSize: 10 }} />
              <ReferenceLine x={tile} stroke="#475569" />
              <Line type="monotone" dataKey="intensity" stroke="#5b7cfa" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      <Checkpoint
        onComplete={onComplete}
        questions={[
          {
            q: 'Why does on-chip VMEM let an op become compute-bound at far lower arithmetic intensity than HBM?',
            options: [
              'Because VMEM is ~22x higher bandwidth than HBM',
              'Because VMEM stores weights in int8',
              'Because VMEM is bigger than HBM',
              'Because VMEM does the matmul itself',
            ],
            answer: 0,
            explain: 'Higher bandwidth means each byte is much cheaper to move, so the FLOPs/byte you need to saturate the MXU drops to ~10-20.',
          },
          {
            q: 'Tiling a matmul into on-chip chunks has what effect on arithmetic intensity?',
            options: [
              'It raises intensity toward 2·B',
              'It lowers effective intensity to ≈ bm·bn/(bm+bn) because tiles are re-loaded',
              'It always keeps intensity above the ridge',
              'It has no effect on intensity',
            ],
            answer: 1,
            explain: 'Re-loading tiles from HBM adds bytes beyond the naive O(N²) estimate, dropping effective intensity to ≈ bm·bn/(bm+bn).',
          },
        ]}
      />
    </LessonShell>
  );
}
