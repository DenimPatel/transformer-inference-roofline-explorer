import { useMemo } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ComposedChart, Scatter, Line,
} from 'recharts';
import { cn } from '../../lib/utils';
import GlassCard from '../ui/GlassCard';
import KpiCard from '../ui/KpiCard';
import Figure from '../shell/Figure';
import { useConfig } from '../../state/ConfigContext';

function RooflineGrounding({
  peakFlops, peakBw, ridge, opIntensity, opName,
}: {
  peakFlops: number; peakBw: number; ridge: number; opIntensity: number; opName: string;
}) {
  const data = useMemo(() => {
    const pts: { intensity: number; achievable: number }[] = [];
    for (let i = Math.log10(0.05); i <= Math.log10(400000); i += 0.08) {
      const intensity = Math.pow(10, i);
      pts.push({ intensity, achievable: Math.min(peakBw * intensity, peakFlops) });
    }
    return pts;
  }, [peakFlops, peakBw]);

  const achieved = Math.min(peakBw * opIntensity, peakFlops);
  const isMemBound = opIntensity < ridge;

  return (
    <div>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 20, right: 24, left: 0, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.4} />
            <XAxis dataKey="intensity" scale="log" domain={['dataMin', 'dataMax']} type="number"
              tickFormatter={(v) => (Number(v) < 1 ? Number(v).toFixed(1) : Number(v).toFixed(0))}
              label={{ value: 'Arithmetic Intensity (FLOPs/B) — log', position: 'bottom', offset: -10, fontSize: 11 }} />
            <YAxis scale="log" domain={['dataMin', 'dataMax']} type="number"
              tickFormatter={(v) => (v / 1e12).toFixed(1)}
              label={{ value: 'Throughput (TFLOPs/s) — log', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip labelFormatter={(v) => `Intensity: ${Number(v).toFixed(2)} FLOPs/B`}
              formatter={(v) => [`${(Number(v) / 1e12).toFixed(2)} TFLOP/s`]} />
            <Line type="monotone" dataKey="achievable" name="Roofline" stroke="#0088b0" strokeWidth={3} dot={false} />
            <ReferenceLine x={ridge} stroke="#d6006c" strokeDasharray="4 4"
              label={{ position: 'top', value: `ridge ≈ ${ridge.toFixed(0)}`, fill: '#d6006c', fontSize: 10 }} />
            <Scatter data={[{ intensity: opIntensity, achieved }]} dataKey="achieved"
              fill={isMemBound ? '#c8963a' : '#2f8365'}
              name={`${opName}: ${isMemBound ? 'memory-bound' : 'compute-bound'}`}
              shape="circle" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-2 mt-3 text-xs">
        <span className="glass-chip px-2.5 py-1 text-slate-600">
          Operating intensity: <span className="font-mono font-bold">{opIntensity.toFixed(2)}</span> FLOPs/B
        </span>
        <span className="glass-chip px-2.5 py-1 text-slate-600">
          Hardware ridge: <span className="font-mono font-bold">{ridge.toFixed(1)}</span> FLOPs/B
        </span>
        <span className={cn('px-2.5 py-1 rounded-full font-bold',
          isMemBound ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700')}>
          {isMemBound ? 'Memory-bound (below ridge)' : 'Compute-bound (at ridge/peak)'}
        </span>
      </div>
    </div>
  );
}

/**
 * The reader's own configuration, placed on the roofline they have just read
 * about. This was the Interactive Lab tab; it works far better as the closing
 * move of the roofline section than as a separate destination.
 */
export default function OperatingPoint() {
  const {
    units, currentStat, currentStatIntensity, activeProfileId, flopsTera, tdpWatts,
    maxTokensPerSec, optimalBatchSize, drainTime,
  } = useConfig();

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        <KpiCard label="Limiting Factor" value={currentStat.isMemoryBound ? 'Memory Bound' : 'Compute Bound'}
          colorClass={currentStat.isMemoryBound ? 'text-amber-500' : 'text-emerald-600'}
          subValue={`Cost: $${currentStat.totalCost1M.toFixed(2)} / 1M tkns`}
          conceptId={currentStat.isMemoryBound ? 'memory-bound' : 'compute-bound'} />
        <KpiCard label="Energy per Token"
          value={currentStat.joulesPerToken.toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 })}
          unit="Joules" subValue={`Pwr Draw: ${currentStat.powerDrawW.toFixed(0)}W`} conceptId="tco" />
        <KpiCard label="Performance Density"
          value={(flopsTera * 1000 / tdpWatts).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
          unit="TFLOP/W" conceptId="flops" />
        <KpiCard label="Max Token Throughput"
          value={maxTokensPerSec.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          unit="tokens/s" conceptId="latency-throughput" />
        <KpiCard label="Optimal Batch (Balance)"
          value={optimalBatchSize.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          unit="seqs" conceptId="critical-batch" />
        <KpiCard label="Drain Time (Capacity/BW)" value={drainTime.toFixed(1)} unit="ms" conceptId="memory-bound" />
      </div>

      <GlassCard className="p-5">
        <h3 className="text-lg font-bold text-slate-800 mb-1">Where is your operating point?</h3>
        <p className="text-sm text-slate-500 mb-2">
          A single decode step on <strong>{activeProfileId}</strong>, with the model and batch size
          you have set. Change either in the configuration panel and the dot moves.
        </p>
        <Figure
          takeaway={
            currentStat.isMemoryBound
              ? `The dot sits to the left of the pink ridge line, on the sloped part of the roof. That means bandwidth is the limit: the chip is waiting for weights, not doing arithmetic. Moving right — larger batches — buys throughput until the dot reaches the ridge.`
              : `The dot sits at or past the pink ridge, on the flat part of the roof. Arithmetic is now the limit; more batching buys latency, not throughput.`
          }
          caption="Both axes are logarithmic. The roof is the best any operation can do on this chip."
        >
          <RooflineGrounding
            peakFlops={units.flops}
            peakBw={units.memBw}
            ridge={units.hardwareRatio}
            opIntensity={currentStatIntensity}
            opName="Current decode step"
          />
        </Figure>
      </GlassCard>
    </div>
  );
}
