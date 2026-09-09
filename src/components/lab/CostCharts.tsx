import type { ReactNode } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ComposedChart, Line,
} from 'recharts';
import { cn } from '../../lib/utils';
import GlassCard from '../ui/GlassCard';
import Figure from '../shell/Figure';
import { useConfig } from '../../state/ConfigContext';

const TooltipShell = ({ children }: { children: ReactNode }) => (
  <div className="glass-tooltip p-3 rounded-xl text-xs space-y-1 text-slate-700 min-w-[200px] z-50">{children}</div>
);

const LatencyTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <TooltipShell>
      <div className="font-bold text-slate-900 mb-2 border-b border-slate-200 pb-1">Batch Size: {Number(label).toLocaleString()} seqs</div>
      <div className="text-emerald-600 flex justify-between gap-4"><span>Compute Time:</span><span>{p.tCompute.toFixed(2)} ms</span></div>
      <div className="text-amber-600 flex justify-between gap-4"><span>Memory Time:</span><span>{p.tMemory.toFixed(2)} ms</span></div>
      <div className="text-slate-500 pl-4 flex justify-between gap-4"><span>Weight Fetch:</span><span>{p.tWeightFetch.toFixed(2)} ms</span></div>
      <div className="text-slate-500 pl-4 flex justify-between mb-1 pb-1 border-b border-slate-200 gap-4"><span>KV Fetch:</span><span>{p.tKvFetch.toFixed(2)} ms</span></div>
      <div className="text-slate-900 font-bold flex justify-between pt-1 gap-4"><span>Total Latency:</span><span>{p.latency.toFixed(2)} ms</span></div>
      <div className={cn('mt-2 font-bold', p.isMemoryBound ? 'text-amber-600' : 'text-emerald-600')}>
        {p.isMemoryBound ? 'Memory Bound' : 'Compute Bound'}
      </div>
    </TooltipShell>
  );
};

const CostTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <TooltipShell>
      <div className="font-bold text-slate-900 mb-2 border-b border-slate-200 pb-1">Batch Size: {Number(label).toLocaleString()} seqs</div>
      <div className="text-amber-600 flex justify-between gap-4"><span>Power Cost:</span><span>${p.costElec1M.toFixed(2)}</span></div>
      <div className="text-slate-600 flex justify-between gap-4 pb-1 border-b border-slate-200"><span>Hardware Cost:</span><span>${p.costHardware1M.toFixed(2)}</span></div>
      <div className="text-emerald-600 font-bold flex justify-between gap-4 pt-1"><span>Total TCO:</span><span>${p.totalCost1M.toFixed(2)}</span></div>
    </TooltipShell>
  );
};

const AXIS = { stroke: '#d7d3d3', tick: { fill: '#7d7979', fontSize: 11 } } as const;

/**
 * Latency and cost against batch size, for the reader's own configuration.
 * Formerly the right-hand half of the Interactive Lab tab.
 */
export default function CostCharts() {
  const { chartData, currentStat, regionId, amortizationYears } = useConfig();

  const cheapest = chartData.reduce((a, b) => (b.totalCost1M < a.totalCost1M ? b : a));

  return (
    <div className="space-y-5">
      <GlassCard className="p-5">
        <h3 className="text-lg font-bold text-slate-800 mb-0.5">Latency vs batch size</h3>
        <p className="text-sm text-slate-500">Total latency = max(memory time, compute time).</p>
        <Figure
          takeaway={
            'The two dashed lines are the two budgets: memory time is nearly flat at small batches (you pay to read the weights no matter how few requests you have), while compute time rises linearly. The solid line — what you actually experience — is whichever is higher. Where they cross is the ridge, and to its left you are getting the arithmetic for free.'
          }
          caption="Both axes logarithmic. The vertical marker is your current batch size."
        >
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="#eae7e7" vertical={false} />
                <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                  tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} {...AXIS}
                  label={{ value: 'Batch Size (log)', position: 'insideBottom', offset: -12, fill: '#7d7979', fontSize: 11 }} />
                <YAxis {...AXIS} scale="log" domain={['dataMin', 'dataMax']} tickFormatter={(v) => v.toFixed(1)}
                  label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', fill: '#7d7979', fontSize: 11 }} />
                <Tooltip content={<LatencyTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line type="monotone" dataKey="tCompute" name="Compute Time" stroke="#2f8365" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                <Line type="monotone" dataKey="tMemory" name="Memory Time" stroke="#c8963a" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                <Line type="monotone" dataKey="latency" name="Total Latency" stroke="#2d2b2b" strokeWidth={3} dot={false} />
                <ReferenceLine x={currentStat.batchSize} stroke="#928e8e" strokeWidth={2}
                  label={{ position: 'top', value: 'Current', fill: '#7d7979', fontSize: 10 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </GlassCard>

      <GlassCard className="p-5">
        <h3 className="text-lg font-bold text-slate-800 mb-0.5">Total cost of ownership</h3>
        <p className="text-sm text-slate-500">
          Electricity ({regionId}) + hardware amortisation over {amortizationYears} years.
        </p>
        <Figure
          takeaway={
            `Cost per token falls steeply as batching spreads the fixed cost of reading the weights across more requests, then flattens once you are compute-bound and there is nothing left to amortise. On this configuration the floor is about $${cheapest.totalCost1M.toFixed(2)} per million tokens at a batch of ${Math.round(cheapest.batchSize).toLocaleString()} — past that you pay in latency for nothing.`
          }
          caption="Hardware CapEx dominates at small batches; electricity dominates once the chip is busy."
        >
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 24, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="4 4" stroke="#eae7e7" vertical={false} />
                <XAxis dataKey="batchSize" scale="log" domain={['dataMin', 'dataMax']} type="number"
                  tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} {...AXIS}
                  label={{ value: 'Batch Size (log)', position: 'insideBottom', offset: -12, fill: '#7d7979', fontSize: 11 }} />
                <YAxis {...AXIS} tickFormatter={(v) => '$' + Number(v).toFixed(2)}
                  label={{ value: 'Cost (USD/1M tkns)', angle: -90, position: 'insideLeft', fill: '#7d7979', fontSize: 11 }} />
                <Tooltip content={<CostTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line type="monotone" dataKey="costElec1M" name="Power Cost" stroke="#c8963a" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                <Line type="monotone" dataKey="costHardware1M" name="Hardware CapEx" stroke="#7d7979" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                <Line type="monotone" dataKey="totalCost1M" name="Total TCO" stroke="#2f8365" strokeWidth={3} dot={false} />
                <ReferenceLine x={currentStat.batchSize} stroke="#928e8e" strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </GlassCard>
    </div>
  );
}
