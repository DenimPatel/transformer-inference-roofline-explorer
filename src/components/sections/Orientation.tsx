import { Fragment, useEffect } from 'react';
import { ArrowRight, Cpu, Gauge, HardDrive } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import KpiCard from '../ui/KpiCard';
import { useConfig } from '../../state/ConfigContext';
import { MODEL_PROFILES } from '../../lib/models';

/**
 * The front door. One headline number, the three budgets that produce it, and
 * an honest statement of what a reader needs to know before starting.
 */
export default function Orientation({ onComplete }: { onComplete: () => void }) {
  const {
    activeProfileId, activeModelId, units, currentStat, currentBatchSize, maxTokensPerSec,
  } = useConfig();

  // Reading the front door counts as having started.
  useEffect(() => { onComplete(); }, [onComplete]);

  const model = MODEL_PROFILES.find((m) => m.id === activeModelId);
  const modelName = model?.name ?? activeModelId;

  return (
    <>
      <GlassCard className="p-5 sm:p-6 space-y-4">
        <p className="text-sm text-slate-500">
          Right now you have <strong className="text-slate-700">{modelName}</strong> running on{' '}
          <strong className="text-slate-700">{activeProfileId}</strong> at a batch size of{' '}
          <strong className="text-slate-700">{currentBatchSize}</strong>. On that setup:
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <KpiCard
            label="Cost per million tokens"
            value={`$${currentStat.totalCost1M.toFixed(2)}`}
            colorStyle={{ color: 'var(--color-accent-2)' }}
            subValue="electricity + amortised hardware"
            conceptId="tco"
          />
          <KpiCard
            label="Throughput"
            value={Math.round(currentStat.throughput).toLocaleString()}
            unit="tok/s"
            subValue={`ceiling ${Math.round(maxTokensPerSec).toLocaleString()} tok/s`}
            conceptId="latency-throughput"
          />
          <KpiCard
            label="Latency per step"
            value={currentStat.latency.toFixed(1)}
            unit="ms"
            subValue={currentStat.isMemoryBound ? 'memory-bound' : 'compute-bound'}
            conceptId={currentStat.isMemoryBound ? 'memory-bound' : 'compute-bound'}
          />
          <KpiCard
            label="The ridge"
            value={Math.round(units.hardwareRatio)}
            unit="FLOPs/byte"
            subValue="where the two limits meet"
            conceptId="ridge-point"
          />
        </div>
        <p className="text-sm text-slate-600 leading-relaxed">
          Every one of those numbers moves when you change the configuration in the panel.
          The rest of this site explains why each of them is the number it is.
        </p>
      </GlassCard>

      <div className="grid sm:grid-cols-3 gap-3 pt-2">
        {[
          { icon: Cpu, title: 'Arithmetic', body: 'How fast the chip multiplies. Measured in FLOPs per second.' },
          { icon: Gauge, title: 'Bandwidth', body: 'How fast it can fetch the numbers to multiply. Bytes per second.' },
          { icon: HardDrive, title: 'Capacity', body: 'How much it can hold at once. Bytes. This one is a hard wall.' },
        ].map(({ icon: Icon, title, body }) => (
          <Fragment key={title}>
          <GlassCard className="p-4 space-y-1.5">
            <Icon style={{ width: 18, height: 18, color: 'var(--color-accent)' }} />
            <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
            <p className="text-sm text-slate-500 leading-relaxed">{body}</p>
          </GlassCard>
          </Fragment>
        ))}
      </div>

      <p className="text-base text-slate-600 leading-relaxed max-w-[62ch] pt-2">
        A chip gives you those three budgets. A model spends them in a fixed proportion.
        Almost everything interesting about inference — why the first token is fast and the
        rest are slow, why batching helps up to a point, why a bigger context is expensive,
        why quantization buys throughput — falls out of comparing the proportion the model
        spends against the proportion the chip supplies.
      </p>

      <GlassCard className="p-5 space-y-2">
        <h3 className="text-sm font-semibold text-slate-800">What you need to know already</h3>
        <ul className="text-sm text-slate-600 leading-relaxed space-y-1.5 list-disc pl-5">
          <li>Matrix multiplication, and roughly what it costs.</li>
          <li>That a transformer is a stack of matrix multiplications with attention between them.</li>
          <li>Powers of two, and reading a log-scaled axis. Every chart here carries a plain-English reading.</li>
        </ul>
        <p className="text-sm text-slate-500 pt-1">
          You do not need any computer-architecture background. Part I builds it.
        </p>
      </GlassCard>

      <p className="flex items-center gap-2 text-sm text-slate-500 pt-2">
        <ArrowRight style={{ width: 15, height: 15 }} />
        Next: what an accelerator actually gives you.
      </p>
    </>
  );
}
