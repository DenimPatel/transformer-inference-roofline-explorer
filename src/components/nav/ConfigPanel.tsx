import type { ReactNode } from 'react';
import { Server, DollarSign, Layers, Activity, Cpu } from 'lucide-react';
import { cn } from '../../lib/utils';
import { HARDWARE_PROFILES } from '../../lib/hardware';
import { MODEL_PROFILES, findModel } from '../../lib/models';
import SliderControl from '../ui/SliderControl';
import InfoPopover from '../ui/InfoPopover';
import { useConfig, ECONOMICS_REGIONS } from '../../state/ConfigContext';

function Group({ title, icon, children, conceptId }: {
  title: string; icon: ReactNode; children: ReactNode; conceptId?: string;
}) {
  return (
    <div className="glass-card p-5">
      <h2 className="flex items-center text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-4">
        <span className="mr-2 p-1.5 rounded-lg bg-white">{icon}</span> {title}
        {conceptId && <span className="ml-auto normal-case"><InfoPopover conceptId={conceptId} iconSize={14} /></span>}
      </h2>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

/**
 * The configuration that every section reads from.
 *
 * These controls used to be the "Interactive Lab" tab, which meant the numbers
 * a reader chose were invisible to every explanation on the site. Docking them
 * makes the whole curriculum read against one setup.
 */
export default function ConfigPanel({ showComparison = false }: { showComparison?: boolean }) {
  const c = useConfig();

  return (
    <div className="space-y-5">
      <Group title="Hardware" icon={<Server className="w-4 h-4 text-[var(--color-accent)]" />} conceptId="roofline">
        <div>
          <label className="text-xs font-semibold text-slate-500 mb-1.5 block uppercase tracking-wider flex items-center gap-1.5">
            Preset <InfoPopover conceptId="ridge-point" iconSize={13} />
          </label>
          <select className="glass-input w-full text-sm py-2 px-3 text-slate-800 font-medium"
            value={c.activeProfileId} onChange={(e) => c.selectProfile(e.target.value)}>
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
        <SliderControl label="Compute" value={c.flopsTera} min={10} max={100000} step={1} onChange={c.setFlopsTera} unit="TFLOP/s" conceptId="flops" />
        <SliderControl label="Memory Bandwidth" value={c.memBwTera} min={0.5} max={200.0} step={0.1} onChange={c.setMemBwTera} unit="TB/s" conceptId="bandwidth" />
        <SliderControl label="Memory Capacity" value={c.memCapGb} min={0.5} max={512} step={0.5} onChange={c.setMemCapGb} unit="GB" />
        <SliderControl label="Thermal Design Power" value={c.tdpWatts} min={50} max={3000} step={10} onChange={c.setTdpWatts} unit="W" />
        <SliderControl label="Est. System Cost" value={c.hardwarePrice} min={1000} max={400000} step={1000} onChange={c.setHardwarePrice} unit="$" />
      </Group>

      <Group title="Model" icon={<Layers className="w-4 h-4 text-blue-500" />} conceptId="matmul-intensity">
        <div>
          <label className="text-xs font-semibold text-slate-500 mb-1.5 block uppercase tracking-wider flex items-center gap-1.5">
            Preset <InfoPopover conceptId="kv-cache" iconSize={13} />
          </label>
          <select className="glass-input w-full text-sm py-2 px-3 text-slate-800 font-medium"
            value={c.activeModelId} onChange={(e) => c.selectModel(e.target.value)}>
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
          {findModel(c.activeModelId)?.description && (
            <p className="mt-2 text-[11px] leading-snug text-slate-500">{findModel(c.activeModelId)?.description}</p>
          )}
        </div>
        <SliderControl label="Total Parameters" value={c.totalParamsB} min={1} max={10000} step={1} onChange={c.setTotalParamsB} unit="B" />
        <SliderControl label="Active Parameters" value={c.activeParamsB} min={1} max={c.totalParamsB} step={1} onChange={c.setActiveParamsB} unit="B"
          comment={`Sparsity: ${(c.totalParamsB / c.activeParamsB).toFixed(1)}x`} conceptId="kv-cache" />
        <div className="pt-2">
          <label className="text-xs text-slate-600 mb-1.5 block">Parameter Precision</label>
          <div className="flex glass p-1 rounded-xl">
            {([{ v: 0.5, l: 'FP4' }, { v: 1, l: 'FP8' }, { v: 2, l: 'FP16/BF16' }] as const).map((o) => (
              <button key={o.v} type="button" onClick={() => c.setBytesPerParam(o.v)}
                className={cn('flex-1 text-xs py-1.5 rounded-lg transition-colors cursor-pointer',
                  c.bytesPerParam === o.v ? 'bg-white shadow text-slate-900 font-bold' : 'text-slate-500')}>
                {o.l}
              </button>
            ))}
          </div>
        </div>
        <SliderControl label="Context Length" value={c.contextLen} min={128} max={5000000} step={128} onChange={c.setContextLen} unit="tkns" logScale conceptId="kv-cache" />
        <SliderControl label="KV Size / Token" value={c.bytesPerTokenKb} min={16} max={2048} step={16} onChange={c.setBytesPerTokenKb} unit="KB" />
      </Group>

      <Group title="Operating Point" icon={<Activity className="w-4 h-4 text-emerald-500" />} conceptId="latency-throughput">
        <SliderControl label="Batch Size" value={c.currentBatchSize} min={1} max={32768} step={1} onChange={c.setCurrentBatchSize} unit="seqs" logScale conceptId="critical-batch" />
      </Group>

      <Group title="Datacenter Economics" icon={<DollarSign className="w-4 h-4 text-emerald-500" />} conceptId="tco">
        <div>
          <label className="text-xs font-semibold text-slate-500 mb-1.5 block uppercase tracking-wider">Geography / Power</label>
          <select className="glass-input w-full text-sm py-2 px-3 text-slate-800 font-medium"
            value={c.regionId} onChange={(e) => c.selectRegion(e.target.value)}>
            {ECONOMICS_REGIONS.map((r) => <option key={r.id} value={r.id}>{r.id} ({r.desc})</option>)}
          </select>
        </div>
        <SliderControl label="Electricity Price" value={c.priceKwh} min={0.01} max={0.50} step={0.01} onChange={c.setPriceKwh} unit="$/kWh" />
        <SliderControl label="PUE (Efficiency)" value={c.pue} min={1.0} max={2.0} step={0.01} onChange={c.setPue} unit=" "
          comment={c.isLiquid ? 'Liquid cooling applied automatically (max 1.05).' : ''} />
        <SliderControl label="Hardware Utilization" value={c.utilization} min={10} max={100} step={1} onChange={c.setUtilization} unit="%" />
        <SliderControl label="Amortization Period" value={c.amortizationYears} min={1} max={10} step={1} onChange={c.setAmortizationYears} unit="Yrs" />
      </Group>

      {showComparison && (
        <Group title="Hardware Selection" icon={<Cpu className="w-4 h-4 text-[var(--color-accent)]" />}>
          <p className="text-xs text-slate-500 mb-3">Profiles to compare on this section:</p>
          <div className="space-y-1 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
            {HARDWARE_PROFILES.map((profile) => (
              <label key={profile.id} className="flex items-center space-x-2 py-1.5 px-2 hover:bg-slate-100 cursor-pointer rounded-lg">
                <input type="checkbox" className="rounded border-slate-300 text-[var(--color-accent)] focus:ring-[var(--color-accent)]"
                  checked={c.selectedProfiles.includes(profile.id)} onChange={() => c.toggleProfile(profile.id)} />
                <span className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: profile.color }} />
                  <span className="text-sm font-medium text-slate-700">{profile.id}</span>
                </span>
              </label>
            ))}
          </div>
        </Group>
      )}
    </div>
  );
}
