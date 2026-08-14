import type { CSSProperties, ChangeEvent } from 'react';
import InfoPopover from './InfoPopover';

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
  comment?: string;
  logScale?: boolean;
  conceptId?: string;
  accent?: string;
}

export default function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange,
  unit = '',
  comment,
  logScale,
  conceptId,
  accent = 'var(--color-accent)',
}: SliderControlProps) {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const raw = parseFloat(e.target.value);
    if (logScale) {
      const minLog = Math.log10(min);
      const maxLog = Math.log10(max);
      const valLog = minLog + (raw / 100) * (maxLog - minLog);
      onChange(Math.round(Math.pow(10, valLog) / step) * step);
    } else {
      onChange(raw);
    }
  };

  const sliderValue = (() => {
    if (logScale) {
      const minLog = Math.log10(min);
      const maxLog = Math.log10(max);
      const valLog = Math.log10(value);
      return ((valLog - minLog) / (maxLog - minLog)) * 100;
    }
    return value;
  })();

  const bgStyle: CSSProperties = {
    background: `linear-gradient(90deg, ${accent}, ${accent}99)`,
  };

  return (
    <div>
      <div className="flex justify-between items-end mb-1.5 gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <label className="text-xs font-medium text-slate-600 truncate">{label}</label>
          {conceptId && <InfoPopover conceptId={conceptId} iconSize={13} />}
        </div>
        <div className="glass-chip px-2 py-0.5 shrink-0">
          <span className="font-mono font-semibold text-[12px] text-slate-800">
            {value.toLocaleString()} <span className="text-[10px] text-slate-400">{unit}</span>
          </span>
        </div>
      </div>
      <div className="relative flex items-center">
        <input
          type="range"
          min={logScale ? 0 : min}
          max={logScale ? 100 : max}
          step={logScale ? 0.1 : step}
          value={sliderValue}
          onChange={handleChange}
          style={bgStyle}
          className="glass-slider w-full"
        />
      </div>
      {comment && <div className="text-[10px] text-slate-400 mt-1 text-right">{comment}</div>}
      {logScale && <div className="flex justify-between text-[9px] text-slate-300 mt-1 px-0.5"><span>{min.toLocaleString()}</span><span>{max.toLocaleString()}</span></div>}
    </div>
  );
}
