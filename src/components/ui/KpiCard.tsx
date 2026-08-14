import { motion } from 'motion/react';
import type { CSSProperties, ReactNode } from 'react';
import { cn } from '../../lib/utils';
import InfoPopover from './InfoPopover';

interface KpiCardProps {
  label: string;
  value: string | number;
  unit?: string;
  colorClass?: string;
  colorStyle?: CSSProperties;
  subValue?: string;
  info?: string;
  conceptId?: string;
  icon?: ReactNode;
}

export default function KpiCard({
  label,
  value,
  unit,
  colorClass = 'text-slate-800',
  colorStyle,
  subValue,
  info,
  conceptId,
  icon,
}: KpiCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="glass-card glass-card-hover p-5 flex flex-col justify-between items-start relative"
    >
      <div className="flex justify-between w-full items-start mb-2 gap-2">
        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-1.5 min-w-0">
          {icon && <span className="text-[var(--color-accent)] shrink-0">{icon}</span>}
          <span className="truncate">{label}</span>
        </div>
        {(info || conceptId) && <InfoPopover conceptId={conceptId} title={conceptId ? undefined : label} summary={conceptId ? undefined : info} body={conceptId ? undefined : (info ? [info] : undefined)} className="text-slate-300" />}
      </div>
      <div className="flex items-baseline gap-1.5">
        <motion.div
          key={String(value)}
          initial={{ opacity: 0.4 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className={cn('text-2xl font-mono font-bold tracking-tight tabular-nums', colorClass)}
          style={colorStyle}
        >
          {value}
        </motion.div>
        {unit && <div className="text-sm text-slate-400">{unit}</div>}
      </div>
      {subValue && <div className="text-[10px] text-slate-400 mt-1 uppercase tracking-wider">{subValue}</div>}
    </motion.div>
  );
}
