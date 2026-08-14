import { motion } from 'motion/react';
import { cn } from '../../lib/utils';

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  hint?: string;
}

interface SegmentedControlProps<T extends string> {
  options: SegmentOption<T>[];
  value: T;
  onChange: (v: T) => void;
  size?: 'sm' | 'md';
  className?: string;
}

export default function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
  className,
}: SegmentedControlProps<T>) {
  return (
    <div className={cn('relative flex glass rounded-xl p-1', className)}>
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={cn(
              'relative flex-1 rounded-lg font-medium transition-colors',
              size === 'sm' ? 'text-xs px-2 py-1.5' : 'text-sm px-3 py-1.5',
              active ? 'text-slate-800' : 'text-slate-500 hover:text-slate-700'
            )}
          >
            {active && (
              <motion.span
                layoutId="segmented-pill"
                className="absolute inset-0 rounded-lg bg-white shadow border border-slate-200/70"
                transition={{ type: 'spring', stiffness: 380, damping: 32 }}
              />
            )}
            <span className="relative z-10 flex items-center justify-center gap-1.5">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
