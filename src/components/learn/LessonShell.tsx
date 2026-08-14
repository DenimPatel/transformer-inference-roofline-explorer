import type { ReactNode } from 'react';
import { motion } from 'motion/react';
import { BookOpen } from 'lucide-react';
import GlassCard from '../ui/GlassCard';

interface LessonShellProps {
  number?: number;
  title: string;
  subtitle?: string;
  sourceRef?: string;
  children: ReactNode;
}

export default function LessonShell({ number, title, subtitle, sourceRef, children }: LessonShellProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="space-y-4 pb-10"
    >
      <div className="flex items-start gap-3">
        {number && (
          <span className="flex items-center justify-center w-10 h-10 rounded-2xl text-lg font-bold text-white shrink-0"
            style={{ background: 'linear-gradient(135deg, #5b7cfa, #7f6bf0)' }}>
            {number}
          </span>
        )}
        <div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">{title}</h2>
          {subtitle && <p className="text-sm text-slate-500 mt-0.5">{subtitle}</p>}
          {sourceRef && (
            <p className="text-[11px] text-slate-400 mt-1 flex items-center gap-1.5">
              <BookOpen style={{ width: 12, height: 12 }} /> {sourceRef}
            </p>
          )}
        </div>
      </div>

      {children}
    </motion.div>
  );
}
