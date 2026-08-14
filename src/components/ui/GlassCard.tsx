import { type ReactNode } from 'react';
import { cn } from '../../lib/utils';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  shimmer?: boolean;
}

export default function GlassCard({ children, className, hover = false, shimmer = false }: GlassCardProps) {
  return (
    <div className={cn('glass-card', hover && 'glass-card-hover', shimmer && 'shimmer overflow-hidden', className)}>
      {children}
    </div>
  );
}
