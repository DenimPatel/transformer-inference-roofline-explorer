import { useRef, useMemo, useCallback } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { cn } from '../../lib/utils';

interface InteractiveRooflineProps {
  peakFlops: number;   // FLOPs/s
  peakBw: number;      // bytes/s
  intensity: number;   // current op arithmetic intensity (FLOPs/B)
  onIntensityChange?: (i: number) => void;
  minI?: number;
  maxI?: number;
  opLabel?: string;
  className?: string;
}

const W = 640;
const H = 340;
const PADX = 52;
const PADY_B = 34;
const PADY_T = 14;

/**
 * A precise, draggable roofline rendered as SVG. Drag the operating point
 * along the log-intensity x-axis to see it cross the ridge.
 */
export default function InteractiveRoofline({
  peakFlops,
  peakBw,
  intensity,
  onIntensityChange,
  minI = 0.05,
  maxI = 200000,
  opLabel = 'Operating point',
  className,
}: InteractiveRooflineProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const Wc = W - PADX - 18;
  const Hc = H - PADY_T - PADY_B;

  const scaleX = (i: number) => {
    const l = (Math.log10(i) - Math.log10(minI)) / (Math.log10(maxI) - Math.log10(minI));
    return PADX + l * Wc;
  };
  const invX = (x: number) => {
    const l = (x - PADX) / Wc;
    return Math.pow(10, Math.log10(minI) + l * (Math.log10(maxI) - Math.log10(minI)));
  };

  const yMin = peakFlops * minI * 0.5;
  const yMax = peakFlops * 1.2;
  const scaleY = (v: number) => {
    const l = (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin));
    return PADY_T + Hc - l * Hc;
  };

  const ridge = peakFlops / peakBw;

  const roof = useMemo(() => {
    const pts = [];
    for (let i = Math.log10(minI); i <= Math.log10(maxI); i += 0.05) {
      const intensity = Math.pow(10, i);
      const achievable = Math.min(peakBw * intensity, peakFlops);
      pts.push({ x: scaleX(intensity), y: scaleY(achievable) });
    }
    return pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  }, [peakFlops, peakBw, minI, maxI]);

  const ridgeX = scaleX(ridge);
  const clamped = Math.min(intensity, maxI);
  const pointX = scaleX(clamped);
  const pointY = scaleY(Math.min(peakBw * clamped, peakFlops));
  const isMemBound = intensity < ridge;

  const handlePointer = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      if (!onIntensityChange) return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * W;
      const i = invX(Math.max(PADX, Math.min(PADX + Wc, x)));
      onIntensityChange(i);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onIntensityChange]
  );

  return (
    <div className={className}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className={cn('w-full select-none', onIntensityChange && 'cursor-ew-resize touch-none')}
        onPointerDown={onIntensityChange ? handlePointer : undefined}
        onPointerMove={onIntensityChange ? (e) => e.buttons === 1 && handlePointer(e) : undefined}
        role={onIntensityChange ? 'slider' : undefined}
        aria-label="Roofline operating point"
        aria-valuenow={Math.round(intensity)}
      >
        <defs>
          <clipPath id="roofclip">
            <rect x={PADX} y={PADY_T} width={Wc} height={Hc} rx={8} />
          </clipPath>
          <linearGradient id="memgrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f5a623" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#f5a623" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {/* banding panels: memory (left of ridge) vs compute (right) */}
        <rect x={PADX} y={PADY_T} width={ridgeX - PADX} height={Hc} fill="#f5a623" opacity="0.08" />
        <rect x={ridgeX} y={PADY_T} width={PADX + Wc - ridgeX} height={Hc} fill="#22c48b" opacity="0.07" />

        {/* gridlines */}
        {[0.1, 1, 10, 100, 1000, 10000, 100000].map((t) => (
          <g key={t}>
            <line x1={PADX} y1={PADY_T} x2={PADX} y2={PADY_T + Hc} stroke="#64748b" strokeOpacity="0.14" />
            <text x={scaleX(t)} y={H - 8} textAnchor="middle" fontSize="9.5" fill="#94a3b8">
              {t >= 1000 ? `${t / 1000}k` : t}
            </text>
          </g>
        ))}

        {/* y-axis labels */}
        {[0.9, 1e12, 1e14].map((v) => {
          if (v < yMin) v = Math.max(v, yMin * 1.01);
          if (v > yMax) v = yMax * 0.99;
          return (
            <text key={v} x={PADX - 6} y={scaleY(v) + 3} textAnchor="end" fontSize="9.5" fill="#94a3b8">
              {v >= 1e12 ? `${(v / 1e12).toFixed(1)}T` : v.toFixed(1)}
            </text>
          );
        })}

        {/* memory & compute labels */}
        <text x={(PADX + ridgeX) / 2} y={PADY_T + 18} textAnchor="middle" fontSize="10" fontWeight="700" fill="#c08a1d">
          MEMORY BOUND
        </text>
        <text x={(ridgeX + PADX + Wc) / 2} y={PADY_T + 18} textAnchor="middle" fontSize="10" fontWeight="700" fill="#15896a">
          COMPUTE BOUND
        </text>

        {/* roof path */}
        <path
          d={`M ${roof}`}
          fill="none"
          stroke="#5b7cfa"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* ridge line */}
        <line
          x1={ridgeX}
          y1={PADY_T}
          x2={ridgeX}
          y2={PADY_T + Hc}
          stroke="#f25f7d"
          strokeWidth="1.5"
          strokeDasharray="5 4"
        />
        <text x={ridgeX + 4} y={PADY_T + 12} fontSize="9.5" fill="#f25f7d" fontWeight="700">
          ridge
        </text>

        {/* crosshair + point */}
        <line x1={pointX} y1={PADY_T} x2={pointX} y2={PADY_T + Hc} stroke={isMemBound ? '#c08a1d' : '#15896a'} strokeWidth="1" strokeDasharray="3 3" opacity="0.5" />
        <circle cx={pointX} cy={pointY} r="6.5" fill={isMemBound ? '#f5a623' : '#22c48b'} stroke="#fff" strokeWidth="2.5" />
        <circle cx={pointX} cy={pointY} r="13" fill={isMemBound ? '#f5a623' : '#22c48b'} opacity="0.2" />

        {/* formula labels */}
        <text x={PADX + 4} y={H - PADY_B + 22} fontSize="9" fill="#94a3b8">
          x → Arithmetic intensity (FLOPs/B, log)
        </text>
        <text x={12} y={PADY_T + 12} fontSize="9" fill="#94a3b8" transform={`rotate(-90 12 ${PADY_T + 12})`} textAnchor="end">
          Throughput (FLOPs/s, log)
        </text>
      </svg>

      <div className="flex flex-wrap gap-2 mt-2 text-xs">
        <span className="glass-chip px-2 py-1 text-slate-600">
          {opLabel} intensity:{' '}
          <span className="font-mono font-bold">{intensity.toFixed(1)} FLOPs/B</span>
        </span>
        <span className="glass-chip px-2 py-1 text-slate-600">
          Ridge: <span className="font-mono font-bold">{ridge.toFixed(1)}</span> FLOPs/B
        </span>
        <span
          className={cn(
            'px-2 py-1 rounded-full font-bold',
            isMemBound ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'
          )}
        >
          {isMemBound ? 'Memory-bound (below ridge)' : 'Compute-bound (at/over ridge)'}
        </span>
        {onIntensityChange && (
          <span className="ml-auto text-[11px] text-slate-400 self-center">Drag the point ⟷</span>
        )}
      </div>
    </div>
  );
}
