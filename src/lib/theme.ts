/**
 * Chart palette for the "instrument" theme, shared with the interactive
 * courses site.
 *
 * Every value is a CSS `var()` reference, not a literal. Recharts and SVG
 * attributes accept `var()`, so charts resolve the token at paint time and
 * follow the light/dark switch with no re-render. The CSS custom properties
 * in `src/index.css` are the single source of truth for the values.
 *
 * Because these are not hex strings, do not append an alpha suffix
 * (`${color}22`); use `tint()` instead.
 */

export const INK = 'var(--c-fg)';
export const PAPER = 'var(--c-bg)';
export const SURFACE = 'var(--c-surface)';

/** Cool neutral ramp (the `slate` utilities). Inverted in dark mode. */
export const NEUTRAL = {
  100: 'var(--color-slate-100)',
  200: 'var(--color-slate-200)',
  300: 'var(--color-slate-300)',
  400: 'var(--color-slate-400)',
  500: 'var(--color-slate-500)',
  600: 'var(--color-slate-600)',
  700: 'var(--color-slate-700)',
  800: 'var(--color-slate-800)',
  900: 'var(--color-slate-900)',
} as const;

/** Primary hue: azure. Reserved for interaction and the focal series. */
export const ACCENT = {
  100: 'var(--color-accent-100)',
  200: 'var(--color-accent-200)',
  300: 'var(--color-accent-300)',
  400: 'var(--color-accent-400)',
  500: 'var(--color-accent-500)',
  600: 'var(--color-accent-600)',
  700: 'var(--color-accent-700)',
  800: 'var(--color-accent-800)',
  900: 'var(--color-accent-900)',
} as const;

/** Second series hue: teal. */
export const ACCENT_2 = {
  100: 'var(--color-accent-2-100)',
  200: 'var(--color-accent-2-200)',
  300: 'var(--color-accent-2-300)',
  400: 'var(--color-accent-2-400)',
  500: 'var(--color-accent-2-500)',
  600: 'var(--color-accent-2-600)',
  700: 'var(--color-accent-2-700)',
  800: 'var(--color-accent-2-800)',
  900: 'var(--color-accent-2-900)',
} as const;

/** The categorical series slots; fixed across modes. */
export const SERIES = {
  azure: 'var(--chart-series-1)',
  teal: 'var(--chart-series-2)',
  amber: 'var(--chart-series-3)',
  violet: 'var(--chart-series-4)',
  crimson: 'var(--chart-series-5)',
  lime: 'var(--chart-series-6)',
  cyan: 'var(--chart-series-7)',
} as const;

/** Compute-side and memory-side of the roofline. */
export const COMPUTE = 'var(--chart-compute)';
export const MEMORY = 'var(--chart-memory)';
export const GREEN = COMPUTE;
export const MUSTARD = SERIES.amber;
export const PLUM = SERIES.violet;

/** A translucent version of any colour token, including a `var()`. */
export const tint = (color: string, pct: number) =>
  `color-mix(in srgb, ${color} ${pct}%, transparent)`;

/** Ordered series colours for charts drawn from app state. */
export const SERIES_COLORS = [
  SERIES.azure,
  SERIES.teal,
  SERIES.amber,
  SERIES.violet,
  SERIES.crimson,
  SERIES.lime,
  SERIES.cyan,
  NEUTRAL[600],
] as const;

/** Named roles used by the chart components. */
export const CHART = {
  compute: COMPUTE,
  memory: MEMORY,
  ridge: MEMORY,
  accent: ACCENT[700],
  accentSoft: ACCENT[400],
  sky: ACCENT[500],
  amber: MUSTARD,
  violet: PLUM,
  slate: NEUTRAL[400],
  rose: MEMORY,
  ink: INK,
} as const;
