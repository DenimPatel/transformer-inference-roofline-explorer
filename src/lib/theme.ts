/**
 * Broadsheet palette — shared with the companion site at
 * github.com/DenimPatel/AI. Near-black serif on paper white, with cyan and
 * magenta used sparingly as spot color.
 *
 * The CSS custom properties in `src/index.css` are the source of truth for
 * anything rendered by CSS; these constants exist for the places that must
 * hand a literal color to a chart library (Recharts) or an SVG attribute.
 */

export const INK = '#201e1d';
export const PAPER = '#f3f2f2';
export const SURFACE = '#eae9e9';

/** Warm newsprint greys (the remapped "slate" ramp). */
export const NEUTRAL = {
  100: '#f8f4f4',
  200: '#eae7e7',
  300: '#d7d3d3',
  400: '#928e8e',
  500: '#7d7979',
  600: '#605d5d',
  700: '#444141',
  800: '#2d2b2b',
  900: '#201e1d',
} as const;

/** Primary spot color: cyan. */
export const ACCENT = {
  100: '#e9f8ff',
  200: '#cbeeff',
  300: '#99e0ff',
  400: '#62c5ee',
  500: '#38a6cf',
  600: '#1186ac',
  700: '#006786',
  800: '#004961',
  900: '#0a303e',
} as const;

/** Secondary spot color: magenta. */
export const ACCENT_2 = {
  100: '#fff1f4',
  200: '#ffdee6',
  300: '#ffc0d0',
  400: '#ff90b1',
  500: '#ff458e',
  600: '#d6006c',
  700: '#aa0b56',
  800: '#790e3d',
  900: '#4b1528',
} as const;

/** Muted semantic hues, kept at newsprint saturation. */
export const GREEN = '#2f8365';
export const MUSTARD = '#c8963a';
export const PLUM = '#7b4b90';

/**
 * Ordered series colors for charts drawn from app state (rather than from a
 * vendor's own brand color, which the hardware profiles carry in their data).
 * Hue count is deliberately small; separation comes from lightness.
 */
export const SERIES_COLORS = [
  ACCENT[700],
  ACCENT_2[600],
  GREEN,
  MUSTARD,
  ACCENT[400],
  PLUM,
  NEUTRAL[600],
  ACCENT_2[400],
] as const;

/**
 * Named roles used by the chart components. Recharts wants literal colors, so
 * these mirror the CSS custom properties rather than reading them.
 */
export const CHART = {
  compute: GREEN,
  memory: ACCENT_2[500],
  ridge: ACCENT_2[600],
  accent: ACCENT[700],
  accentSoft: ACCENT[400],
  sky: ACCENT[500],
  amber: MUSTARD,
  violet: PLUM,
  slate: NEUTRAL[400],
  rose: ACCENT_2[500],
  ink: INK,
} as const;
