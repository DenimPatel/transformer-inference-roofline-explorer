import { useSyncExternalStore } from 'react';

/**
 * Reader preferences — the same contract as the interactive-courses site's
 * reading-settings panel. One versioned localStorage envelope, validated key
 * by key, resolved to `data-*` attributes and `--pref-*` custom properties on
 * <html>. The inline boot script in index.html duplicates `resolve()` so the
 * first frame is already correct; keep the two tables in step.
 */

export type Theme = 'system' | 'light' | 'dark';
export type Measure = 'narrow' | 'normal' | 'wide';
export type Leading = 'tight' | 'normal' | 'relaxed';
export type Density = 'compact' | 'normal' | 'spacious';
export type Motion = 'system' | 'reduced' | 'full';

export interface Prefs {
  theme: Theme;
  text: 90 | 100 | 115 | 130;
  measure: Measure;
  leading: Leading;
  density: Density;
  motion: Motion;
  chartGrid: boolean;
  focusMode: boolean;
}

export const DEFAULTS: Prefs = {
  theme: 'system',
  text: 100,
  measure: 'normal',
  leading: 'normal',
  density: 'normal',
  motion: 'system',
  chartGrid: true,
  focusMode: false,
};

/** Preference value -> the CSS value it resolves to. `normal` is the look the
 *  site had before preferences existed. */
export const TEXT_SCALE = { 90: 0.9, 100: 1, 115: 1.15, 130: 1.3 } as const;
export const MEASURE = { narrow: '54ch', normal: '62ch', wide: '76ch' } as const;
export const LINE_HEIGHT = { tight: '1.4', normal: '1.6', relaxed: '1.8' } as const;
export const SPACE_SCALE = { compact: '0.85', normal: '1', spacious: '1.15' } as const;

const KEY = 'roofline:prefs';
const LEGACY_THEME_KEY = 'roofline:theme';

const OPTIONS: { [K in keyof Prefs]: readonly Prefs[K][] } = {
  theme: ['system', 'light', 'dark'],
  text: [90, 100, 115, 130],
  measure: ['narrow', 'normal', 'wide'],
  leading: ['tight', 'normal', 'relaxed'],
  density: ['compact', 'normal', 'spacious'],
  motion: ['system', 'reduced', 'full'],
  chartGrid: [true, false],
  focusMode: [true, false],
};

function load(): Prefs {
  const prefs = { ...DEFAULTS };
  try {
    const raw = window.localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      const src = parsed && typeof parsed === 'object' && parsed.prefs ? parsed.prefs : parsed;
      if (src && typeof src === 'object') {
        for (const k of Object.keys(DEFAULTS) as (keyof Prefs)[]) {
          if ((OPTIONS[k] as readonly unknown[]).includes(src[k])) (prefs[k] as unknown) = src[k];
        }
      }
    } else {
      // Migrate the earlier theme-only key.
      const t = window.localStorage.getItem(LEGACY_THEME_KEY);
      if (t === 'light' || t === 'dark' || t === 'system') prefs.theme = t;
    }
  } catch { /* storage unavailable or corrupt: defaults */ }
  return prefs;
}

function save(prefs: Prefs) {
  try { window.localStorage.setItem(KEY, JSON.stringify({ v: 1, prefs })); } catch { /* ignore */ }
}

const mq = (q: string) => window.matchMedia(q).matches;

/** Write the DOM contract for a preference set. Idempotent. */
export function apply(prefs: Prefs, fade = false) {
  const root = document.documentElement;
  const dark = prefs.theme === 'dark' || (prefs.theme === 'system' && mq('(prefers-color-scheme: dark)'));
  const still = prefs.motion === 'reduced' || (prefs.motion === 'system' && mq('(prefers-reduced-motion: reduce)'));

  if (fade && !still) {
    root.classList.add('theme-fade');
    window.setTimeout(() => root.classList.remove('theme-fade'), 350);
  }
  root.setAttribute('data-theme', dark ? 'dark' : 'light');
  root.setAttribute('data-pref-theme', prefs.theme);
  root.setAttribute('data-focus-mode', prefs.focusMode ? 'on' : 'off');
  root.setAttribute('data-chart-grid', prefs.chartGrid ? 'on' : 'off');
  if (still) root.setAttribute('data-pref-motion', 'reduced');
  else root.removeAttribute('data-pref-motion');
  root.style.colorScheme = dark ? 'dark' : 'light';
  root.style.setProperty('--pref-text-scale', String(TEXT_SCALE[prefs.text]));
  root.style.setProperty('--pref-measure', MEASURE[prefs.measure]);
  root.style.setProperty('--pref-line-height', LINE_HEIGHT[prefs.leading]);
  root.style.setProperty('--pref-space', SPACE_SCALE[prefs.density]);
  root.style.setProperty('--pref-motion-scale', still ? '0' : '1');
}

export function countChanged(p: Prefs): number {
  return (Object.keys(DEFAULTS) as (keyof Prefs)[]).filter((k) => p[k] !== DEFAULTS[k]).length;
}

/* ---- a tiny external store ------------------------------------------------ */

let current: Prefs | null = null;
const listeners = new Set<() => void>();

function get(): Prefs {
  if (!current) current = load();
  return current;
}

function emit() { listeners.forEach((l) => l()); }

export function setPref<K extends keyof Prefs>(key: K, value: Prefs[K]) {
  current = { ...get(), [key]: value };
  save(current);
  apply(current, key === 'theme');
  emit();
}

export function resetPrefs() {
  current = { ...DEFAULTS };
  save(current);
  apply(current, true);
  emit();
}

/** Cycle System -> Light -> Dark, for the header button. */
export function cycleTheme() {
  const order = OPTIONS.theme;
  setPref('theme', order[(order.indexOf(get().theme) + 1) % order.length]);
}

let watching = false;
function subscribe(cb: () => void) {
  listeners.add(cb);
  if (!watching) {
    // A System choice keeps following the OS while the page is open.
    watching = true;
    for (const q of ['(prefers-color-scheme: dark)', '(prefers-reduced-motion: reduce)']) {
      window.matchMedia(q).addEventListener('change', () => { apply(get()); emit(); });
    }
  }
  return () => { listeners.delete(cb); };
}

export function usePrefs(): Prefs {
  return useSyncExternalStore(subscribe, get, () => DEFAULTS);
}

/** True when the page should render still (explicit Reduced, or System + OS). */
export function useStill(): boolean {
  const p = usePrefs();
  return p.motion === 'reduced' || (p.motion === 'system' && mq('(prefers-reduced-motion: reduce)'));
}
