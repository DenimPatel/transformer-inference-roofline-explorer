import { useCallback, useEffect, useState } from 'react';

export type ThemePref = 'system' | 'light' | 'dark';

const KEY = 'roofline:theme';
const ORDER: ThemePref[] = ['system', 'light', 'dark'];

function readPref(): ThemePref {
  try {
    const v = window.localStorage.getItem(KEY);
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch { /* private mode: fall through */ }
  return 'system';
}

/** Resolve a preference to the concrete mode and write it to <html>. */
function apply(pref: ThemePref, fade: boolean) {
  const root = document.documentElement;
  const dark = pref === 'dark'
    || (pref === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  if (fade) {
    root.classList.add('theme-fade');
    window.setTimeout(() => root.classList.remove('theme-fade'), 350);
  }
  root.setAttribute('data-theme', dark ? 'dark' : 'light');
  root.setAttribute('data-pref-theme', pref);
  root.style.colorScheme = dark ? 'dark' : 'light';
}

/**
 * The reader's theme choice. `system` follows the operating system and keeps
 * following it while the page is open; an explicit choice is persisted.
 * The inline boot script in index.html has already set the first frame.
 */
export function useTheme() {
  const [pref, setPref] = useState<ThemePref>(readPref);

  useEffect(() => {
    if (pref !== 'system') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => apply('system', false);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [pref]);

  const cycle = useCallback(() => {
    const next = ORDER[(ORDER.indexOf(pref) + 1) % ORDER.length];
    try { window.localStorage.setItem(KEY, next); } catch { /* ignore */ }
    apply(next, true);
    setPref(next);
  }, [pref]);

  return { pref, cycle };
}
