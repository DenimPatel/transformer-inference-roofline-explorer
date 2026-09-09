import { useEffect, useState, useCallback } from 'react';

/**
 * Minimal hash router.
 *
 * Hash — not history — on purpose: the site is served from a GitHub Pages
 * sub-path with no SPA 404 rewrite, so a real path would 404 on refresh.
 *
 * Shape: #/<sectionId>?<config>   e.g. #/the-roofline?hw=H100%20SXM5&b=256
 * The query part after the hash is owned by ConfigContext; this module only
 * reads and writes the path part and leaves the rest untouched.
 */

export function parseHash(hash: string): { section: string; query: URLSearchParams } {
  const raw = hash.replace(/^#\/?/, '');
  const qIndex = raw.indexOf('?');
  const path = qIndex === -1 ? raw : raw.slice(0, qIndex);
  const query = new URLSearchParams(qIndex === -1 ? '' : raw.slice(qIndex + 1));
  return { section: path || '', query };
}

/** Rebuild a hash from its two halves, dropping an empty query. */
export function buildHash(section: string, query: URLSearchParams): string {
  const q = query.toString();
  return `#/${section}${q ? `?${q}` : ''}`;
}

/** Current section id, plus a setter that preserves the config query string. */
export function useRoute(fallback: string) {
  const read = useCallback(
    () => parseHash(window.location.hash).section || fallback,
    [fallback],
  );
  const [section, setSectionState] = useState<string>(read);

  useEffect(() => {
    const onChange = () => setSectionState(read());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, [read]);

  const navigate = useCallback((id: string) => {
    const { query } = parseHash(window.location.hash);
    window.location.hash = buildHash(id, query);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  return [section, navigate] as const;
}
