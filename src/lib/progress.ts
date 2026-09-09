const LEGACY_KEY = 'roofline-learn-progress';
const KEY = 'roofline-progress-v2';

/**
 * The thirteen lesson ids the old Learn journey stored, mapped onto the
 * section slugs that replaced them, so a returning reader keeps their ticks.
 */
const LEGACY_IDS: Record<string, string> = {
  roofline: 'the-roofline',
  intensity: 'arithmetic-intensity',
  prefillgen: 'prefill-vs-generation',
  pareto: 'latency-vs-throughput',
  kvcache: 'kv-cache',
  cost: 'tco',
  quant: 'quantization',
  memory: 'memory-hierarchy',
  sharding: 'sharding',
  attention: 'attention',
  serving: 'serving-in-practice',
  insidechip: 'inside-the-chip',
  networkroofline: 'the-fabric',
};

export function loadProgress(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as string[];
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (!legacy) return [];
    const migrated = (JSON.parse(legacy) as string[])
      .map((id) => LEGACY_IDS[id])
      .filter(Boolean);
    localStorage.setItem(KEY, JSON.stringify(migrated));
    return migrated;
  } catch {
    return [];
  }
}

export function saveProgress(ids: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(ids));
  } catch {
    /* private browsing — progress is a convenience, not state we depend on */
  }
}
