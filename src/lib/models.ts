/**
 * Model profiles are loaded from one JSON file per model under
 * src/data/models/*.json. To host another open-source model, drop a new JSON
 * file in that folder (matching the ModelProfile schema below) and reload.
 */

export interface ModelProfile {
  id: string;
  name: string;
  family: string;
  vendor: string;
  type: 'dense' | 'moe';
  totalParamsB: number;
  activeParamsB: number;
  contextLen: number;    // default context window in tokens
  kvPerTokenKb: number;  // recommended KV cache bytes/token (KB)
  bytesPerParam: number; // default precision (2 = FP16/BF16)
  description?: string;
}

const modules = import.meta.glob('../data/models/*.json', { eager: true }) as Record<string, ModelProfile>;

export const MODEL_PROFILES: ModelProfile[] = Object.values(modules).sort((a, b) =>
  a.name.localeCompare(b.name)
);

export function findModel(id: string): ModelProfile | undefined {
  return MODEL_PROFILES.find((m) => m.id === id);
}
