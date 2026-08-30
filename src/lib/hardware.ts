/**
 * Hardware profiles are loaded from one JSON file per configuration under
 * src/data/hardware/*.json. To add a new chip, just drop a new JSON file into
 * that folder (matching the HardwareProfile schema below) and reload — Vite's
 * import.meta.glob picks it up automatically and the app rebuilds around it.
 */

export interface HardwareProfile {
  vendor: string;
  id: string;
  tflops: number;
  memBw: number;        // TB/s
  capacity: number;     // GB
  arch: string;
  bytesPerParam: number; // 2 for FP16/BF16, 1 for FP8, 0.5 for FP4
  tdp: number;          // Watts
  price: number;        // Hardware cost (USD)
  color: string;
}

const modules = import.meta.glob('../data/hardware/*.json', { eager: true }) as Record<string, HardwareProfile>;

export const HARDWARE_PROFILES: HardwareProfile[] = Object.values(modules).sort((a, b) =>
  a.id.localeCompare(b.id)
);
