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

  /**
   * Optional hardware-internals fields, sourced from the scaling book's TPU and
   * GPU chapters (reference/scaling-book/tpus.md, gpus.md). Every consumer must
   * treat these as optional — chips whose internals are not publicly documented
   * (Groq, SambaNova, the hybrid profile) deliberately omit them.
   */
  vectorTflops?: number;   // VPU / CUDA-core (non-matmul) TFLOP/s — the second ridge
  onChipMemMB?: number;    // VMEM on TPUs, SMEM + L2 on GPUs
  onChipBwTBs?: number;    // On-chip scratchpad bandwidth, TB/s
  linkBwGBs?: number;      // ICI / NVLink per chip, bidirectional, GB/s
  scaleOutBwGBs?: number;  // DCN / InfiniBand per chip, GB/s
  hostBwGBs?: number;      // PCIe per chip, GB/s
  topology?: string;       // e.g. "3D torus 16x20x28", "NVLink 8-GPU node"
  domainSize?: number;     // Chips in the tightly-coupled (ICI pod / NVLink) domain
  mfu?: number;            // Realistic fraction of peak FLOPs achieved (0-1)
  computeUnits?: string;   // e.g. "132 SMs x 4 tensor cores", "2 TensorCores x 4 MXUs"
}

/** Fallback MFU by vendor when a profile does not declare one. */
const VENDOR_MFU: Record<string, number> = {
  Google: 0.95,
  NVIDIA: 0.82,
  AMD: 0.75,
  AWS: 0.75,
  Groq: 0.9,
  SambaNova: 0.75,
  Hybrid: 0.8,
};

/** Realistic achieved-FLOPs fraction for a profile. */
export function effectiveMfu(hw: Pick<HardwareProfile, 'vendor' | 'mfu'>): number {
  return hw.mfu ?? VENDOR_MFU[hw.vendor] ?? 0.8;
}

/**
 * Ratio of on-chip (VMEM/SMEM) bandwidth to HBM bandwidth. The scaling book
 * quotes ~22x for TPUs; fall back to that when a profile omits the numbers.
 */
export function onChipBwRatio(hw: Pick<HardwareProfile, 'memBw' | 'onChipBwTBs'>): number {
  if (!hw.onChipBwTBs || !hw.memBw) return 22;
  return hw.onChipBwTBs / hw.memBw;
}

const modules = import.meta.glob('../data/hardware/*.json', { eager: true }) as Record<string, HardwareProfile>;

export const HARDWARE_PROFILES: HardwareProfile[] = Object.values(modules).sort((a, b) =>
  a.id.localeCompare(b.id)
);
