import LessonGpuToken from '../learn/LessonGpuToken';

/**
 * Part II, right after "One token, end to end": the same forward pass, now
 * followed through the physical parts of one H100 (host, PCIe, HBM, L2, SMs,
 * tensor cores) for Qwen3.8-27B or Qwen3-8B, as a step-through animation.
 */
export default function OneTokenOnGpu({ onComplete }: { onComplete: () => void }) {
  return <LessonGpuToken onComplete={onComplete} />;
}
