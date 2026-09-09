import LessonMemory from '../learn/LessonMemory';
import GoDeeper from '../shell/GoDeeper';
import { DeepMemory } from '../DeepDive';
import { HwHierarchy } from '../Hardware';

/** Why staying on-chip is the point of most inference optimisation. */
export default function MemoryHierarchy({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonMemory onComplete={onComplete} />
      <GoDeeper label="The math" hint="the second roofline, and tiling">
        <DeepMemory />
      </GoDeeper>
      <GoDeeper label="Chip by chip" hint="what each accelerator actually publishes">
        <HwHierarchy />
      </GoDeeper>
    </>
  );
}
