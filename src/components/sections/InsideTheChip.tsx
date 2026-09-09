import LessonInsideChip from '../learn/LessonInsideChip';
import GoDeeper from '../shell/GoDeeper';
import { HwTpu, HwSystolic, HwGpu } from '../Hardware';

/** Where a chip's FLOPs number physically comes from. */
export default function InsideTheChip({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonInsideChip onComplete={onComplete} />
      <GoDeeper label="Open the chip up" hint="TPUs, systolic arrays, and how a GPU differs">
        <HwTpu />
        <HwSystolic />
        <HwGpu />
      </GoDeeper>
    </>
  );
}
