import LessonServing from '../learn/LessonServing';
import GoDeeper from '../shell/GoDeeper';
import { ServingPractice } from '../Serving';

/** Four answers to one problem: the chip is idle. */
export default function ServingInPractice({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonServing onComplete={onComplete} />
      <GoDeeper label="A worked request" hint="per-request FLOPs and memory, phase by phase">
        <ServingPractice />
      </GoDeeper>
    </>
  );
}
