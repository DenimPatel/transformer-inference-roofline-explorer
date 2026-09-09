import LessonPrefillGen from '../learn/LessonPrefillGen';
import GoDeeper from '../shell/GoDeeper';
import { DeepPrefillGen } from '../DeepDive';

/** Same arithmetic, two batch sizes, two different limits. */
export default function PrefillVsGeneration({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonPrefillGen onComplete={onComplete} />
      <GoDeeper label="The math" hint="why the crossover sits where it does">
        <DeepPrefillGen />
      </GoDeeper>
    </>
  );
}
