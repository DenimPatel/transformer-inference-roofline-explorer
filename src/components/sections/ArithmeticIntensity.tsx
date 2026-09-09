import LessonIntensity from '../learn/LessonIntensity';
import GoDeeper from '../shell/GoDeeper';
import { DeepIntensity, DeepMatmul } from '../DeepDive';

/** The ratio that decides which budget you run out of first. */
export default function ArithmeticIntensity({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonIntensity onComplete={onComplete} />
      <GoDeeper label="The math" hint="deriving intensity from first principles">
        <DeepIntensity />
      </GoDeeper>
      <GoDeeper label="Worked example: a matmul" hint="the B rule, with the algebra">
        <DeepMatmul />
      </GoDeeper>
    </>
  );
}
