import LessonQuant from '../learn/LessonQuant';
import GoDeeper from '../shell/GoDeeper';
import { DeepQuant } from '../DeepDive';

/** Fewer bits per weight moves the ridge. */
export default function Quantization({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonQuant onComplete={onComplete} />
      <GoDeeper label="The math" hint="the β rule, and when int8 stops helping">
        <DeepQuant />
      </GoDeeper>
    </>
  );
}
