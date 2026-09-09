import LessonAttention from '../learn/LessonAttention';
import GoDeeper from '../shell/GoDeeper';
import { DeepAttention } from '../DeepDive';

/** Attention's flip from compute-bound to memory-bound. */
export default function Attention({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonAttention onComplete={onComplete} />
      <GoDeeper label="The math" hint="attention FLOPs, and when they stop mattering">
        <DeepAttention />
      </GoDeeper>
    </>
  );
}
