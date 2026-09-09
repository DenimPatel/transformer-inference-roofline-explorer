import LessonKVCache from '../learn/LessonKVCache';
import GoDeeper from '../shell/GoDeeper';
import { DeepKvCache } from '../DeepDive';

/** Where inference memory actually goes. */
export default function KvCache({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonKVCache onComplete={onComplete} />
      <GoDeeper label="The math" hint="cache size, GQA, and the capacity wall">
        <DeepKvCache />
      </GoDeeper>
    </>
  );
}
