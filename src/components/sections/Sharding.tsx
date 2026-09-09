import LessonSharding from '../learn/LessonSharding';
import GoDeeper from '../shell/GoDeeper';
import { HwScalingRules } from '../Hardware';

/** Cutting a model across chips, and what each cut costs. */
export default function Sharding({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonSharding onComplete={onComplete} />
      <GoDeeper label="Rules of thumb" hint="when each strategy runs out of road">
        <HwScalingRules />
      </GoDeeper>
    </>
  );
}
