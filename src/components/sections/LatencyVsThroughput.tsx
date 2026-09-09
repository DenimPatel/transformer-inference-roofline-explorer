import LessonPareto from '../learn/LessonPareto';
import GoDeeper from '../shell/GoDeeper';
import { DeepLatency } from '../DeepDive';

/** The trade you cannot escape. */
export default function LatencyVsThroughput({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonPareto onComplete={onComplete} />
      <GoDeeper label="The math" hint="the Pareto frontier, drawn">
        <DeepLatency />
      </GoDeeper>
    </>
  );
}
