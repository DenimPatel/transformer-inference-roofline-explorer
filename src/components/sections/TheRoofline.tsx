import LessonRoofline from '../learn/LessonRoofline';
import GoDeeper from '../shell/GoDeeper';
import OperatingPoint from '../lab/OperatingPoint';
import { DeepFramework } from '../DeepDive';

/**
 * The roofline, its formal statement, then the reader's own configuration
 * placed on it. That last part used to be the Interactive Lab tab, three
 * clicks away from the explanation it illustrates.
 */
export default function TheRoofline({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonRoofline onComplete={onComplete} />
      <GoDeeper label="The math" hint="the two clocks, stated formally">
        <DeepFramework />
      </GoDeeper>
      <div className="pt-2">
        <h3 className="text-xl font-bold text-slate-800 mb-3">Your configuration, on this roof</h3>
        <OperatingPoint />
      </div>
    </>
  );
}
