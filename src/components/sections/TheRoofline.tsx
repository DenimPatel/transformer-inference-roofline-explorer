import LessonRoofline from '../learn/LessonRoofline';
import OperatingPoint from '../lab/OperatingPoint';

/**
 * The roofline, then the reader's own configuration placed on it. The second
 * half used to be the Interactive Lab tab, three clicks away from the
 * explanation it illustrates.
 */
export default function TheRoofline({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonRoofline onComplete={onComplete} />
      <div className="pt-2">
        <h3 className="text-xl font-bold text-slate-800 mb-3">Your configuration, on this roof</h3>
        <OperatingPoint />
      </div>
    </>
  );
}
