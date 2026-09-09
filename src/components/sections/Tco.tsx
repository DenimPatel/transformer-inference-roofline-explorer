import LessonCost from '../learn/LessonCost';
import CostCharts from '../lab/CostCharts';

/** Cost of ownership, followed by the live latency and cost curves. */
export default function Tco({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonCost onComplete={onComplete} />
      <div className="pt-2">
        <h3 className="text-xl font-bold text-slate-800 mb-3">Your configuration, priced</h3>
        <CostCharts />
      </div>
    </>
  );
}
