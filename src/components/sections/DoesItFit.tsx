import { useEffect } from 'react';
import { ServingFit } from '../Serving';

/**
 * Part IV closer. Capacity is the one budget that is a wall rather than a
 * rate: a deployment that does not fit does not run slowly, it does not run.
 */
export default function DoesItFit({ onComplete }: { onComplete: () => void }) {
  useEffect(() => { onComplete(); }, [onComplete]);
  return (
    <>
      <p className="text-base text-slate-600 leading-relaxed max-w-[62ch]">
        Compute and bandwidth are rates: run short of either and you are simply slower. Capacity
        is not. This section takes one real model on one real accelerator and asks the question
        that has to be answered before any of the previous four parts matter — does it fit?
      </p>
      <ServingFit />
    </>
  );
}
