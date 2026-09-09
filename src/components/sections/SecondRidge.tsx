import { useEffect } from 'react';
import { DeepSecondRidge } from '../DeepDive';

/**
 * Part III closer. The matrix unit is not the only unit on the chip, and the
 * vector unit's far lower peak gives elementwise work a ridge of its own —
 * which is why a softmax or a layernorm is memory-bound at any batch size.
 */
export default function SecondRidge({ onComplete }: { onComplete: () => void }) {
  useEffect(() => { onComplete(); }, [onComplete]);
  return (
    <>
      <p className="text-base text-slate-600 leading-relaxed max-w-[62ch]">
        Everything so far has measured operations against the matrix unit&rsquo;s peak. But a
        transformer is not only matmuls: softmax, layer norm, activation functions and the
        residual adds all run on a different, much smaller unit. Measured against{' '}
        <em>its</em> peak, the ridge moves — and these operations turn out to sit permanently
        on the wrong side of it.
      </p>
      <DeepSecondRidge />
    </>
  );
}
