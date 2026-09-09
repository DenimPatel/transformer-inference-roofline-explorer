import { useEffect } from 'react';
import { DeepMoe } from '../DeepDive';

/**
 * Part II closer. A mixture-of-experts model separates the parameters it must
 * hold from the parameters it actually uses — which breaks the assumption that
 * "bigger model" means "more arithmetic per token".
 */
export default function Moe({ onComplete }: { onComplete: () => void }) {
  useEffect(() => { onComplete(); }, [onComplete]);
  return (
    <>
      <p className="text-base text-slate-600 leading-relaxed max-w-[62ch]">
        Until now &ldquo;the model&rdquo; has been one number of parameters. Mixture-of-experts
        models split that in two: a large <strong>resident</strong> set that must sit in memory,
        and a much smaller <strong>active</strong> set that participates in any given token. The
        gap between them is why the configuration panel has two parameter sliders, and why the
        two move the roofline in opposite directions.
      </p>
      <DeepMoe />
    </>
  );
}
