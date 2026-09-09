import { useEffect } from 'react';
import CompetitiveAnalysis from '../CompetitiveAnalysis';
import { useConfig } from '../../state/ConfigContext';

/**
 * Part V closer. The comparison used to be a top-level tab with no stated
 * purpose; here it is the payoff of the cost section — the same accelerators,
 * ranked against the model the reader has been carrying since Part 0.
 */
export default function ComparingHardware({ onComplete }: { onComplete: () => void }) {
  const { units, rowInputs, selectedProfiles } = useConfig();

  useEffect(() => { onComplete(); }, [onComplete]);

  return (
    <>
      <p className="text-base text-slate-600 leading-relaxed max-w-[62ch]">
        Each accelerator below is evaluated against the model and economics you set in the
        configuration panel. A chip wins here not by having the largest headline FLOPs number
        but by having the ratio that suits this model at this batch size — which is the
        argument the previous four parts were building toward.
      </p>
      <CompetitiveAnalysis units={units} rowInputs={rowInputs} selectedProfiles={selectedProfiles} />
    </>
  );
}
