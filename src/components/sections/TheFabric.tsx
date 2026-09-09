import LessonNetworkRoofline from '../learn/LessonNetworkRoofline';
import GoDeeper from '../shell/GoDeeper';
import { DeepNetwork } from '../DeepDive';
import { HwNetwork } from '../Hardware';

/** When the interconnect becomes the roof. */
export default function TheFabric({ onComplete }: { onComplete: () => void }) {
  return (
    <>
      <LessonNetworkRoofline onComplete={onComplete} />
      <GoDeeper label="The math" hint="collective costs and the comms roofline">
        <DeepNetwork />
      </GoDeeper>
      <GoDeeper label="Topologies" hint="torus versus switched domain">
        <HwNetwork />
      </GoDeeper>
    </>
  );
}
