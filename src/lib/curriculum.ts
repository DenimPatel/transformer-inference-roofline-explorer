import type { ComponentType } from 'react';

import Orientation from '../components/sections/Orientation';

import LessonInsideChip from '../components/learn/LessonInsideChip';
import LessonMemory from '../components/learn/LessonMemory';
import LessonNetworkRoofline from '../components/learn/LessonNetworkRoofline';
import OneToken from '../components/sections/OneToken';
import LessonAttention from '../components/learn/LessonAttention';
import LessonKVCache from '../components/learn/LessonKVCache';
import LessonIntensity from '../components/learn/LessonIntensity';
import TheRoofline from '../components/sections/TheRoofline';
import LessonPrefillGen from '../components/learn/LessonPrefillGen';
import LessonQuant from '../components/learn/LessonQuant';
import LessonPareto from '../components/learn/LessonPareto';
import LessonSharding from '../components/learn/LessonSharding';
import LessonServing from '../components/learn/LessonServing';
import Tco from '../components/sections/Tco';
import ComparingHardware from '../components/sections/ComparingHardware';

export type SectionComponent = ComponentType<{ onComplete: () => void }>;

export interface Section {
  /** URL slug — stable, human-readable, used as the hash path. */
  id: string;
  title: string;
  /** One-sentence plain-English claim shown before any math. */
  standfirst: string;
  minutes: number;
  /** Concept ids into CONCEPTS (src/lib/concepts.ts) — powers cross-links. */
  concepts: string[];
  Component: SectionComponent;
}

export interface Part {
  id: string;
  /** Roman numeral or label shown above the section title. */
  label: string;
  title: string;
  blurb: string;
  sections: Section[];
}

/**
 * The information architecture of the whole site, in one place.
 *
 * Replaces the seven-tab bar and the parallel thirteen-lesson journey that
 * used to live in Dashboard.tsx and learn/LearnJourney.tsx. One spine, read
 * top to bottom, answering a single question: what does it cost to serve one
 * token, and why?
 *
 * Each concept has exactly one canonical section. Anywhere else that mentions
 * it links here via <ConceptTag id="..." /> rather than re-explaining it.
 */
export const CURRICULUM: Part[] = [
  {
    id: 'start',
    label: 'Part 0',
    title: 'Start here',
    blurb: 'What this site answers, and what you need to know first.',
    sections: [
      {
        id: 'orientation',
        title: 'The question',
        standfirst:
          'Serving a token costs money. Where that money goes is decided by three numbers on a chip and two numbers in a model.',
        minutes: 4,
        concepts: ['flops', 'bandwidth', 'roofline'],
        Component: Orientation,
      },
    ],
  },
  {
    id: 'machine',
    label: 'Part I',
    title: 'The machine',
    blurb: 'What an accelerator actually gives you, and what it charges for.',
    sections: [
      {
        id: 'inside-the-chip',
        title: 'Inside the chip',
        standfirst:
          'Matrix multiplication does not happen in a general-purpose core. It happens in a dedicated grid of multipliers, and that grid has its own rules.',
        minutes: 7,
        concepts: ['systolic-array', 'tpu-architecture', 'sm-streaming-multiprocessor', 'vector-unit-ridge'],
        Component: LessonInsideChip,
      },
      {
        id: 'memory-hierarchy',
        title: 'The memory hierarchy',
        standfirst:
          'On-chip memory is roughly twenty times faster than HBM. Almost every optimisation in this site is a way of staying on the fast side of that gap.',
        minutes: 6,
        concepts: ['memory-hierarchy', 'tiling', 'flash-attention'],
        Component: LessonMemory,
      },
      {
        id: 'the-fabric',
        title: 'The fabric',
        standfirst:
          'Once a model spans more than one chip, the network gets a roofline of its own — and batching cannot lift it.',
        minutes: 7,
        concepts: ['network-roofline', 'collectives', 'nvlink-domain', 'ici-topology', 'tpu-networking'],
        Component: LessonNetworkRoofline,
      },
    ],
  },
  {
    id: 'model',
    label: 'Part II',
    title: 'The model',
    blurb: 'What a transformer does to produce one token, and what it stores.',
    sections: [
      {
        id: 'one-token',
        title: 'One token, end to end',
        standfirst:
          'Characters in, one token out. Every stage in between is a matrix multiplication, and each will turn out to have a different appetite for the chip.',
        minutes: 8,
        concepts: ['prefill', 'generation', 'kv-cache', 'flops-per-token'],
        Component: OneToken,
      },
      {
        id: 'attention',
        title: 'Attention',
        standfirst:
          'Attention is compute-bound while reading a prompt and memory-bound while writing an answer. That single flip explains most of what follows.',
        minutes: 6,
        concepts: ['attention-intensity', 'attention-flops', 'flash-attention', 'gqa', 'attention-params'],
        Component: LessonAttention,
      },
      {
        id: 'kv-cache',
        title: 'The KV cache',
        standfirst:
          'Generation avoids redoing work by remembering it. The memory that remembering costs is where inference capacity actually goes.',
        minutes: 6,
        concepts: ['kv-cache', 'gqa', 'paged-attention', 'kv-sharding'],
        Component: LessonKVCache,
      },
    ],
  },
  {
    id: 'collision',
    label: 'Part III',
    title: 'The collision',
    blurb:
      'Put the model on the machine. The roofline is what happens at the contact point — the heart of the site.',
    sections: [
      {
        id: 'arithmetic-intensity',
        title: 'Arithmetic intensity',
        standfirst:
          'Every operation has a ratio: how much arithmetic it does per byte it moves. That one number decides which of the chip’s budgets you run out of first.',
        minutes: 6,
        concepts: ['arithmetic-intensity', 'matmul-intensity', 'flops-per-token', 'dot-product-intensity'],
        Component: LessonIntensity,
      },
      {
        id: 'the-roofline',
        title: 'The roofline and the ridge',
        standfirst:
          'Plot achievable speed against intensity and you get a roof with two slopes. Where they meet is the ridge, and which side you are on is the whole game.',
        minutes: 6,
        concepts: ['roofline', 'ridge-point', 'compute-bound', 'memory-bound', 'overlap', 'mfu'],
        Component: TheRoofline,
      },
      {
        id: 'prefill-vs-generation',
        title: 'Prefill vs generation',
        standfirst:
          'Reading a prompt and writing a reply are the same arithmetic at different batch sizes — which is why one saturates the chip and the other starves it.',
        minutes: 5,
        concepts: ['prefill', 'generation', 'critical-batch', 'ttft'],
        Component: LessonPrefillGen,
      },
      {
        id: 'quantization',
        title: 'Quantization',
        standfirst:
          'Using fewer bits per weight moves the ridge. It buys throughput not by making the maths faster but by making the bytes fewer.',
        minutes: 5,
        concepts: ['quantization', 'critical-batch', 'ridge-point'],
        Component: LessonQuant,
      },
    ],
  },
  {
    id: 'system',
    label: 'Part IV',
    title: 'The system',
    blurb: 'Everything people build on top of the roofline to move along it.',
    sections: [
      {
        id: 'latency-vs-throughput',
        title: 'Latency vs throughput',
        standfirst:
          'Bigger batches serve more users per second and make each of them wait longer. There is no setting that wins both.',
        minutes: 6,
        concepts: ['latency-throughput', 'critical-batch', 'ttft'],
        Component: LessonPareto,
      },
      {
        id: 'sharding',
        title: 'Sharding the model',
        standfirst:
          'When a model outgrows one chip you must cut it up. Each way of cutting trades memory for communication, and each becomes comms-bound somewhere.',
        minutes: 7,
        concepts: [
          'model-parallelism', 'tensor-parallelism', 'pipeline-parallelism',
          'expert-parallelism', 'data-parallelism', 'collectives', 'collective-matmul',
        ],
        Component: LessonSharding,
      },
      {
        id: 'serving-in-practice',
        title: 'Serving in practice',
        standfirst:
          'Continuous batching, disaggregation, prefix caching and speculative decoding are four different answers to the same problem: the chip is idle.',
        minutes: 7,
        concepts: [
          'continuous-batching', 'disaggregated-serving', 'prefix-caching',
          'speculative-decoding', 'paged-attention', 'ttft',
        ],
        Component: LessonServing,
      },
    ],
  },
  {
    id: 'bill',
    label: 'Part V',
    title: 'The bill',
    blurb: 'Turn the physics back into a number in dollars.',
    sections: [
      {
        id: 'tco',
        title: 'Total cost of ownership',
        standfirst:
          'Electricity plus amortised hardware, divided by tokens. Every earlier section shows up somewhere in this fraction.',
        minutes: 6,
        concepts: ['tco', 'mfu', 'latency-throughput'],
        Component: Tco,
      },
      {
        id: 'comparing-hardware',
        title: 'Comparing hardware',
        standfirst:
          'With the model fixed, accelerators stop being spec sheets and become points on a cost-latency frontier.',
        minutes: 6,
        concepts: ['tco', 'roofline', 'ridge-point'],
        Component: ComparingHardware,
      },
    ],
  },
];

/** Every section in reading order — the spine. */
export const FLAT_SECTIONS: { part: Part; section: Section; index: number }[] =
  CURRICULUM.flatMap((part) => part.sections.map((section) => ({ part, section, index: 0 })))
    .map((entry, index) => ({ ...entry, index }));

export const SECTION_IDS = FLAT_SECTIONS.map((e) => e.section.id);

export const FIRST_SECTION_ID = SECTION_IDS[0];

export function findSection(id: string) {
  return FLAT_SECTIONS.find((e) => e.section.id === id);
}

export function neighbours(id: string) {
  const entry = findSection(id);
  if (!entry) return { prev: undefined, next: undefined };
  return {
    prev: FLAT_SECTIONS[entry.index - 1],
    next: FLAT_SECTIONS[entry.index + 1],
  };
}

/** Section number as shown in the header, e.g. "2 of 4". */
export function sectionNumber(id: string): string | undefined {
  const entry = findSection(id);
  if (!entry) return undefined;
  const within = entry.part.sections.indexOf(entry.section) + 1;
  return `${within} of ${entry.part.sections.length}`;
}

export const TOTAL_MINUTES = FLAT_SECTIONS.reduce((sum, e) => sum + e.section.minutes, 0);
