import { useEffect } from 'react';
import TokenGenerationTab from '../TokenGeneration';

/**
 * Part II opener: the whole path from characters to a sampled token.
 *
 * This was the "Token Generation" tab and its first eight sub-tabs. It belongs
 * here, before attention and the KV cache, because those two sections are
 * about specific stages of the pipeline this one lays out end to end.
 */
export default function OneToken({ onComplete }: { onComplete: () => void }) {
  useEffect(() => { onComplete(); }, [onComplete]);
  return <TokenGenerationTab />;
}
