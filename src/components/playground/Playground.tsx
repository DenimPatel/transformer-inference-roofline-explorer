import { useRef, useState } from 'react';
import { FlaskConical } from 'lucide-react';
import Section from '../shell/Section';
import PlaygroundIntro from './PlaygroundIntro';
import MicroGptLab from './MicroGptLab';
import TrainPanel from './TrainPanel';
import { MicroGPT } from '../../lib/microgpt';

/**
 * Off-spine. A real port of reference/microgpt.py that trains and generates in
 * the browser — useful, but a different kind of thing from the roofline
 * curriculum, so it gets its own route rather than a place in the reading order.
 */
export default function Playground() {
  const modelRef = useRef<MicroGPT | null>(null);
  if (!modelRef.current) modelRef.current = new MicroGPT();
  const sharedModel = modelRef.current;

  const [trained, setTrained] = useState(false);
  const [sessionKey, setSessionKey] = useState(0);
  const bumpSession = () => setSessionKey((k) => k + 1);
  const markTrained = () => { setTrained(true); bumpSession(); };
  const markUntrained = () => { setTrained(false); bumpSession(); };

  return (
    <article className="space-y-4">
      <header className="space-y-2">
        <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400 flex items-center gap-1.5">
          <FlaskConical style={{ width: 12, height: 12 }} /> The bench · off the main path
        </p>
        <h2 className="text-3xl font-bold text-slate-800 tracking-tight">microGPT playground</h2>
      </header>

      <PlaygroundIntro />

      <Section title="Lab" standfirst="Type a prompt, prefill it to build the KV cache, then generate one token at a time while watching the cache grow and attention spread.">
        <MicroGptLab model={sharedModel} trained={trained} sessionKey={sessionKey} />
      </Section>

      <Section title="Train" standfirst="Train microGPT from scratch with Adam, right here in JavaScript — then the Lab above shares the same trained weights.">
        <TrainPanel model={sharedModel} trained={trained} onTrained={markTrained} onUntrained={markUntrained} />
      </Section>
    </article>
  );
}
