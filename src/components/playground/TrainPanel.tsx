import { useState, useMemo, useRef } from 'react';
import { Line, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { RotateCcw, Sparkles, GraduationCap, Loader2 } from 'lucide-react';
import Figure from '../shell/Figure';
import { MicroGPT } from '../../lib/microgpt';
import { SAMPLE_CORPUS, rawNames, sampleNames } from '../../lib/names_data';
import { CHART as C } from '../../lib/theme';

export default function TrainPanel({ model, trained, onTrained, onUntrained }: {
  model: MicroGPT;
  trained: boolean;
  onTrained: () => void;
  onUntrained: () => void;
}) {
  const [corpus, setCorpus] = useState(SAMPLE_CORPUS.join('\n'));
  const [steps, setSteps] = useState(1000);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [loadingFull, setLoadingFull] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const docs = useMemo(() => corpus.split('\n').map((s) => s.trim()).filter((s) => s.length > 0), [corpus]);

  const handleLoadSample = (which: 'inline' | 'names') => {
    setSamples([]);
    setLossHistory([]);
    setTraining(false);
    if (which === 'names') setCorpus(sampleNames(300, rawNames).join('\n'));
    else setCorpus(SAMPLE_CORPUS.join('\n'));
  };

  const handleLoadFull = async () => {
    setSamples([]);
    setLossHistory([]);
    setTraining(false);
    setError('');
    setLoadingFull(true);
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}names.txt`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const lines = text.split('\n').map((s) => s.trim()).filter(Boolean);
      setCorpus(lines.join('\n'));
    } catch {
      setError('Could not load the full names.txt file.');
    } finally {
      setLoadingFull(false);
    }
  };

  const handleTrain = async () => {
    if (docs.length === 0) { setError('Add at least one document (a word per line) to train on.'); return; }
    setError('');
    setTraining(true);
    setProgress(0);
    setLossHistory([]);
    setSamples([]);
    const controller = new AbortController();
    abortRef.current = controller;
    await model.train(docs, steps, {
      signal: controller.signal,
      onStep: (step, loss) => {
        setProgress(step);
        if (step % 10 === 0 || step === steps) {
          setLossHistory((h) => { const next = [...h, { step, loss }]; return next.length > 400 ? next.slice(-400) : next; });
        }
      },
    });
    abortRef.current = null;
    setTraining(false);
    setProgress((p) => (controller.signal.aborted ? p : steps));
    onTrained();
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleRandomize = () => {
    model.reset();
    setLossHistory([]);
    setSamples([]);
    setProgress(0);
    setTraining(false);
    onUntrained();
  };

  const handleSampleNames = () => {
    setSamples(model.generate(12, 0.6));
  };

  const lossData = lossHistory.length >= 2
    ? lossHistory
    : [{ step: 0, loss: lossHistory.length ? lossHistory[0].loss : 0 }, ...lossHistory];

  return (
    <div className="space-y-5">
      {/* corpus + controls */}
      <div className="glass rounded-xl p-4 space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[240px]">
            <label className="text-xs font-semibold text-slate-500 mb-1 block uppercase tracking-wider">
              Training corpus <span className="normal-case font-normal">(one document per line)</span>
            </label>
            <textarea
              value={corpus}
              onChange={(e) => setCorpus(e.target.value)}
              rows={6}
              spellCheck={false}
              className="glass-input w-full text-sm py-2 px-3 font-mono text-slate-800 resize-y"
            />
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-600">Steps</span>
              <input type="number" min={1} max={10000} value={steps}
                onChange={(e) => setSteps(Math.max(1, Math.min(10000, Number(e.target.value) || 1)))}
                className="glass-input w-24 py-1.5 px-2 text-sm font-mono" />
            </div>
            {training ? (
              <button type="button" onClick={() => handleStop()}
                className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
                style={{ background: 'var(--color-accent-2)' }}>
                <span className="inline-flex items-center gap-1.5">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Stop ({progress}/{steps})
                </span>
              </button>
            ) : (
              <button type="button" onClick={() => handleTrain()}
                disabled={docs.length === 0}
                className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
                style={{ background: '#7b4b90' }}>
                <span className="inline-flex items-center gap-1.5">
                  <GraduationCap className="w-3.5 h-3.5" /> Train
                </span>
              </button>
            )}
            <button type="button" onClick={() => handleSampleNames()}
              disabled={training || !trained}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-slate-600 bg-white hover:bg-white disabled:opacity-40">
              <span className="inline-flex items-center gap-1.5"><Sparkles className="w-3.5 h-3.5" /> Sample names</span>
            </button>
            <button type="button" onClick={() => handleRandomize()}
              disabled={training}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-slate-600 bg-white hover:bg-white">
              <span className="inline-flex items-center gap-1.5"><RotateCcw className="w-3.5 h-3.5" /> Random weights</span>
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-semibold text-slate-600">Dataset:</span>
          <button type="button" onClick={() => handleLoadSample('inline')} disabled={training}
            className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white hover:bg-white">30 inline names</button>
          <button type="button" onClick={() => handleLoadSample('names')} disabled={training}
            className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white hover:bg-white">
            Load names.txt subset (300)
          </button>
          <button type="button" onClick={() => handleLoadFull()} disabled={training || loadingFull}
            className="rounded-lg px-3 py-1.5 text-xs font-semibold bg-white hover:bg-white">
            <span className="inline-flex items-center gap-1.5">
              {loadingFull && <Loader2 className="w-3 h-3 animate-spin" />}
              Load full names.txt (32k)
            </span>
          </button>
          <span className="text-[11px] text-slate-400">{docs.length} documents ready</span>
        </div>
        {error && <p className="text-xs text-rose-500 font-semibold">{error}</p>}
      </div>

      {/* loss curve + samples */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="glass rounded-xl p-4">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
            Training loss <span className="normal-case font-normal">(cross-entropy / token)</span>
          </h4>
          {lossHistory.length < 2 ? (
            <div className="h-40 flex items-center justify-center text-sm text-slate-400">
              {training ? <span className="flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> optimizing…</span> : 'No training run yet.'}
            </div>
          ) : (
            <Figure takeaway="Loss is cross-entropy per character; it should fall toward 0 as the model learns to predict the next character in your corpus.">
              <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossData} margin={{ top: 8, right: 10, left: -4, bottom: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis type="number" dataKey="step" tickFormatter={(v) => v.toLocaleString()} fontSize={10} />
                    <YAxis type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => Number(v).toFixed(2)} fontSize={10} />
                    <Tooltip labelFormatter={(v) => `step ${Number(v).toLocaleString()}`} formatter={(v: any) => [`loss ${(Number(v)).toFixed(3)}`]} />
                    <Line type="monotone" dataKey="loss" stroke={C.violet} strokeWidth={2.5} dot={false} name="loss" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Figure>
          )}
          {trained && <p className="text-[11px] text-slate-400 mt-1">Adam ran for the full run — weights now produce corpus-like tokens.</p>}
        </div>

        <div className="glass rounded-xl p-4">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
            Hallucinated names <span className="normal-case font-normal">· inference on the trained model</span>
          </h4>
          <p className="text-[11px] text-slate-400 mb-3">
            Autoregressive sampling (temperature ~0.6), mirroring the reference&rsquo;s inference loop. Click
            <strong> Sample names</strong> after training.
          </p>
          {samples.length === 0 ? (
            <p className="text-sm text-slate-400">
              {trained ? 'The model is trained — hit "Sample names" to hear what it learned.' : 'Train first, then this readout fills with fresh names.'}
            </p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {samples.map((s, i) => (
                <span key={i} className="rounded-lg px-2.5 py-1.5 text-sm font-bold text-white"
                  style={{ background: '#7b4b90' }}>
                  {s}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
