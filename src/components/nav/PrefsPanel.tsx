import { useEffect, useId, useRef, type ReactNode } from 'react';
import { SlidersHorizontal, X } from 'lucide-react';
import {
  DEFAULTS, LINE_HEIGHT, MEASURE, countChanged, resetPrefs, setPref, usePrefs,
  type Prefs,
} from '../../lib/prefs';

interface PrefsPanelProps {
  open: boolean;
  onClose: () => void;
}

interface Choice<V> { value: V; label: string }

/** A native radio group drawn as a segmented control, so a screen reader
 *  announces "2 of 3, checked" and arrow keys work with no script. */
function Segment<K extends 'theme' | 'text' | 'measure' | 'leading' | 'density' | 'motion'>({
  name, title, desc, readout, choices, note,
}: {
  name: K;
  title: string;
  desc: string;
  readout?: string;
  choices: Choice<Prefs[K]>[];
  note?: ReactNode;
}) {
  const prefs = usePrefs();
  const uid = useId();
  const descId = `${uid}-desc`;
  return (
    <fieldset className="prefs__group" role="radiogroup" aria-label={title}>
      <legend className="prefs__label">
        {title}
        {readout && <output className="prefs__readout">{readout}</output>}
      </legend>
      <p className="prefs__desc" id={descId}>{desc}</p>
      <div className="prefs__segment" aria-describedby={descId}>
        {choices.map((c) => {
          const id = `${uid}-${String(c.value)}`;
          return (
            <span key={String(c.value)} className="contents">
              <input
                className="prefs__radio" type="radio" name={`${uid}-${name}`} id={id}
                checked={prefs[name] === c.value}
                onChange={() => setPref(name, c.value)}
              />
              <label className="prefs__cell" htmlFor={id}>{c.label}</label>
            </span>
          );
        })}
      </div>
      {note}
    </fieldset>
  );
}

function Switch({ name, label, desc }: { name: 'chartGrid' | 'focusMode'; label: string; desc: string }) {
  const prefs = usePrefs();
  const uid = useId();
  return (
    <label className="prefs__switch" htmlFor={`${uid}-sw`}>
      <input
        className="prefs__check" type="checkbox" role="switch" id={`${uid}-sw`}
        checked={prefs[name]} onChange={(e) => setPref(name, e.target.checked)}
        aria-labelledby={`${uid}-l`} aria-describedby={`${uid}-d`}
      />
      <span className="prefs__switch-track" aria-hidden="true"><span className="prefs__switch-thumb" /></span>
      <span className="prefs__switch-text">
        <span className="prefs__label prefs__label--row" id={`${uid}-l`}>{label}</span>
        <span className="prefs__desc" id={`${uid}-d`}>{desc}</span>
      </span>
    </label>
  );
}

/** The reading-and-display settings dialog. A real <dialog>, opened with
 *  showModal(), so the top layer supplies the scrim, focus containment and
 *  Escape. */
export default function PrefsPanel({ open, onClose }: PrefsPanelProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const prefs = usePrefs();
  const changed = countChanged(prefs);
  const osStill = typeof window !== 'undefined'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="prefs"
      aria-labelledby="prefs-title"
      onClose={onClose}
      onClick={(e) => { if (e.target === ref.current) onClose(); }}
    >
      <div className="prefs__sheet">
        <header className="prefs__head">
          <span className="prefs__head-icon" aria-hidden="true"><SlidersHorizontal size={14} /></span>
          <h2 className="prefs__title" id="prefs-title">Reading and display settings</h2>
          <button type="button" className="prefs__close" onClick={onClose} aria-label="Close settings" title="Close settings">
            <X size={14} aria-hidden="true" />
          </button>
        </header>

        <div className="prefs__body">
          <Segment
            name="theme" title="Appearance"
            desc="System follows your operating system and keeps following it, even if you change it later. The header's sun, moon and monitor cycle the same three options."
            choices={[
              { value: 'light', label: 'Light' },
              { value: 'dark', label: 'Dark' },
              { value: 'system', label: 'System' },
            ]}
          />
          <Segment
            name="text" title="Text size (%)" readout={`${prefs.text}%`}
            desc="Scales every text size on the site, tables and chart labels included."
            choices={([90, 100, 115, 130] as const).map((v) => ({ value: v, label: String(v) }))}
          />
          <Segment
            name="measure" title="Reading width" readout={MEASURE[prefs.measure]}
            desc="How much text runs across one line of prose. Narrower is easier to track; wider fits more on screen."
            choices={[
              { value: 'narrow', label: 'Narrow' },
              { value: 'normal', label: 'Normal' },
              { value: 'wide', label: 'Wide' },
            ]}
          />
          <Segment
            name="leading" title="Line spacing" readout={LINE_HEIGHT[prefs.leading]}
            desc="The leading on body text. Taller leading helps a tired eye stay on the line."
            choices={[
              { value: 'tight', label: 'Tight' },
              { value: 'normal', label: 'Normal' },
              { value: 'relaxed', label: 'Relaxed' },
            ]}
          />
          <Segment
            name="density" title="Density"
            desc="Padding and gaps around controls, cards and tables — how much breathing room the layout leaves itself."
            choices={[
              { value: 'compact', label: 'Compact' },
              { value: 'normal', label: 'Normal' },
              { value: 'spacious', label: 'Spacious' },
            ]}
          />
          <Segment
            name="motion" title="Motion"
            desc="System follows your operating system. Reduced removes every transition on this site. Full keeps them on unless your system asks for less."
            choices={[
              { value: 'system', label: 'System' },
              { value: 'reduced', label: 'Reduced' },
              { value: 'full', label: 'Full' },
            ]}
            note={osStill && prefs.motion === 'full' ? (
              <p className="prefs__note">
                Your system asks for reduced motion, so every choice renders a still page —{' '}
                <strong>Full</strong> included. A system-level accessibility setting is not something this site overrules.
              </p>
            ) : undefined}
          />

          <div className="prefs__switches">
            <Switch name="chartGrid" label="Chart grid lines"
              desc="Draw the horizontal and vertical guides behind every chart." />
            <Switch name="focusMode" label="Focus mode"
              desc="A reading aid: it centres the text column and dims the side panels until you hover them. Navigation stays reachable." />
          </div>
        </div>

        <footer className="prefs__foot">
          <p className="prefs__count">{changed === 0 ? 'Using the defaults' : `${changed} changed`}</p>
          <button type="button" className="prefs__reset" onClick={resetPrefs} disabled={changed === 0}>Reset</button>
        </footer>
      </div>
    </dialog>
  );
}

export { DEFAULTS };
