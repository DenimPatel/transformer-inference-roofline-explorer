import { Sun, Moon, Monitor } from 'lucide-react';
import { useTheme, type ThemePref } from '../../lib/useTheme';

const NEXT: Record<ThemePref, ThemePref> = { system: 'light', light: 'dark', dark: 'system' };
const ICON = { system: Monitor, light: Sun, dark: Moon } as const;

/** One button cycling System -> Light -> Dark, announcing the choice it holds. */
export default function ThemeToggle() {
  const { pref, cycle } = useTheme();
  const Icon = ICON[pref];
  return (
    <button
      type="button"
      onClick={cycle}
      className="nav-icon-btn"
      aria-label={`Theme: ${pref}. Switch to ${NEXT[pref]}.`}
      title={`Theme: ${pref} — switch to ${NEXT[pref]}`}
    >
      <Icon style={{ width: 16, height: 16 }} aria-hidden="true" />
    </button>
  );
}
