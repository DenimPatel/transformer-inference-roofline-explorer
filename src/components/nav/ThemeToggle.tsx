import { Sun, Moon, Monitor } from 'lucide-react';
import { cycleTheme, usePrefs, type Theme } from '../../lib/prefs';

const NEXT: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' };
const ICON = { system: Monitor, light: Sun, dark: Moon } as const;

/** One button cycling System -> Light -> Dark, announcing the choice it holds. */
export default function ThemeToggle() {
  const { theme } = usePrefs();
  const Icon = ICON[theme];
  return (
    <button
      type="button"
      onClick={cycleTheme}
      className="nav-icon-btn"
      aria-label={`Theme: ${theme}. Switch to ${NEXT[theme]}.`}
      title={`Theme: ${theme} — switch to ${NEXT[theme]}`}
    >
      <Icon style={{ width: 16, height: 16 }} aria-hidden="true" />
    </button>
  );
}
