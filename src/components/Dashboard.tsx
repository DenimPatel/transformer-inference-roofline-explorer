import React, { useState, useEffect, useCallback } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { ArrowLeft, ArrowRight, HelpCircle, SlidersHorizontal, X, List } from 'lucide-react';
import {
  CURRICULUM, FIRST_SECTION_ID, findSection, neighbours, sectionNumber,
} from '../lib/curriculum';
import { useRoute } from '../lib/useRoute';
import { loadProgress, saveProgress } from '../lib/progress';
import { ConfigProvider } from '../state/ConfigContext';
import ChapterRail from './nav/ChapterRail';
import ConfigPanel from './nav/ConfigPanel';
import Section from './shell/Section';
import ConceptGlossary from './learn/ConceptGlossary';
import Playground from './playground/Playground';
import Problems from './sections/Problems';

/** Sections that want the comparison checkbox list in the config panel. */
const COMPARISON_SECTIONS = new Set(['comparing-hardware']);

/** Routes where the global hardware/economics Configure panel has no effect. */
const HIDES_CONFIG = new Set(['playground']);

export default function Dashboard() {
  return (
    <ConfigProvider>
      <Shell />
    </ConfigProvider>
  );
}

function Shell() {
  const [route, navigate] = useRoute(FIRST_SECTION_ID);
  const [completed, setCompleted] = useState<string[]>(loadProgress);
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [contentsOpen, setContentsOpen] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);

  useEffect(() => { saveProgress(completed); }, [completed]);

  useEffect(() => {
    document.body.style.overflow = glossaryOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [glossaryOpen]);

  const go = useCallback((id: string) => {
    navigate(id);
    setContentsOpen(false);
  }, [navigate]);

  const entry = findSection(route);
  const BENCH: Record<string, () => React.ReactElement> = { playground: Playground, problems: Problems };
  const BenchPage = BENCH[route];
  const { prev, next } = neighbours(route);

  const markComplete = useCallback((id: string) => {
    setCompleted((c) => (c.includes(id) ? c : [...c, id]));
  }, []);

  // An unknown slug (a stale bookmark) falls back to the front door rather
  // than rendering nothing.
  useEffect(() => {
    if (!entry && !BenchPage) navigate(FIRST_SECTION_ID);
  }, [entry, BenchPage, navigate]);

  const showComparison = COMPARISON_SECTIONS.has(route);
  const showConfig = !HIDES_CONFIG.has(route);

  return (
    <div className="min-h-screen text-slate-900 font-sans flex flex-col selection:bg-[var(--color-accent)]/20">
      <div className="aurora" aria-hidden="true" />
      <ConceptGlossary open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />

      <header className="sticky top-0 z-40 glass-strong px-4 sm:px-6 py-2.5 flex items-center gap-4 justify-between shrink-0 border-b border-[var(--color-divider)]">
        <div className="flex items-baseline gap-3 min-w-0">
          <button type="button" onClick={() => go(FIRST_SECTION_ID)}
            className="nav-brand text-lg sm:text-xl truncate cursor-pointer bg-transparent">
            Roofline&nbsp;/
          </button>
          <p className="text-[11px] uppercase tracking-[0.08em] text-slate-500 hidden lg:block truncate">
            Transformer inference economics
          </p>
        </div>

        <div className="flex items-center gap-2 sm:gap-4 shrink-0">
          <button type="button" onClick={() => setContentsOpen((o) => !o)}
            className="btn-ghost xl:hidden text-xs">
            <List style={{ width: 14, height: 14 }} /> Contents
          </button>
          {showConfig && (
            <button type="button" onClick={() => setConfigOpen((o) => !o)}
              className="btn-ghost text-xs" aria-expanded={configOpen}>
              <SlidersHorizontal style={{ width: 14, height: 14 }} />
              <span className="hidden sm:inline">Configure</span>
            </button>
          )}
          <button type="button" onClick={() => setGlossaryOpen(true)} className="btn-ghost text-xs">
            <HelpCircle style={{ width: 14, height: 14 }} />
            <span className="hidden sm:inline">Concepts</span>
          </button>
        </div>
      </header>

      <div className="w-full max-w-[1600px] mx-auto flex-1 flex gap-6 p-4 sm:p-6">
        {/* Contents rail */}
        <aside className={`${contentsOpen ? 'block' : 'hidden'} xl:block w-full xl:w-64 shrink-0 xl:sticky xl:top-24 xl:self-start xl:max-h-[calc(100vh-8rem)] xl:overflow-y-auto custom-scrollbar`}>
          <ChapterRail
            current={route}
            completed={completed}
            onNavigate={go}
            onOpenGlossary={() => setGlossaryOpen(true)}
          />
        </aside>

        {/* The spine */}
        <main className={`${contentsOpen ? 'hidden xl:block' : 'block'} flex-1 min-w-0`}>
          <AnimatePresence mode="wait">
            <motion.div
              key={route}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            >
              {BenchPage ? (
                <BenchPage />
              ) : entry ? (
                <Section
                  part={`${entry.part.label} · ${entry.part.title}`}
                  number={sectionNumber(route)}
                  title={entry.section.title}
                  standfirst={entry.section.standfirst}
                  concepts={entry.section.concepts}
                >
                  <entry.section.Component onComplete={() => markComplete(route)} />
                </Section>
              ) : null}
            </motion.div>
          </AnimatePresence>

          {!BenchPage && (
            <nav className="flex items-stretch justify-between gap-3 mt-10 pt-6 border-t border-slate-200">
              {prev ? (
                <button type="button" onClick={() => go(prev.section.id)}
                  className="group flex-1 text-left p-3 hover:bg-slate-100 transition-colors cursor-pointer bg-transparent">
                  <span className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                    <ArrowLeft style={{ width: 13, height: 13 }} /> Previous
                  </span>
                  <span className="block mt-0.5 text-sm font-semibold text-slate-700">{prev.section.title}</span>
                </button>
              ) : <span className="flex-1" />}
              {next ? (
                <button type="button" onClick={() => go(next.section.id)}
                  className="group flex-1 text-right p-3 hover:bg-slate-100 transition-colors cursor-pointer bg-transparent">
                  <span className="flex items-center justify-end gap-1.5 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                    Next <ArrowRight style={{ width: 13, height: 13 }} />
                  </span>
                  <span className="block mt-0.5 text-sm font-semibold text-slate-700">{next.section.title}</span>
                </button>
              ) : <span className="flex-1" />}
            </nav>
          )}
        </main>

        {/* Configuration — docked, and read by every section */}
        {showConfig && configOpen && (
          <>
            <button
              type="button"
              aria-label="Close configuration"
              onClick={() => setConfigOpen(false)}
              className="fixed inset-0 z-30 bg-black/20 xl:hidden cursor-default"
            />
            <aside className="fixed xl:sticky right-0 top-0 xl:top-24 z-40 xl:z-auto h-full xl:h-auto xl:self-start w-[min(22rem,90vw)] shrink-0 overflow-y-auto xl:max-h-[calc(100vh-8rem)] custom-scrollbar glass-strong xl:bg-transparent p-4 xl:p-0">
              <div className="flex items-center justify-between mb-3 xl:hidden">
                <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Configuration</p>
                <button type="button" onClick={() => setConfigOpen(false)} className="btn-ghost text-xs">
                  <X style={{ width: 14, height: 14 }} />
                </button>
              </div>
              <ConfigPanel showComparison={showComparison} />
            </aside>
          </>
        )}
      </div>

      <footer className="w-full max-w-[1600px] mx-auto px-4 sm:px-6 py-6 text-[11px] text-slate-400 border-t border-[var(--color-divider)]">
        {CURRICULUM.length} parts · one reading order · every number recomputed from your configuration.
      </footer>
    </div>
  );
}
