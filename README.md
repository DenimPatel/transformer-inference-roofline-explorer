
# transformer-inference-roofline-explorer

An interactive, learning-first visualizer for Transformer inference roofline
analysis — arithmetic intensity, batch size, latency, and cost tradeoffs.

**Live demo:** https://denimpatel.github.io/transformer-inference-roofline-explorer/

## Structure

The site is a single linear curriculum rather than a set of tabs. `src/lib/curriculum.ts`
is the one source of truth for it: five parts, nineteen sections, read top to bottom,
answering one question — what does it cost to serve one token, and why?

| Part | Covers |
| --- | --- |
| 0 · Start here | The headline number, the three budgets, and the prerequisites |
| I · The machine | Inside the chip, the memory hierarchy, the fabric |
| II · The model | One token end to end, attention, the KV cache, mixture of experts |
| III · The collision | Arithmetic intensity, the roofline and the ridge, prefill vs generation, quantization, the second ridge |
| IV · The system | Latency vs throughput, sharding, serving in practice, does it fit |
| V · The bill | Total cost of ownership, comparing hardware |

Every concept has exactly one canonical section; anywhere else that mentions it links
there with `<ConceptTag id="..." />` rather than explaining it again. Derivations sit in
collapsible `GoDeeper` blocks attached to the concept they belong to, so a first reader
gets the plain-English layer and a specialist expands in place. Every chart is wrapped in
`Figure`, whose `takeaway` prop is required — a reader who cannot yet parse a log-log plot
still gets the point.

Off the spine: a glossary of 62 concepts (`src/lib/concepts.ts`), a problem set, and a
microGPT playground that trains a real transformer in the browser.

Sections are routed by hash (`#/the-roofline`) so every one is linkable. The hardware,
model and economics configuration lives in `src/state/ConfigContext.tsx`, is read by every
section, and round-trips through the query string — so a configuration is shareable, and
the prose quotes live numbers from it.

## Design

The site is set in the **Broadsheet** design language shared with the companion
site at [DenimPatel/AI](https://github.com/DenimPatel/AI): near-black Source Serif
on paper white, cyan and magenta used sparingly as spot color, hierarchy from type
scale and whitespace rather than boxes. The tokens live in `src/index.css`; chart
literals that cannot read CSS variables come from `src/lib/theme.ts`.

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Run the app:
   `npm run dev`

## Deployment

Pushes to `main` are automatically built and published to GitHub Pages by
the `.github/workflows/deploy-pages.yml` workflow. Enable Pages once under
**Settings → Pages → Source: GitHub Actions** for the workflow to publish.
