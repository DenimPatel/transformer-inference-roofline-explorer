
# transformer-inference-roofline-explorer

An interactive, learning-first visualizer for Transformer inference roofline
analysis — arithmetic intensity, batch size, latency, and cost tradeoffs.

**Live demo:** https://denimpatel.github.io/transformer-inference-roofline-explorer/

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
