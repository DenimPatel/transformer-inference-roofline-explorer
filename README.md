
# transformer-inference-roofline-explorer

An interactive, learning-first visualizer for Transformer inference roofline
analysis — arithmetic intensity, batch size, latency, and cost tradeoffs.

**Live demo:** https://denimpatel.github.io/transformer-inference-roofline-explorer/

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
