# runD Site

This directory owns the public web experience built for
`https://sigee-min.github.io/runD/`.

It contains one product story with two connected surfaces:

- [`index.html`](./index.html) is the implemented landing surface.
- [`docs/`](./docs/README.md) contains the implemented documentation routes and
  their content contract.

Shared visual and interaction rules live in
[`design/`](./design/README.md). The audience, message hierarchy, information
architecture, source ownership, and delivery gates live in
[`plan.md`](./plan.md).

Use the checked project operators from this directory:

```sh
npm install
npm run dev
npm run build
npm run check
```

The production build uses the `/runD/` GitHub Pages base path and emits static
HTML for every documented route. `public/` owns shared browser assets. The
site deliberately omits a generated social image; text sharing metadata
remains complete. `dist/` and `node_modules/` are local generated state.

The landing and documentation routes are the sole public learning and
integration surfaces. A user must be able to understand, install, integrate,
run, and troubleshoot the released product without leaving GitHub Pages.

Checked-in repository and subsystem documents remain the normative engineering
and verification evidence behind those public explanations. They inform and
validate the site, but they are not required steps in a public user journey.
