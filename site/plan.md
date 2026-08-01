# Public Site Plan

## Objective

Make a qualified C++ runtime, simulation, or GPU engineer understand three
things in the first minute:

1. runD expresses one bounded typed computation across admitted execution
   backends; the computation authority is not tied to one GPU API.
2. Its useful differentiator is bit-identical authoritative output, not a
   promise that every GPU workload is faster.
3. The claim has inspectable contracts, exact-byte evidence, explicit failure,
   and a runnable Darwin ARM64 Alpha.

The landing should let that visitor decide fit, reach a complete checked run,
interpret the evidence correctly, and reject an unsupported host without
working through a marketing narrative.

## Audience

| Visitor | Question on arrival | Evidence that earns the next click |
| --- | --- | --- |
| Simulation or engine architect | Can one path support lockstep, replay, and local acceleration? | Identical state bytes, stable ordering, canonical input, and replay evidence. |
| GPU or runtime engineer | Is this a real execution model or a CPU fallback behind an API? | One graph identity, explicit target selection, resident execution shapes, typed backend failure. |
| Technical lead or evaluator | What works today, and what is still Alpha? | Exact platform matrix, release artifact, verifier, limitations, and evidence links. |
| Contributor or researcher | Where are the laws and proof surfaces? | Numeric, Compute, kernel, acceptance, test, and measurement authorities. |

The primary visitor is a technically skeptical evaluator. The experience
should assume that “same bits across CPU and GPU” sounds improbable and make
verification easier than dismissal.

## Message Hierarchy

### Primary claim

> Write the state transition once. Choose the backend at runtime.

### Supporting sentence

> Keep one bounded C++20 Flow for CPU and supported accelerator targets. The
> selected target returns the accepted bytes or its typed failure, with no
> implicit fallback.

### Proof line

> A short public API excerpt and the checked parity output appear together in
> the first viewport.

### Calls to action

| Priority | Label | Destination | Visitor intent |
| --- | --- | --- | --- |
| Primary | Run the parity example | Versioned Quick Start | Prove it locally. |
| Secondary | Explore the Compute API | Compute guide | Evaluate the integration model. |
| Tertiary | Check whether it fits | Landing fit section | Reject an unsuitable workload early. |

The Alpha label stays visible beside release calls to action. The site never
implies a supported platform beyond the published release boundary.

## Decision Journey

The landing follows five decisions without a flow diagram or simulated
execution:

1. see the application-level Flow/Target shape and checked result;
2. accept or reject the workload against the exact Compute scope;
3. choose the smallest useful execution shape and see how the SDK grows into
   resident Compute and Session-owned runtime work;
4. decide which setup, warm-execution, or submission boundary to measure;
5. check host support and choose a concrete next action.

The landing summarizes only what is required for those decisions. Docs own the
complete installation, examples, contracts, execution shapes, errors, recovery,
and limitations.

## Information Architecture

| Route | Page purpose | Primary next action |
| --- | --- | --- |
| `/runD/` | Landing: problem, fit, verified route, scoped measurements, and 1.0.1 support boundary. | Run the checked example or reject the fit. |
| `/runD/docs/` | Documentation home and task-first learning paths. | Choose Start, Guides, or Reference. |
| `/runD/docs/start/` | Verify, install, link, and run the first Flow. | Compare available targets. |
| `/runD/docs/troubleshooting/` | Diagnose release, verifier, package, dependency, and backend failures. | Recover or report exact evidence. |
| `/runD/docs/determinism/` | Numeric, graph identity, ordering, fallback, and proof laws. | Inspect exact authorities. |
| `/runD/docs/compute/` | Flow, Program, Job, Batch, Pipeline, and target selection. | Pick an execution shape. |
| `/runD/docs/replay/` | Canonical inputs, checkpoints, scenarios, and evidence. | Build a replay path. |
| `/runD/docs/runtime/` | Session lifecycle, bounded tasks, networking, and telemetry. | Integrate the runtime boundary. |
| `/runD/docs/numerics/` | Integer and Fixed widths, overflow, rounding, and quantization. | Choose an admitted scalar law. |
| `/runD/docs/performance/` | Cost model, resident work, batching, and measured evidence. | Measure the shipped boundary. |
| `/runD/docs/platforms/` | Supported, validated, and unsupported tuples. | Download the matching artifact. |
| `/runD/docs/api/` | Focused headers, result types, public errors, and API map. | Open the relevant guide or resolve an error. |

The compact navigation groups these routes as:

```text
Start
  Quick Start
  Troubleshooting

Guides
  Determinism
  Compute
  Replay
  Runtime
  Numerics

Reference
  Performance
  Platforms
  API & Errors
```

## Public Authority Contract

The landing and Docs are the sole public learning and integration authority.
Every released setup step, supported behavior, limitation, example, error, and
recovery path must be complete within those routes. Source links may expose
implementation or verification evidence, but following one is never required
to finish a public task.

Repository and subsystem docs remain the normative engineering evidence used to
review the public material:

| Public subject | Public owner | Engineering and verification evidence |
| --- | --- | --- |
| Installation, first Flow, and recovery | Quick Start and Troubleshooting | [`../package/docs/acceptance.md`](../package/docs/acceptance.md) and [`../package/docs/platform/support.md`](../package/docs/platform/support.md) |
| Determinism and numeric law | Determinism and Numerics | [`../docs/architecture/numeric.md`](../docs/architecture/numeric.md) and [`../kernel/docs/README.md`](../kernel/docs/README.md) |
| Compute behavior and execution shapes | Compute | [`../docs/reference/compute.md`](../docs/reference/compute.md) |
| Replay and runtime integration | Replay and Runtime | [`../node/docs/README.md`](../node/docs/README.md) |
| Performance and platform claims | Performance and Platforms | [`../docs/reference/performance/gpu.md`](../docs/reference/performance/gpu.md) and [`../package/docs/platform/support.md`](../package/docs/platform/support.md) |
| Public headers, results, and errors | API & Errors | [`../package/docs/api/stability.md`](../package/docs/api/stability.md) and the checked public headers |

Landing copy may shorten a public Docs claim but must not broaden it. Numeric
measurements carry their host, workload, execution shape, and evidence scope.
Quick Start owns the complete checked `compute.cpp` and `parity.cpp` programs.
The landing may retain only a short target-selection fragment labeled
`public API excerpt`, linked to the complete Docs-owned program.

## Content Rules

- Lead with the result a developer can obtain, then expose the mechanism.
- Prefer exact terms: `Flow`, `Target`, `Program`, `Batch`, `Pipeline`,
  state bytes, graph identity, and typed failure.
- Pair every surprising assertion with a nearby path to evidence.
- Say “supported targets” where the current release boundary matters.
- Separate deterministic authority from wall-clock timing and native event
  arrival.
- Present performance as a cost model and measured workload, never as
  “GPU is faster.”
- Make limitations visible before download, not hidden in a footer.
- Do not claim that arbitrary host callbacks, unconstrained floating point, or
  browser visualization are part of the deterministic Compute language.
- Prefer short rule groups, operational steps, and evidence rows to dense fact
  tables. Do not use execution diagrams, byte illustrations, target-toggle
  simulations, proof cards, or performance bars as substitutes for evidence.

## Delivery Stages

1. Freeze this page plan, the landing sequence, docs navigation, and visual
   direction.
2. Implement a static, base-path-safe site inside this directory.
3. Populate every public route in the compact IA with complete setup,
   integration, examples, limitations, errors, and recovery paths.
4. Add link, accessibility, responsive, and production-build verification.
5. Publish the exact built revision to GitHub Pages through GitHub Actions.

No deployment occurs until all production links use the `/runD/` base path and
the release, platform, and measurement claims match their current authorities.

## Acceptance

- The first viewport states the application problem, shows the public
  Flow/Target shape and checked parity output, names the typed-failure rule,
  keeps the Darwin ARM64 Alpha boundary visible, and offers the checked-run
  action without scrolling.
- A visitor can reach Quick Start, Troubleshooting, Determinism, Compute,
  Replay, Runtime, Numerics, Performance, Platforms, API & Errors, GitHub, and
  the release in at most two actions.
- Desktop and narrow mobile layouts preserve the problem-to-fit-to-action order.
- Navigation and copy controls work by keyboard and retain visible focus; the
  landing contains no simulated backend interaction.
- The visual system uses warm paper, ink, restrained oxide red, one system sans
  stack plus code-only mono, square controls, and document rules; it contains
  no generated product art or depth effects.
- Text and controls meet WCAG AA contrast; meaning never depends on color
  alone.
- The production build has no broken internal link and works at `/runD/`
  rather than assuming a domain root.
- No public learning or integration task requires a repository or subsystem
  document; those links are optional engineering evidence.
- Search metadata describes the supported product without unsupported
  superlatives.
- Any benchmark shown includes the exact host and workload caveat.
- Full checked programs and execution instructions are owned by Docs. A short
  landing fragment is explicitly labeled as an excerpt and no visualization is
  presented as backend execution.
