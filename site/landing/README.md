# Landing Page

The landing is the shortest useful evaluation path for a C++ runtime, engine,
or simulation developer. It is not a release ledger and it is not a second
manual. In one pass, a visitor must be able to answer:

1. What application code does runD replace or simplify?
2. Is its exactness contract appropriate for this workload?
3. What does integration look like in public C++?
4. When is an accelerator path worth measuring?
5. Can the current Alpha run on this host?

Complete commands, checked programs, API contracts, errors, and recovery steps
live in Docs.

## First Viewport

The first viewport leads with the developer job:

- author one bounded `Flow` instead of application-owned backend kernels;
- choose an explicit `Target` at execution time;
- receive the accepted bytes or that target's typed failure, with no implicit
  fallback.

The right side is a small, labeled `public API excerpt`, followed by the actual
checked parity output:

```text
same bytes: cpu = metal = vulkan [3, 5, 7, 9]
```

This code-and-result pair replaces the release fact table. It shows the
integration model and the observable contract, which are the two facts a
prospective SDK user needs first. Release status remains a compact line:
`1.0.3 Alpha · Darwin ARM64 binary · C++20 · MIT`.

The primary action opens the complete parity Quick Start. The secondary action
opens the Compute guide.

## Five-Section Sequence

### 1. Hero: One application Flow, explicit targets

State the useful proposition in application terms: write the bounded state
transition once, select the admitted execution backend, and preserve the
accepted result. Pair it with the smallest honest public API excerpt and its
checked output. Do not lead with package inventory, dependency versions, or a
backend diagram.

### 2. Fit: Protect state that cannot drift

Use two short rule groups rather than a comparison table.

`Use runD when`:

- lockstep, replay, simulation, or authoritative state must reproduce;
- CPU and supported accelerator execution should share one application Flow;
- an unavailable backend must remain an explicit failure.

`Not this release`:

- arbitrary runtime C++ callbacks or unconstrained floating point must match
  bit-for-bit;
- automatic fastest-device selection is required;
- a supported Windows binary or general Linux binary matrix is required.

Each group links to the owning Determinism, Compute, API, or Platforms guide.

### 3. SDK: Start small, add execution structure when it pays

Show the integration path a real application can grow through:

- `Flow + collect()` for one host result;
- `Program` when the graph repeats with new input;
- `resident()` when state should stay on the selected device;
- `Pipeline` for resident producer-to-consumer work;
- `Batch` for independent jobs that should share a submission boundary.

Keep the package contract visible as one CMake line:

```cmake
target_link_libraries(app PRIVATE runD::sdk)
```

Also name the adjacent runtime surface once: Session, Tasks, Host input,
Network, Storage, Replay, and Telemetry. Link to Compute and Runtime rather than
summarizing those guides.

### 4. Performance: Decide what to amortize

Present the two measurements that help an evaluator choose an execution shape,
not a device leaderboard:

- **Warm execution:** `Compute-heavy warm map`, `N = 262,144`;
- **Submission:** 64 small GPU jobs, serial calls versus one `Batch`.

The checked M4 Pro medians remain adjacent to each question. State once that
resident creation and initial upload are excluded and must be amortized. No
bars, winner labels, setup-cost leaderboard, or portable CPU-versus-GPU
conclusion are allowed.

### 5. Availability: Check the Alpha boundary

Use three compact status rows:

- Darwin ARM64: supported binary; CPU, native Metal, Vulkan through MoltenVK;
- Linux x64: validated source candidate; no release backend matrix;
- Windows x64: unsupported.

The MoltenVK dependency limitation stays visible before the download action,
but it does not occupy the hero.

## Content Rules

- Every landing block must help an evaluator decide fit, integrate, measure, or
  reject the release.
- Prefer a public API shape, checked output, explicit limitation, or next action
  over a slogan.
- State one proposition once. The next section must add new information.
- Keep labels readable and use ordinary developer language before internal
  vocabulary.
- Link details to Docs rather than compressing a manual into the landing.

Remove or reject:

- release-fact tables in the hero;
- dense requirement, benchmark, or platform tables when a short rule group is
  easier to scan;
- execution diagrams, simulated target switching, and decorative proof
  graphics;
- repeated exactness slogans, trust strips, closing pitches, and card walls;
- “write any C++ once,” unconstrained deterministic floating-point claims,
  automatic backend claims, or unqualified performance rankings;
- future backend promises without a released contract and evidence.

## Interaction Contract

- Navigation and code copy are the only landing interactions.
- Static HTML retains every claim, limitation, and next action.
- No control pretends to execute a runD backend in the page.
- Meaning never depends on color, motion, or graphical comparison.
