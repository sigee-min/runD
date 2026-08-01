# Documentation Site

The documentation surface turns landing-page curiosity into a verified first
run, a correct mental model, and a complete integration path. Together with the
landing, it is the sole public learning and integration authority.

Repository and subsystem docs remain normative engineering and verification
evidence. A public guide may link to that evidence for inspection, but it must
contain every instruction, limitation, example, error, and recovery step needed
to complete its task.

## Home

The docs home opens with three routes:

```text
Run the first Flow       Fix a failed setup       Understand the guarantee
Verify -> Link -> Run    Diagnose -> Recover      Numbers -> Graph -> Evidence
```

Below them, task cards route by intent:

- Build with Compute.
- Record and replay.
- Integrate runtime lifecycle, networking, and telemetry.
- Choose an admitted numeric law.
- Inspect performance, platforms, APIs, and errors.

A visible `1.0.3 Alpha` version badge and Darwin ARM64 support note stay beside
the first-run action.

## Navigation

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

Every navigation leaf is a site-native public page. Source evidence is
secondary and cannot substitute for unfinished public content.

## Learning Paths

### Evaluate the claim

1. Check the supported platform and dependency boundary.
2. Run the same complete example on available explicit targets.
3. Compare output bytes and graph identity.
4. Read the numeric and ordering boundaries.
5. Inspect the scoped performance and release evidence.

### Install and recover

1. Download one coherent release tuple.
2. Verify and install the versioned SDK prefix.
3. Build and run the complete first Flow.
4. Follow Troubleshooting for any verifier, package, dependency, or backend
   failure.
5. Record the exact typed error and environment evidence if recovery fails.

### Build a workload

1. Choose an admitted scalar and bounded shape in Numerics.
2. Express one Flow and collect the first result in Compute.
3. Compile a Program when inputs repeat.
4. Use resident, Pipeline, or Batch only for the matching cost boundary.
5. Add replay, runtime lifecycle, networking, and telemetry at their documented
   application boundaries.

### Inspect the public surface

1. Find the focused header in API & Errors.
2. Follow its site-native guide and complete checked example.
3. Handle its documented result and typed failures.
4. Use repository source links only when implementation or verification
   evidence is needed.

## Page Pattern

Every site-native guide should answer these questions in order:

1. What outcome will the reader obtain?
2. Which product contract makes it valid?
3. What is the smallest checked example?
4. Which failure or limitation matters?
5. How can the reader verify the claim?
6. What is the next decision?

Reference pages may favor dense tables, but every code sample identifies
whether it is a complete checked source or a contextual fragment.

## Versioning and Discoverability

- Navigation and the sitemap expose every public teaching, integration,
  reference, error, and troubleshooting route.
- The version badge defaults to the currently published site version and never
  presents `main` behavior as a released guarantee.
- A page whose contract changed after the published Alpha shows that distinction
  before the content.
- Stable route names do not contain version numbers; the selected version is
  part of the rendered content and source mapping.

## Documentation Quality Gates

- One public behavior has one owning site route.
- Examples copied from checked consumer sources retain byte identity with that
  source.
- Quick Start is copy-paste complete and names its expected output and installed
  prefix.
- Every direct public header has a site-native task explanation and route to
  its result and recovery contract.
- Complete examples copied from installed consumer sources are byte-checked;
  contextual snippets are presented as patterns, not standalone proof.
- Troubleshooting covers release integrity, verifier, package discovery,
  dependency, and backend failures without requiring an external guide.
- Navigation works without requiring the landing-page interaction.
- Headings, landmarks, skip links, focus order, and code-copy controls are
  keyboard and screen-reader usable.
- Unsupported platform paths never end in a download action.
- Every performance value links to its measurement method and scope.
- Every page names the version or source state it describes.
- Repository and subsystem links are optional evidence, never required public
  journey steps.
