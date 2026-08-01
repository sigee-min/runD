# Repository Layout

This page owns the durable repository root layout policy: root admission,
generated/local state, and physical root changes. Subsystem internals remain
owned by their local docs.

## Root Categories

`docs/architecture/root/layout.tsv` is the single machine-readable authority
for exact root names and classifies each one as an admitted file, admitted
directory, local directory, or platform debris. This page owns the policy and
the registry owns membership; scripts and build files must consume the registry
instead of copying its name set.

Admitted entries cover root contracts, project metadata, public contribution
and security policy, GitHub automation, CMake support, documentation, tooling,
SDK packaging, the public landing and documentation website, and product-level
subsystems. The `site` root owns the GitHub Pages presentation layer for the
landing and documentation experience; it links to product contracts instead
of becoming a second product authority. Local entries are limited to generated
or private state. Debris is not repository authority. Source evidence excludes
registered platform-debris names. Generated Python bytecode and `__pycache__`
directories under an admitted root fail closed instead of becoming
product-source identity; tool caches belong under `.cache/`.

## Root Admission

A new tracked root directory is allowed only when it is one of these:

- a durable repository-wide owner that cannot belong to an existing owner;
- a product-level subsystem with its own `docs`, build integration, tests, and
  SDK exposure decision;
- a platform-required metadata directory such as `.github`.

Feature slices, experiments, adapters, benchmark outputs, scratch evidence,
release artifacts, and temporary work do not get root directories. They belong
under the owning subsystem, `tools`, `docs`, `site`, `package`, or `.cache`.

## Update Rule

Any change that adds, removes, or renames a tracked root entry must update this
page in the
same commit. If a root entry is only temporary or generated, put it under
`.cache` or add it to `.gitignore` instead of admitting it here.

## Root Relocation

Moving existing product roots under an umbrella directory such as `subsystems`
changes CMake, package artifacts, docs links, CI, and user workflows. It
therefore requires an explicit repository-layout change with measured benefit
and complete consumer evidence. Bridge directories and symlink aliases are not
admitted.

## Verification

`tools/source/manifest` validates every actual root entry against the registry,
fails closed on unknown entries or missing admitted owners, and is exercised by
the source-manifest contract. Symlinks, special filesystem entries, and
ambiguous path spellings below an admitted root fail closed instead of
disappearing into byte evidence. Generated Python caches are excluded because
they are neither source nor durable repository state.
Admitted product paths use printable ASCII without backslashes; control
characters and ambiguous separator spellings fail closed before hashing.
Tracked root changes are reviewed against this page and exercised by the
affected build, package, and release contracts.
