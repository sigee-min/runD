# Installed SDK Boundary

CMake install rules own the physical installed prefix. The package must
contain every direct header owned by [`headers.tsv`](./headers.tsv), their
required support headers, the `runD::sdk` package configuration, and the
Kernel, Node, and Cluster libraries. `share/runD/LICENSE` is part of every
distributed artifact and must be byte-identical to the repository license.
`share/runD/licenses/xxhash/LICENSE` is the sole installed third-party notice
and must be byte-identical to its checked-in source authority.

The checked-in [`headers.tsv`](./headers.tsv) owns the direct entries and
private paths. At install time, each direct header is preprocessed as an
independent C++20 translation unit. The union of its transitive project-local
dependencies is installed, sorted, and recorded in
`share/runD/sdk-headers.tsv`. Duplicate install-relative ownership, symlinks,
private-path reachability, and any file present outside that inventory fail the
artifact. Required transitive support headers are staged but are not consumer
entry points. Source, test, tooling, cache, and generated evidence trees are
never installed.

The executable proof is `package.consumer`. The generated CTest target route
first selects `runD_package_install`, whose install owner builds one fresh,
complete candidate prefix. It promotes that candidate only when its sorted
path, byte-size, and SHA-256 inventory differs from the current stage. An
identical stage keeps downstream dependency timestamps for the warm developer
loop without weakening candidate freshness. Static archive member metadata is
normalized with the toolchain's reproducible mode before comparison; build or
install time is not permitted to perturb artifact identity. The consumer then validates that prefix, configures a
separate CMake project with only that prefix, builds representative SDK and
Compute programs, and runs them. It cannot build or install the repository
tree itself. This installed consumer is the package surface authority; no
parallel artifact model exists. The same proof rejects package components,
requires the sole `runD::sdk` target to be imported with exactly C++20 and the
producer's strict floating-point options, and verifies the installed license
against its source authority. Its exact link closure is the three installed
archives in dependency order (`cluster`, `node`, `kernel`), followed once by
the platform dependencies resolved by the consumer package configuration.
The target publishes that ordered closure once as package metadata; the
consumer compares the actual interface to that one owner instead of rebuilding
a second platform-order list. It independently requires the archive prefix and
membership of every resolved native dependency. Every archive must exist and
no dependency may repeat. The imported target is
marked with its install prefix; an existing same-named target from another
prefix is rejected instead of being silently accepted, and an isolated
expected-failure configure proves that collision boundary. Component and
collision probes enable no compiler language because they inspect only the
CMake package graph; each remains a fresh isolated configure. The positive
consumer requires the imported target's include directory, archive order, and
complete link closure to resolve from the fresh prefix.
Component rejection runs in an isolated expected-failure configure and must
leave `runD::sdk` unimported. The subsequent positive installed consumer uses
the producer's `Release` build type. Its build tree is incremental, but every
run reconfigures it, reruns all semantic compiler-rejection probes, rebuilds
against changed installed inputs, and executes every consumer. A fresh Release
route starts without either stage or consumer tree, so reuse cannot substitute
for release-candidate freshness evidence.

The same external project compiles every distinct official `cpp compile`
source as an isolated installed-SDK translation unit and runs every distinct
source marked `run`. Multiple fences naming the same canonical consumer source
share one target after byte-equality validation. Context-only excerpts must be
marked `cpp fragment`; bare, unknown, empty, or unclosed C++ fences and
canonical-example drift fail before the consumer build. The classifier first
validates a complete candidate tree, then promotes its files by content at
stable source paths and removes stale paths. Identical generated files retain
their timestamps, so the incremental consumer does not rebuild an unchanged
source merely because classification ran again.

Every validated candidate archive is a copy of that installed prefix with one
release-owned addition: `share/runD/release/` contains the source manifest,
source identity, artifact identity, each file's SHA-256 seal, and
`identity.cmake`, the archive-local schema and public-target-hash validator.
The source files come from the producing `tools/release/run` route; the artifact
identity binds the compiler/dependency tuple and installed
`runDTargets.cmake` SHA-256. The common `tools/internal/package/candidate`
owner never mutates `runD_package_install`. It publishes the archive,
checksum, and `rund-verify` as one three-file candidate. Before publication it
runs that exact verifier against the temporary archive; only a verified prefix
is passed to this same consumer authority before the three final per-file
renames. When
expected Release provenance is supplied, the consumer requires all embedded
payloads and canonical seals to be byte-identical to that producing route and
requires the public-target hash to match the extracted file.
