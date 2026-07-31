# Platform Support Policy

This page owns runD SDK artifact platform support policy.

## Status Terms

| Status | Meaning |
| --- | --- |
| `supported` | A release artifact exists for the tuple and package, consumer, and component contract validation passed for that artifact. |
| `validated` | Source-level tests pass on the platform, but no release artifact is promised for consumers. |
| `not supported` | No supported release artifact or conformance claim is published for the tuple. A temporary candidate may exist. |

The structured registry spells the final status `not-supported`; it has the
same meaning as the prose term above.

## Artifact Tuples

[`tuples.tsv`](./tuples.tsv) is the machine-readable and human-reviewable
authority for triplet, status, producer, runner, compiler, and artifact class.
Rows are unique and sorted by triplet. Release tooling reads the registry
through `tools/internal/platform/status`; workflows and prose do not maintain
parallel status rows.

Status and artifact class form one fail-closed pair. `supported` requires
`release`, `validated` requires `candidate`, and `not-supported` requires
`none` together with `none` for producer, runner, and compiler. The status
reader rejects every other combination before a release or candidate route
starts. Producer-specific spellings are not artifact classes; the producer
column already owns that distinction.

## Darwin Release Boundary

`tools/release/darwin` consumes the `darwin-arm64` row before Release work,
then runs the same-manifest Release gate, seals the exact artifact identity,
runs the archive's exact `rund-verify`, and reruns the installed consumer from
the verified prefix before publishing the local archive, checksum, and
verifier. The current `supported/release` row authorizes the archive, checksum,
and verifier hosted together in the
[1.0.0 release](https://github.com/sigee-min/runD/releases/tag/1.0.0) as one
consumer release.

A `supported/release` row requires its hosted archive, checksum, and
self-tested verifier to exist together and match passing evidence.
Documentation and workflows derive the state from that row; a local archive,
successful source build, or prose edit cannot promote a tuple.

## Backend Dependencies

The `darwin-arm64` release contains the runD archives, headers, package files,
and license. Metal and Foundation come from the consumer's matching Apple SDK;
Vulkan/MoltenVK is an external backend runtime dependency. Its version and
resolved native library identity are part of the artifact
identity tuple and must match the sealed
`share/runD/release/artifact-identity.tsv` carried by the verified SDK prefix.
The installed package resolves them on the consumer host during
`find_package`; it never retains the producer's Xcode, Homebrew, source, or
build paths. Missing Metal, Foundation, or Vulkan dependencies fail
package discovery rather than silently disabling a declared backend.

The current Homebrew package names used by the validated Apple Silicon route
are `molten-vk`, `vulkan-headers`, and `vulkan-loader`. A
successful install of differently versioned dependencies does not establish
the required binary tuple; use the exact formula versions in that identity file.
The same file records the actual compiler target, target architecture,
deployment-target macro, standard-library version, and Apple SDK build; the
producer rejects a non-arm64 or ambiguous dependency identity. Its
`public_target_sha256` binds the extracted `runDTargets.cmake` bytes used for
package discovery.

## Linux Candidate Boundary

The manual Linux workflow validates that the selected tag names the exact
checked-in SDK version, runs the package consumer and component contracts,
embeds sealed source and Linux artifact identities, verifies the archive and
checksum with the adjacent verifier, and reruns the same external installed
consumer from a disjoint verified prefix before retaining the three-file
candidate as a workflow artifact. The
workflow reads the `linux-x64` row from this page; no wiki or release guide
owns a second platform matrix. That validated candidate is not a supported
release artifact and does not promote `linux-x64` by itself.
The workflow installs `util-linux` and records `flock --version`; release and
accelerator critical sections fail closed if neither Linux `flock` nor
Darwin/BSD `lockf` is available.

Release publication is disabled by default. It may be enabled only after this
page deliberately lists `linux-x64` as `supported` and the release has the
matching compiler, package, consumer, and component evidence. Until that
promotion, consumers must ignore Linux candidate artifacts even when a
workflow run succeeded.

## Determinism Claim Boundary

runD supports deterministic scheduler-visible ordering after a host observation
stream is admitted. Host observation admission is the boundary: the runtime can
order admitted observations deterministically, but the platform can still
deliver native readiness observations at different wall-clock times or in
different backend ready-list orders.

runD does not promise identical wall-clock readiness timing and does not
promise identical native backend ready-list order across kernels, operating
systems, NICs, peers, or platform event mechanisms.

The supported `darwin-arm64` conformance proof includes:

- package consumer configure/build/run;
- component contract tests for the artifact-producing source tree;
- verifier-owned archive snapshot, checksum, entry-safety, sealed provenance,
  exact host-tuple validation, and a second consumer run from the verified
  versioned prefix;
- a sealed exact artifact-identity schema whose bytes match the producer and
  whose public-target SHA-256 matches the extracted package.

Configuration does not reject other host tuples. Successful configuration,
source compilation, or candidate workflow execution does not create a
supported product tuple; only a deliberately published `supported` row and its
release evidence own that claim. The current table classifies the published
Darwin ARM64 1.0.0 artifact as `supported/release`; Linux remains a validated
candidate.
