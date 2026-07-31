# Release Artifacts

Use only an SDK archive attached to the selected release and listed as
`supported` in [Platform Support](https://github.com/sigee-min/runD/blob/main/wiki/Platforms). Source compilation and
temporary CI artifacts do not establish a supported binary tuple.

## Artifact Names

```text
rund-sdk-<version>-<platform-triplet>.tar.gz
rund-sdk-<version>-<platform-triplet>.sha256
rund-verify
```

The current supported Darwin release shape is:

```text
rund-sdk-1.0.0-darwin-arm64.tar.gz
rund-sdk-1.0.0-darwin-arm64.sha256
rund-verify
```

The repository operator route that creates and validates these exact files is
`tools/release/darwin`. It leaves them under `.cache/release/dist`; publication
to a repository release is a separate authorized action. Darwin and the manual
Linux workflow use one package-owned candidate lifecycle. It consumes the
sealed Release source manifest and platform identity, verifies and extracts a
temporary archive through the exact adjacent verifier, runs the installed
consumer, closes source identity, and then renames all three final files into
place on the same filesystem. Each rename is atomic, but the three-file set is
not a single transaction under an uncatchable termination.

Triplet meanings and current support status are owned only by the canonical
policy reached through [Platform Support](https://github.com/sigee-min/runD/blob/main/wiki/Platforms); this artifact
guide does not mirror that matrix.

The manual Linux workflow produces a temporary candidate archive after
checking the exact SDK version and running package and component contracts.
A workflow candidate is not a supported SDK release. Consume only a separately
published artifact whose tuple the canonical policy lists as `supported`.

## Verify And Install

Run the adjacent verifier instead of manually extracting the archive:

```bash
sh ./rund-verify \
  ./rund-sdk-1.0.0-darwin-arm64.tar.gz \
  ./rund-sdk-1.0.0-darwin-arm64.sha256 \
  "$PWD"
```

It verifies a private archive snapshot, the canonical checksum row, path and
entry-type safety, all three embedded seals, source and artifact schemas, the
installed public-target hash, and the current host's exact compiler, standard
library, platform SDK, and backend-dependency tuple. It publishes the versioned
prefix and prints `artifact-identity.tsv` only after every check succeeds. A
failure leaves no installed prefix; discard all three downloaded files and
obtain one coherent set from the same release.

## Artifact Identity Tuple

An installed binary artifact is identified by all of these values:

- runD SDK version
- platform triplet
- compiler family and version
- compiler target and target architecture
- C++ standard library and version
- build type
- public compile definitions
- installed `runDTargets.cmake` SHA-256
- platform-specific OS, SDK, and toolchain identities
- external backend dependency versions where the producer declares them

The archive's sealed
`share/runD/release/artifact-identity.tsv` is the release-note authority for
that tuple. An archive is valid only when the consumer matches those exact
values; package names alone are not version evidence.

## Installed Layout

The prefix published by `rund-verify` is a CMake package prefix:

```text
rund-sdk-1.0.0-darwin-arm64/
  include/
  lib/
    cmake/
      runD/
        runDConfig.cmake
        runDTargets.cmake
  share/
    runD/
      LICENSE
      release/
        source-manifest.tsv
        source-manifest.tsv.sha256
        source-identity.tsv
        source-identity.tsv.sha256
        artifact-identity.tsv
        artifact-identity.tsv.sha256
        identity.cmake
```

Discover the package with `find_package(runD 1.0.0 EXACT CONFIG REQUIRED)`. Do not
hard-code paths below `include/` or `lib/`; the imported `runD::sdk` target owns
those requirements.
Every distributed archive includes the runD license shown above. The package
defines no CMake components; link the sole `runD::sdk` target.
The package accepts only the exact declared version.
The two provenance seals bind the distributed payload to the source bytes,
revision, and dirty state recorded by the artifact-producing Release route.
The artifact-identity seal binds the compiler, architecture, standard library,
build policy, installed public-target bytes, Apple SDK, and external backend
versions validated by the producer and extracted-package consumer.
`identity.cmake` is the archive-local copy of the same schema and public-target
validator used by the producer. It is consumed by `rund-verify`; applications
do not include or invoke it directly.
