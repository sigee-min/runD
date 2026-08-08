# Release Artifact

## Authority

The current package version is `1.0.5`. CMake install rules in
`cmake/root/package.cmake` own the installed prefix. Candidate routes add only
sealed source and artifact identity before archiving.
The checked-in package identity may advance while it is still a candidate.
Public download and support pages move only after the matching archive,
checksum, and verifier are hosted by the separate publication authority.
The installed external consumer under `package/tests/consumer` owns package
usability for both the staged prefix and the extracted archive.

## Local Release Check

```sh
tools/release/run
```

The generated CTest route first builds the exact selected target union,
including `runD_package_install`, and therefore stages the package before test
execution. Release selects product and package namespaces; repository harness
self-tests remain in their one full Debug owner. The local command and manual
Linux workflow consume the same internal selection authority.
`package.consumer` only validates that staged prefix, configures a
fresh external project with `find_package(runD 1.0.5 EXACT CONFIG REQUIRED)`,
and builds and runs the SDK, Compute, black-box, and private-implementation
consumers. It never starts a nested build of the repository tree. The consumer
configure accepts only the exact `1.0.5` version. All three external configure
probes name
the Ninja generator and the repository state-locking Ninja driver together.
That pair is intrinsic to each fresh consumer build and does not inspect or
inherit a producer CMake cache. The Release route separately owns the producer
generator contract. Staged-prefix and extracted-archive consumers therefore
have identical inputs and no second producer-state authority.
The external consumer build directory and staged package prefix must be
disjoint; the consumer validator rejects either path nested inside the other.
The staged artifact carries `share/runD/LICENSE` and
`share/runD/licenses/xxhash/LICENSE`; the consumer compares both byte-for-byte
with their checked-in authorities before compiling external code.
Before configuring the positive consumer, an isolated expected-failure project
requests a generic unsupported component. The probe is accepted only when
package discovery reports the component unavailable and leaves `runD::sdk`
unimported. A second isolated project proves that an existing foreign-prefix
`runD::sdk` is rejected. These are CMake package-graph checks, so both projects
configure with `LANGUAGES NONE`; invoking compiler identification or ABI probes
would add no evidence. The positive external project is then configured with
`CMAKE_BUILD_TYPE=Release`, matching the producing Release archive.

The package configure keeps five compiler-driven negative probes for lambda
body rules that cannot be expressed by a type trait. The capture-controlled
mutable-global probe is paired with the same valid `compute::select` recipe
over a `constexpr` global in the positive Compute consumer: that recipe must
compile and execute, while only the mutable form may fail. A malformed
host-language branch therefore cannot masquerade as enforcement of the
global-state rule. Fixed type and operator rejection rules are ordinary
`static_assert` contracts in the positive installed translation unit. The five
lambda-body rules are the complete per-rule compiler-probe set. Compiler-driver
process counts and elapsed time remain environment-dependent and are reported
only by a fresh installed-package measurement.

The current Replay, telemetry, and Compute-admission shape is owned
by `tests/consumer/contract/surface.cpp` and compiled once. The isolated
`tests/consumer/contract/session.cpp` proves that the focused Session header
does not import Compute. The installed target registry is filtered by namespace
and must contain exactly `runD::sdk`. Current type constraints and executable
journeys are the release authority.

Release verification keeps installed inventory/closure and the behavioral
lock, snippet, platform, route, verifier, and identity contracts. Each owner is
executed at its public boundary. The installed consumer compares the produced
header manifest and physical include tree exactly, and archive fixtures execute
the published verifier against invalid checksums and entry types.

Process-exit policy is verified by the installed `exit/failure.cpp` and
`exit/assertion.cpp` executables plus the product journeys.
Every executable consumer registers its expected exit and accelerator-resource
class once. Configure writes that registry to `runs.tsv`; both the build
dependency closure and the installed runner consume the same rows. Record,
Scenario, checkpoint continuation, and bounded History are explicit installed
journeys in that registry rather than unexecuted documentation links.

Every official C++ documentation fence is classified. Each distinct `compile`
source builds as an isolated installed-SDK translation unit, `run` sources
execute, and `fragment` fences are explicitly contextual. Repeated fences bound
to the same canonical consumer source share one build target instead of
recompiling identical bytes. Canonical copies are still checked byte-for-byte
against that source. The same canonical documentation list validates every
local link target and heading anchor outside code fences. This is one linear
AWK pass over documentation bytes plus one repository-file registry; it creates
no compiler invocation or independent test target. Classification always
constructs and validates one complete generated candidate. Promotion compares
each candidate file by content at the stable output path, retains identical
file timestamps, and removes paths absent from the candidate. Consequently an
unchanged source does not invalidate its external object, while deleted or
changed fences still update the next configure exactly.

The Compute executable explicitly opens CPU, Metal, and Vulkan. Every
available device must preserve the selected backend and compile and execute the
same typed Map; an unavailable device must report a nonempty `Unsupported` or
`Unavailable` result and may not fall back. The black-box executable records a
deterministic timer observation, exercises replay encode/decode/diff/window and
same-task re-execution, and records then replays host I/O through typed
descriptors with `native == -1`. This makes installed replay success evidence
for archive resolution, not an accidentally successful operating-system call.
Only this Compute executable waits for and holds the repository accelerator
lock. External configure/build and the SDK, black-box, private-header, and
documentation executables remain outside the device critical section. This
wait policy lets package preparation overlap other Release contracts while
preventing a normal CTest scheduling overlap from becoming a false status-75
package failure. Its 1,200-second outer step bound covers the 900-second
competing accelerator-test bound plus margin, while a nested timer preserves
the Compute executable's independent 120-second execution limit.

The route starts its internal Release build from an empty owned tree,
requires the product-source manifest to remain unchanged through execution,
and rejects any Ninja object whose dependency record is empty. The installed
consumer therefore cannot be backed by a stale incremental object.

The consumer pins `runD_DIR` to the newly installed
`<prefix>/lib/cmake/runD` directory and compares the resolved canonical path
with that directory before building. An ambient package registry or another
installed runD artifact cannot satisfy this proof.
The exported target contains only prefix-relative runD archives and symbolic
consumer-side dependency owners. Metal, Foundation, Vulkan are
resolved while the external project runs `find_package`. The installed
consumer requires the imported target's include directory, archive order, and
complete link closure to resolve from the fresh prefix, and the extracted
candidate repeats that proof from a disjoint location.

The Compute consumer also executes installed-header contracts for 32- and
64-bit `Fixed<I,F>` multi-input Flow. Its second `zip_input` keeps an
independent same-width `(I,F)`, while explicit `Down/Wrap` quantization remains
attached through downstream map and combine nodes and is observable in
`graph::Info`. The installed graph-service consumer also checks
`graph::MemoryPlan`, active-count lineage, 256-byte arena offsets,
nonoverlapping reuse, a proved pointwise destructive Map alias, and rejection
of destructive reuse for indexed Maps while executing the same outputs twice.
This is installed SDK proof, not an in-tree header-only compile check. The same
positive translation unit requires float/double
conversion in both directions and mixed-format `+`, `-`, `*`, `/`, `min`, and
`max` to be ill-formed against that same fresh install.

## Candidate Lifecycle

`tools/internal/package/candidate` is the sole archive lifecycle owner for
Darwin and Linux. Given a completed Release install and a platform identity, it
performs one ordered state machine:

```text
validate inputs -> stage prefix -> seal archive -> verify/install -> consume
                -> close source identity -> publish archive, checksum, verifier
```

The owner validates source and artifact seals through
`package/cmake/identity.cmake`, copies the installed prefix, embeds all three
sealed identities plus the schema validator, creates the temporary archive and
checksum, and runs the exact temporary `rund-verify` that will be published.
That verifier snapshots the archive, rejects noncanonical paths and entry
types, verifies the three seals, runs the embedded source/artifact schema and
public-target-hash authority, checks the exact consumer host tuple, and only
then installs a disjoint prefix. The lifecycle invokes
`package/cmake/consumer.cmake` against that prefix with exact expected
provenance and closes source identity before the three final same-filesystem
renames. Cleanup is best-effort for catchable exits. Each final rename is
atomic; the three-file publication is not a single transaction under an
uncatchable termination.

Repository, build, and artifact-identity inputs are resolved once to physical
absolute paths before the lifecycle derives paths or changes the archive
working directory. The relative build directory used by the Linux workflow
and the absolute build directory used by the Darwin route therefore execute
the same state machine and own the same temporary artifact.

`tools/internal/artifact/identity` is the only identity-generation entry. It
dispatches to platform observation code, then applies the same canonical TSV,
seal, SDK version, triplet, and installed-public-target validator consumed by
the extracted package proof. Platform routes do not duplicate lifecycle or
schema validation. Platform collectors emit unsealed observations only;
`package/cmake/identity.cmake` alone owns identity keys, canonical ordering,
serialization, sealing, and validated field reads.

## Darwin ARM64 Artifact

```sh
tools/release/darwin
```

This is the canonical `darwin-arm64` candidate producer. It requires a native
Darwin ARM64 host and reads the SDK version through the package-version owner.
Before any build work, it consumes the canonical platform row and rejects a
tuple that is neither a coherent `validated/candidate` nor
`supported/release` pair. The common archive owner repeats that admission
before staging, so direct invocation cannot bypass platform policy.
The producing Release route's sealed source manifest and identity are the
provenance authority; the identity records the exact revision and dirty state
observed by that route. The producer invokes `tools/release/run` once, so configure, target
selection, component contracts, `runD_package_install`, and the staged package
consumer retain their existing owners. It does not configure or build the
repository through a second path.

After Release and Darwin identity observation pass, the common candidate owner
creates exactly:

```text
.cache/release/dist/rund-sdk-1.0.5-darwin-arm64.tar.gz
.cache/release/dist/rund-sdk-1.0.5-darwin-arm64.sha256
.cache/release/dist/rund-verify
```

`share/runD/release/artifact-identity.tsv` is the release-note and binary-tuple
authority inside the archive; its adjacent `.sha256` seals its exact bytes. The
sorted schema records the actual Release compiler ID, compiler version and
target, C++ standard-library ID and version, C++ standard, build type, public
compile definitions, the installed `runDTargets.cmake` SHA-256, target
architecture, compiler deployment-target macro, Apple SDK
identity/version/build, and the exact installed `molten-vk`, `vulkan-headers`,
and `vulkan-loader` formula versions. Metal
and Foundation record the same Apple SDK version whose actual framework paths
are used by Release. The producer fails when Homebrew reports no version or
more than one installed version, or when a CMake-resolved backend path is not
owned by the matching formula/SDK. It never guesses dependency versions from
package names.

The common validator requires the exact sorted Darwin schema,
SDK/triplet/Release/arm64 facts, installed-public-target SHA-256, and producer
identity bytes. The artifact evidence packet binds the archive, artifact
identity, SDK version, triplet, source manifest, source identity, and published
verifier hashes.
GitHub publication is a separate operator action and is not performed here.
The archive producer proves the exact source bytes and identity it consumed;
the publication authority separately owns tag and hosted-release admission.
The 1.0.5 archive, checksum, and verifier may be externally hosted only after
their bytes match the passing candidate evidence and the canonical Darwin row
is promoted to `supported/release`. Candidate creation never rewrites policy.

## Manual Linux Artifact

`.github/workflows/release.yml` checks out an existing tag,
builds and tests the release configuration, copies the installed prefix into a
versioned directory, and retains the verified archive, SHA-256 checksum, and
self-tested `rund-verify` as a temporary workflow candidate. It does not create
a GitHub release by default. The tag name must equal the checked-in package
version exactly. It
reads that version through the same
`tools/internal/package/version` owner as the Darwin route. The workflow
consumes the repository's single configure,
CTest target-selection, and selection-execution authorities; the
`package.release-workflow` contract requires each shared authority exactly
once. The same contract requires `package.consumer` to receive its staged
install from the generated target route and to retain only the external
configure/build/run lifecycle.

The workflow reads the canonical `linux-x64` classification from
[`platform/tuples.tsv`](./platform/tuples.tsv) through the shared status reader;
release and public site guides own no second platform matrix. Retaining a workflow
candidate never promotes the tuple. The workflow brackets its one repository
build in the registered `.cache/release` root with the source-manifest boundary
and delegates identity generation and archive lifecycle to the same owners as
Darwin.
The candidate step validates its checksum through the path created in that
shell process. It exports archive paths through `GITHUB_ENV` only for later
upload and optional publication steps.
That identity records the Release GNU compiler version and target, C++20 and
libstdc++ version, glibc and LLVM versions, x86-64 triplet, public compile
policy, installed `runDTargets.cmake` SHA-256, and exact SDK version.
Hosted-release upload remains conditional on the canonical platform policy
being changed to the exact `supported` row.

Generated archives belong under `.cache/` or CI artifact storage and are not
repository authority.
