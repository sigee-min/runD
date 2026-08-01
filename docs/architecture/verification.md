# Verification

CTest and subsystem executables are the repository test authority. Public
commands select those owners; evidence packets bind results to exact source
bytes.

## Commands

| Command | Contract |
| --- | --- |
| `tools/test/run` | Short Kernel, Node, and Accel semantic loop with local successful-result reuse. |
| `tools/test/run --fresh ...` | Bypass local successful-result reuse for the selected development route. |
| `tools/test/run <case>` | One exact Node case or exact CTest route. |
| `tools/test/run <case> --backend cpu\|metal\|vulkan` | CPU oracle plus one selected accelerator row, or the isolated CPU row; the focused build graph contains no unselected native backend objects. |
| `tools/test/run --match <regex>` | CTest rows selected by an explicit regular expression. |
| `tools/test/run --list` | List canonical Node cases and routes without configuring. |
| `tools/check/run` | Complete local Debug contract matrix. |
| `tools/release/run` | Release contracts and installed-package consumers. |
| `tools/release/darwin` | Release plus sealed Darwin ARM64 candidate and extracted-prefix consumption. |
| `tools/check/platform/unavailable` | Forced unavailable-platform runtime and Vulkan discovery ownership. |
| `tools/sanitize/run address` | ASan and UBSan product matrix. |
| `tools/sanitize/run thread` | TSan concurrency-owner matrix. |
| `tools/check/leaks` | Native Apple leak checks for the four lifetime owners. |
| `tools/measure/scheduler/run` | Installed scheduler latency, scaling, and memory workloads. |
| `tools/measure/compute/run [--pipeline\|--recurrence\|--window-repeat metal\|vulkan]` | Installed Compute execution, scaling, parity, and warm-cost workloads; focused modes build one physical backend projection while retaining its CPU oracle. |
| `tools/measure/compute/run --checkpoint cpu\|metal\|vulkan` | Compare live device publication, reusable host storage, and immutable host snapshots, including fused copy/hash latency, process allocations, transfer, staging, and RSS high-water observations. |
| `tools/measure/flow/run` | Installed Flow construction and C++ frontend workloads. |
| `tools/measure/graph/services/run` | Installed Program-cache, async-coalescing, and bounded-graph workloads. |
| `tools/measure/telemetry/run` | Installed Disabled, Basic, and Detail Replay telemetry cost and parity. |
| `tools/measure/build/run [build [target]]` | Measure public-header frontend cost, live reverse fan-out, and target dirtiness. |
| `tools/evidence/status` | Validate the newest required packets against the current source manifest. |

## Development Loop

The default and `--match` routes reuse `.cache/dev`. Their configuration stamp
covers profile, focus, verification tags, configure helpers, normalized Node
source topology, CMake identity, and the live cache. When the stamp matches,
the route builds the command-free `rund-contract-graph` target; Ninja
regenerates changed CMake inputs before CTest metadata is read. That warm
graph-sync boundary invokes the configured repository Ninja owner directly;
it does not start a CMake build dispatcher that can only invoke the same
owner again.

Configure emits one sealed route index containing each exact test name, target
union, CTest resource, timeout, and direct-or-accelerator runner. The selection
builder snapshots that index, derives the exact names and target union once,
and invokes the configured Ninja owner directly. If regeneration changes even
one index byte, selection restarts from the new sealed snapshot; byte identity
proves that `f(index, pattern)` is pure. Selection performs no CTest JSON
discovery. Accelerator runner, resource, and timeout agreement is validated
when the route is registered, not reconstructed on every edit loop.

After Ninja proves the selected outputs current, a direct local route may
reuse its last pass. The key binds the frozen selection, complete route index,
generated `CTestTestfile.cmake` set, runner implementation, host and CTest
identity, and every selected executable's bytes. Exact Node cases bind their
generated case index, route, selected backend, process runner, host, and
executable bytes by the same rule. A miss executes normally and publishes the
key atomically only after success; a failure is never stored. Script-only and
`rund_accel` resource routes always execute because filesystem artifacts do
not close over their external state. `--fresh` bypasses reuse.

This pass cache is disposable edit-loop state under the owning build tree. It
is not verification evidence. `tools/check/run`, sanitizer, Release, package,
measurement, and evidence routes do not consume it and always execute their
selected contracts.

Source-topology identity is the sorted live set of Node C++/Objective-C++
relative paths and filesystem kinds. Addition, deletion, rename, and kind
replacement change that set, including untracked or ignored files. Content
edits leave it unchanged and remain ordinary Ninja/depfile inputs. A topology,
tool, helper, option, or cache change performs the canonical configure, whose
explicit lists reject unowned sources.

An exact Node route reads one content-authenticated catalog generated by the
same registry parser, profile table, and group route table consumed by CMake.
Every cache hit hashes the complete parser input set and validates the catalog
seal. The result selects `.cache/focus/<profile>` and its lock. Generated
target, resource, and profile must equal that query before and after build
regeneration. There is no per-case graph, route authority mirror, or second
profile picker.

Local, sanitizer, and Release profiles require Ninja. The configured
repository Ninja owner is invoked directly and owns regeneration, target
selection, output mutation, and exit status. Missing Ninja and an existing
non-Ninja contract tree fail before execution.

Repository verification forces strict warnings after caller options:

- GNU/Clang C++ and Objective-C++: `-Wall -Wextra -Wpedantic -Werror`
- MSVC: `/W4 /WX /permissive-`

`cmake/root/compile.cmake` is the flag owner. These flags are private repository
policy and must not appear on imported `runD::sdk`.

When available, one repository ccache launcher serves C++ and Objective-C++
with `.cache/ccache` and the repository root as stable base. Focused, Debug,
Release, and sanitizer trees share compiler results without sharing mutable
build state. Contract targets use ordinary translation units; no umbrella PCH
is admitted. Test-only Socket access is a source-specific dependency of the
network suites, not a target-wide forced include on unrelated Node contracts.
All C++ contract executables consume the declaration-only
`tools/test/assert.hpp` boundary and link the single compiled
`rund-test-assertion` failure owner; subsystem-local assertion implementations
are not admitted.

## Source Identity

Fresh Debug, Release, sanitizer, platform, package, and measurement routes
capture `before` immediately before configure/build and `after` after execution.
Both are product-source manifests and must be byte-identical. The selected
source-manifest contract reuses `before`; it does not add another tree walk.

`tools/source/manifest` includes tracked and untracked product source, docs,
contracts, tests, package files, release workflows, tools, and public site pages.
Only platform debris registered in `docs/architecture/root/layout.tsv` is
excluded. Generated Python bytecode under admitted roots is ignored because it
is not source authority. Unknown or missing roots, non-regular entries, and
ambiguous paths fail before hashing. Each row records
file SHA-256, executable identity, and path, so content, mode, admission, and
path changes alter the manifest.

A fresh Ninja route also requires at least one object dependency record and
rejects every selected object with zero dependencies. The dependency audit
streams `ninja -t deps` through the byte-oriented C-locale summary owner,
retains only object summaries, and records producer status independently from
filter status. Stale or missing depfiles cannot serve as evidence.

Every route packet stores:

- copied `source-manifest.tsv` and its SHA-256;
- sealed source identity, including revision and worktree bytes;
- route, generator, compiler, host, selection, and pass/fail result;
- raw CTest or workload output required by that route.

A passing measurement packet additionally seals its raw-log name and SHA-256,
`baseline.log` name and SHA-256, selected profile, and compared metric count.
Status validation requires the current host and replays the pure comparator;
its output must be byte-identical to the sealed result. A route-level atomic
attempt marker reports preparation and execution as `in-progress` or `failed`
until a complete new packet is published.

Packets live under `.cache/evidence/<route>/<UTC-run-id>/`. A failed route also
publishes its packet. Results from several routes form one closure claim only
when their copied manifests are identical. `tools/evidence/status` validates
the newest packet for every host-required route and distinguishes missing,
in-progress, failed, corrupt, and stale evidence; it never builds or executes
a workload or upgrades an older result.

## Mutable-State Ownership

`tools/internal/state/roots.tsv` maps each static build identity to one build
root and route-lock leaf. Focus profiles are the only grammar-derived family.
For build identity `p`:

```text
R(p) = route lock for configure + build + test observation
S(p) = state lock for one CMake or Ninja mutation
R(p) != S(p)
nesting order = R(p) then S(p)
```

The route and state locks prevent two processes from mutating `.ninja_log`,
`.ninja_deps`, the generated graph, or outputs, while allowing CMake's compiler
probes under the owning state scope. Default, regex, exact CTest, and full
Debug routes share the `.cache/dev` route identity. Each focus profile owns its
own root. Release consumers share the Release route; a measurement also owns
its workload route.

Darwin/BSD `lockf` and Linux `flock` implement route and state locks. The
portable shell capability pool is `F = {9,8,7,6,5,4}`. For active keys `L`, the
slot assignment `f: L -> F` is injective. Re-entry for path `p` is valid only
when a slot's recorded path, inode, mode `0600`, unlinked state, and zero size
match its open descriptor. Environment text is not ownership. State locks are
non-reentrant; route and accelerator locks admit only capability-proven
re-entry. Pool exhaustion and missing advisory locking fail before child
execution.

Rows that open or submit through an accelerator carry the `rund_accel` CTest
resource and repository device lock. CPU-only work, package configure/build,
and SDK consumers that do not open a device stay outside that critical
section. The Linux SDK workflow uses the registered `release` root at
`.cache/release`; its checkout is isolated, so it shares the same build-state
authority without creating a CI-only mutable root.
section. The installed Compute consumer acquires the device only for its
execution phase. Direct accelerator work has an independent execution bound
inside its wait-plus-run CTest bound, so waiting does not consume the next
owner's execution allowance.

All semantic contracts have a CTest deadline. External exact cases, leak
owners, and measurements use `tools/internal/process/run`, whose required Perl
`alarm`/`exec` owner preserves normal child status and reports deadline expiry
without a polling process. Missing Perl fails before child execution.

## Test Ownership

Subsystem CMake files own executable composition and CTest registration.
`node/tests/contract/cases.def` and its fragments are the sole Node case
registry. One row owns name, symbol, source, group, profile, resource, and
optional verification tags. One route table maps groups to executables and
CTest resources.

The six Runtime-base groups share `node-runtime` but retain six ordered CTest
processes. The contract runner is one target-neutral OBJECT; a thin generated
table supplies each executable's selected cases. Focused source partitioning
normally retains the exact case owner and registry-neutral helpers. A profile
may instead declare profile-scoped materialization only when every retained
case shares one target, resource, source suite, and effective link profile and
no case owns a backend selector. The product profile uses that scope with an
explicit two-owner admission bound, while exact execution still passes only
the requested case.

Standalone Compute has two process groups over one semantic registry:

- `node-compute` is unlocked and closes at CPU Compute.
- `node-compute-accel` owns accelerator serialization and the
  CPU/Metal/Vulkan matrix.

The groups do not duplicate case bodies. A case's registry `backend` tag is
the sole capability authority for `--backend cpu|metal|vulkan`; the tooling
contract applies the same route law to every tagged case instead of retaining
a second name list. This includes `compute.window` alongside the Flow,
collective, and Pipeline owners. CPU selection changes the ordered
backend domain to `{CPU}`, links the isolated profile, and removes the device
resource.
A native selection retains `{CPU, selected accelerator}` so the same-process
oracle is initialized while the focused archive, native links, and picker
catalog omit every unselected accelerator object; it reuses the canonical
accelerator SCC root and lock. `compute.flow-primitives` is already a CPU-only
semantic oracle in the unlocked group, so it has no backend selector or
accelerator link edge. Untagged cases reject the selector before configuration.

The operator selector `telemetry:detail` is a namespaced registry-tag
projection, not a copied case list or regular-expression alias. Its isolated
focus tree and CMake/Ninja identifiers join the same segments with dots because
colons are build-tool syntax. Since dots are forbidden inside a tag segment,
this projection is one-to-one. No hyphen or underscore tag is stored or
translated. The tree
materializes only tagged profile owners, so Compute and Replay
parity can keep distinct minimal link closures while one command runs both.
Compute telemetry is a registered terminal owner, not a helper invoked by the
broader lifecycle case. Its tagged target contains the dispatcher, runner,
shared product support, and telemetry owner only. The same registry row owns
grouped, exact, and tagged execution, so a telemetry edit does not compile or
run unrelated capacity, cancel, concurrency, collective, await, or gate
contract owners.

One all-backend case owns an invariant's complete backend matrix. A
backend-specific case is admitted only for a distinct cache, resource,
lifecycle, or algorithm boundary. Runtime Compute owns admission, completion,
cancel/wait, telemetry, lifetime, and scheduler integration; it does not repeat
Standalone primitive or numeric matrices.

The main Compute semantic owners are:

| Owner | Sole responsibility |
| --- | --- |
| `compute.flow-primitives` | CPU semantic oracles for composition, ordered output identity, branching, indices, grouping, joins, and records |
| `compute.bounded-parity` | compact/filter cardinality, inactive tails, expand, reuse, domain, worker, and backend parity |
| `compute.expressions` | integral expression and unsigned high-bit laws |
| `compute.static-matrix` | exact Flow/Program types and `graph::Info` shape for Matrix, Transform, Factor, Solve, and Spectrum |
| `compute.flow-numeric-modes` | arbitrary Fixed formats, rounding, overflow, and backend parity |
| `compute.graph-services` | graph identity, cache, async coalescing, resource graph, and bounded downstream behavior |

Large Compute cases keep one registered runner and place independent semantic
owners below the matching one-word folder. Each case has one shared
`local`/`model` fixture authority; leaf linkage cannot add registry cases or
change domain/backend execution order. A leaf edit invalidates its one contract
object and relinks the existing case.

A registered test source remains separate only when it adds a distinct
assertion, failure surface, backend, or scheduling mode. Verification-tag
targets derive from the same registry and suite partitions; sanitizer-only
case or source lists are forbidden. An include-only `.cpp` is not a semantic
owner: its definitions belong in the nearest already-registered owner for the
same executable and resource boundary. Each meaningful verification
responsibility has one compiler translation unit and retains its assertions.

The Compute harness classifies each assertion by
`(feature, scalar domain, backend, lifecycle)`. Physical parity has two terminal
owners: `compute.collective-modes` covers collective tails, empty inputs,
extrema, overflow, and segmented reset for six scalar domains on CPU, Metal,
and Vulkan; `compute.boundary-modes` covers the remaining 255-element executor
tails, including bounded group/join composition, on the same three backends. The Flow
UX owner deliberately keeps only the following set:

| Feature | Domain oracle | Backend | Lifecycle |
| --- | --- | --- | --- |
| ordered outputs, alias leaves, canonical read order | I32 | CPU | resident selective read and `read_all` |
| pipe, side/scalar combine, repeat, record, branch, indices | U64 | CPU | one-shot |
| bounded-cardinality pipe | U32 | CPU | one-shot |
| stable group order and segmented projection | I32 | CPU | one-shot |
| stable join order, relational record, bound and empty side | U32 | CPU | one-shot |
| group index-capacity admission | U64 shape | CPU | compile rejection |

For backend set `B = {CPU, Metal, Vulkan}`, the excluded duplicate
projection is `FlowUX x (B - {CPU})`; its physical meaning is already observed
by the collective and boundary owners. Domain repetition is admitted only when
it changes the public Flow meaning. This set difference preserves every unique
Flow invariant while preventing a composition test from reopening the complete
device matrix.

`node/tests/contract/target/selection.hpp` is the only test-side execution
target helper. Collective rows pass their requested two-worker CPU policy
through that helper instead of restating a target projection at every graph.
The non-projectable boundary owner has one fixed three-backend array shared by
all of its domain families.

## Installed Measurement Method

Performance routes consume the installed Release SDK, never a private in-tree
target. Each packet records host, compiler, workload, source manifest, raw
samples, and results. Before a packet can pass, the shared comparator requires
an exact checked-in host profile and semantic digest, then applies the
one-sided limits in [`../reference/performance/baseline.tsv`](../reference/performance/baseline.tsv).
Missing, extra, or semantically changed rows fail. A numeric speedup still
requires an identical-workload comparison; satisfying an upper regression
limit is not a speedup claim. Algorithmic bounds may be claimed independently
when their proof is recorded.

All numeric measurement fields, including diagnostic and derived fields, must
be finite nonnegative canonical decimals. Parser and packet details are owned
by [`../reference/performance/method.md`](../reference/performance/method.md).
Each route parser also owns the unit of every admitted metric. The baseline is
a checked-in expectation, not a second unit source: `observe` prints measured
schema units together with the uniquely selected host profile, and ordinary
comparison rejects any baseline-unit mismatch.
Explicit baseline-cut validation binds the fifteen numbered provenance rows
to retained raw logs, exact current-host packet identities, and the requested
profile; ordinary candidate admission has no dependency on that ignored
calibration cache.

`tools/measure/telemetry/run` uses one installed executable and a balanced
three-setting schedule to separate the cost of installing Basic telemetry from
the incremental cost of Detail. Disabled-to-Basic observations are diagnostic;
only the positive Detail-minus-Basic wall, process-CPU, and allocation median
and p95 values are release gates. Replay status, ordering, transcript, result,
storage, and telemetry payload parity must pass before timing is admissible.

`tools/measure/scheduler/run` separates first-batch and warm behavior. It
measures spawn/complete, yield/resume, join, channel exchange, timer park/wake,
and Reactor request/ack, then scales task count, deterministic payload, and
parked-coroutine memory. Rows report configured and participating workers,
completion identity, frame high-water, resource limits, and memory observations.

`tools/measure/compute/run` measures CPU, Metal, and Vulkan at declared
workload sizes. Each timed row has an unmeasured warm-up, validates every
result, and saturating-sums warm counters immediately after execution. Warm
resident execution requires zero pipeline compiles, buffer allocations,
downloads, and uploaded bytes. A required output read and hash/parity check
occurs after timing; readback counters are correctness evidence and are not
folded into the warm-zero aggregate.

A fixed-capacity ledger makes CPU the oracle for domain, numeric, sparse,
collective, and orchestration rows. Accelerator results must match it. Node
orchestration additionally proves device waiting parks rather than occupies the
scheduler worker. `Stats::dispatches` counts algorithmic dispatches;
`Stats::command_submits` counts physical accelerator queue submissions.
An argument-free contract process enumerates CPU plus only the native
backends compiled into that process. An explicit `--backend` selection remains
the authority for a requested native row and may terminate with its typed
unavailable reason; neither path substitutes CPU for a requested accelerator.

The optional `--resident`, `--collective`, `--sort`, `--bulk`, `--batch`,
`--pipeline`, `--pipeline-profile`, and `--recurrence` modes are
bounded developer diagnostics over one incremental current-source Release
closure. Resident isolates 1,024- and 1,048,576-element creation and transfer;
collective keeps the two collective sizes and overflow evidence; sort keeps the
262,144-element dense and bounded-sparse map/sort comparison; bulk keeps the
1M transform and 512-square matrix throughput model. Pipeline keeps identical
separate-Program and prepared-Pipeline work in balanced ABBA order, requires
value/fingerprint/counter parity, and reports claim and control cost separately
from wall medians. Recurrence measures the fixed-count resident recurrence
path. A selected backend is also the focused build projection, and a selected
accelerator always retains the CPU oracle. Every
mode preserves semantic validation and
graph/output identity, but none invokes the evidence finisher or creates a
release packet. The argument-free installed-SDK route remains the sole
performance-baseline authority.

The optional `--checkpoint` mode runs CPU, Metal, or Vulkan over the same
one-million-element transactional state. It compares a live device handle,
one preallocated two-bank host `SnapshotStorage`, and retained immutable
snapshots. Each row reports the median combined tick/path wall time. Host-export
rows also report an export-only wall median containing fused payload
materialization and per-field FNV updates plus the root metadata-hash fold;
there is deliberately no independent hash timer or second payload pass. The
live-device row reports zero for that export-only field because handle
acquisition occurs once before the measured tick cadence. Generic
Pipeline `Stats` columns describe the last measured tick and checkpoint
operation; reusable checkpoint counters and staging-memory columns are
explicit before/after deltas across the complete sample cadence. RSS columns
are process high-water observations, so their delta is diagnostic rather than
a current-resident-byte claim. Process-wide `operator new` count and requested
bytes are reported as medians over separately bounded run and checkpoint-path
probes. The live-device checkpoint probe is its one selector acquisition;
reusable and immutable probes cover only their export calls. These values
intentionally include loader/driver allocation, such as a Vulkan translation
layer's native queue submission, rather than relabeling it as runD memory.

`tools.compute-focus` compiles that conditional diagnostic surface in the
complete contract matrix and executes its CPU sort projection. This prevents a
Release-only edit-loop entry from remaining outside compiler verification;
native accelerator timing remains under the explicit focused commands and the
repository device lock.

`tools/measure/flow/run` measures recipe construction separately from compile
and execution, and uses independent `-fsyntax-only` samples for public-header
frontend cost. `tools/measure/graph/services/run` verifies cache miss counts,
same-key async coalescing, and bounded sparse execution. Derived ratios remain
diagnostic; the measured component durations own the regression limits.

Every measurement pins `runD_DIR` to the Release install and requires its
source manifest to equal Release provenance before and after the workload.

## Sanitizers And Lifetime

Address mode uses
`-fsanitize=address,undefined -fno-omit-frame-pointer`; thread mode uses
`-fsanitize=thread -fno-omit-frame-pointer`. Both retain assertions, strict
warnings, and bounded source-line debug information. Address/UBSan covers the
complete instrumentable Kernel, Node, Accel, Math, and Cluster contract matrix
with halt-on-error settings.

TSan runs the broad Compute, Runtime Compute, accelerator Runtime Compute,
Task, and Reactor concurrency owners plus the registry-derived cross-thread
Runtime supplement. The registry remains the only case and source authority.

macOS ASan does not provide LeakSanitizer evidence. `tools/check/leaks` reuses
the fresh dependency-audited Release binaries after proving source provenance,
then runs coroutine-frame, CPU resident Compute, accelerator Compute, and
Reactor cleanup lifetime owners. It performs no compilation and records the
native reports under `.cache/evidence/leaks/`.

## Release Gate

`tools/release/run` configures the six subsystem contract owners, stages a
fresh installed prefix, and runs `package.consumer` through an external
`find_package(runD 1.0.3 EXACT CONFIG REQUIRED)` configure/build/run. The
external consumer cannot build the repository. Only its installed Compute
phase takes the accelerator lock.

`tools/release/darwin` invokes the shared Release route once, obtains platform
identity from `tools/internal/artifact/identity`, and delegates archive
construction, seals, embedded identity validator, adjacent `rund-verify`
self-test, second consumer run, and three-file candidate publication to
`tools/internal/package/candidate`. The Linux workflow uses the same identity
and candidate owners. Hosted release publication is a separate authority.

## Update Rules

- Keep test names and routes owner-centered.
- Add a new process only for a distinct isolation, resource, or failure
  boundary.
- Document only live behavior and live verification.
- Update this page and the owning subsystem docs when a live route, evidence
  schema, measurement workload, or acceptance invariant changes.
