# Tools

`/tools` contains the repository's operator commands. Product semantics stay
with the owning subsystem; tools select, execute, measure, and seal those
contracts without defining another product authority.

## Commands

| Command | Purpose |
| --- | --- |
| `tools/test/run` | Run the short Kernel, Node, Accel, and Compute development loop. |
| `tools/test/run <case>` | Run one exact Node case or exact CTest route. |
| `tools/test/run <case> --backend <cpu\|metal\|vulkan>` | Run the CPU row or the CPU oracle plus one selected native accelerator. |
| `tools/test/run --match <regex>` | Run the exact CTest rows selected by a regular expression. |
| `tools/test/run --list` | List canonical cases, targets, resources, and link profiles without building. |
| `tools/check/run` | Run the complete local Debug contract matrix. |
| `tools/release/run` | Run Release contracts and installed-SDK consumers. |
| `tools/release/darwin` | Produce and consume a sealed Darwin ARM64 SDK candidate. |
| `tools/check/platform/unavailable` | Prove unavailable-platform behavior and native discovery ownership without opening a device. |
| `tools/sanitize/run address` | Run the ASan and UBSan matrix. |
| `tools/sanitize/run thread` | Run the TSan concurrency-owner matrix. |
| `tools/check/leaks` | Run the native lifetime leak owners. |
| `tools/measure/scheduler/run` | Measure installed scheduler latency, scaling, and memory. |
| `tools/measure/compute/run` | Measure installed Compute execution, scaling, parity, and warm cost. |
| `tools/measure/compute/run --resident <cpu\|metal\|vulkan>` | Isolate current-source resident creation and transfer setup cost. |
| `tools/measure/compute/run --collective <cpu\|metal\|vulkan>` | Isolate current-source Release collective execution with the CPU oracle retained for native accelerators. |
| `tools/measure/compute/run --sort <cpu\|metal\|vulkan>` | Isolate current-source Release dense and bounded sparse map/sort execution with the CPU oracle retained for native accelerators. |
| `tools/measure/compute/run --bulk <cpu\|metal\|vulkan>` | Isolate current-source Release 1M-transform and 512-square-matrix execution with the CPU oracle retained for native accelerators. |
| `tools/measure/compute/run --batch <metal\|vulkan>` | Compare independently submitted resident Jobs with one prepared native batch. |
| `tools/measure/compute/run --pipeline <metal\|vulkan>` | Compare one prepared dependent Program Pipeline with the same Programs executed separately. |
| `tools/measure/compute/run --checkpoint <cpu\|metal\|vulkan>` | Compare live device state, reusable host checkpoint storage, and immutable snapshots. |
| `tools/measure/compute/run --pipeline-profile <metal\|vulkan>` | Isolate opt-in Pipeline step-profile cost from the otherwise identical disabled path. |
| `tools/measure/compute/run --recurrence <metal\|vulkan>` | Compare independently submitted recurrence with terminal-only and lossless prepared recurrence. |
| `tools/measure/compute/run --window-repeat <metal\|vulkan>` | Measure product-scale nested Seed/Action/Fold execution and sealed repetition. |
| `tools/measure/compute/run --plan-memory <cpu\|metal\|vulkan>` | Plan the product-scale preparation-memory workload and prove a one-byte-short budget rejects before materializing a Pipeline. |
| `tools/measure/compute/run --prepare-memory <cpu\|metal\|vulkan>` | Explicitly materialize that product-scale workload and report plan, backend-memory, failure-location, wall, and RSS evidence. |
| `tools/measure/compute/run --metal-icb-boundary` | Explicitly execute the native Metal 65,536-to-65,537 ICB chunk boundary in one submission; excluded from default tests. |
| `tools/measure/flow/run` | Measure installed Flow construction and C++ frontend cost. |
| `tools/measure/graph/services/run` | Measure installed cache, coalescing, and bounded graph services. |
| `tools/measure/telemetry/run` | Measure Disabled, Basic, and Detail telemetry cost and parity. |
| `tools/measure/build/run [build [target]]` | Measure cold/warm public-header frontend cost, live reverse fan-out, and target dirtiness. |
| `tools/source/manifest` | Hash the complete admitted product source and executable modes. |
| `tools/evidence/status [route ...]` | Validate the newest evidence packets against the current source manifest. |

Every public command accepts `-h` and `--help`. Help exits zero before path,
platform, cache, lock, or build work. Invalid syntax exits 2 with the same
canonical usage. The table above is the complete public command set.

## Development Loop

The default and `--match` routes reuse `.cache/dev`. Exact Node cases reuse
`.cache/focus/<profile>`. A sealed catalog accelerates lookup, but cannot admit
a case: the CMake registry, profile table, route table, generated index, and
post-build index must all agree on the selected owner.

The configured Ninja instance owns regeneration, dependency state, target
mutation, and exit status. A matching configuration identity skips only the
explicit configure step. Source bytes remain compiler inputs; changes to
source paths, filesystem kinds, CMake inputs, helpers, options, or tool
identity invalidate the appropriate graph proof.

CMake's standard `CMAKE_BUILD_PARALLEL_LEVEL` environment value, when set to a
positive integer, is forwarded to that same Ninja owner.
`CTEST_PARALLEL_LEVEL` independently bounds selected test processes. These
controls change only physical concurrency; target identity, selection metadata,
and verification order remain owned by the configured route.

Exact selection builds only the selected profile closure. The short loop does
not include the complete Runtime matrix; `tools/check/run` owns that closure.
Use the narrowest exact case that proves an edit, then widen according to
[Verification](../docs/architecture/verification.md).

The case contract runner owns validation order. Its semantic leaves separately
check the live registry surface, catalog and cache, profile focus, and
configuration state; inclusion alone has no side effects. Shared command
execution, row validation, and fixture construction have one model owner.

## Configuration And Locks

Common configure policy has one owner under `tools/internal/configure/`.
Local, sanitizer, Release, and platform profiles close every root option after
caller arguments. Verification profiles require Ninja and repository strict
warnings; those flags never propagate through the installed SDK target.

Each mutable build identity has one route lock and one nested build-state lock.
The route lock owns configure, build, and observation. The state lock owns one
CMake or Ninja mutation. Accelerator execution uses the separate repository
device lock. Lock capabilities are bounded, descriptor-backed, and fail closed
when ownership cannot be proven.

All command caches, build trees, logs, packets, and temporary artifacts live
under `.cache/`. They are disposable accelerators, never source authority.

## Evidence

Fresh Debug, Release, sanitizer, platform, package, and measurement routes
capture the product-source manifest around their work and require byte
identity. Evidence packets bind route result, toolchain, host, selection, raw
output, and source identity. `tools/evidence/status` validates packets; it does
not run work or promote stale evidence.

Measurements execute against the installed Release SDK. Raw logs are compared
by the single parser and current baseline authority described in
[Performance Method](../docs/reference/performance/method.md). A measurement
command never updates its own baseline, and a diagnostic span is not a product
speedup claim.

Each measurement executable owns CLI selection in its top-level `main.cpp`.
Scenario implementations live in the nearest semantic folder and share one
local model; one suite runner owns output order, and the target's explicit
CMake source list is the executable closure. A scenario-only edit dirties one
object plus the link; the CLI and output schema each retain one authority.

The focused `tools/measure/compute/run --resident`, `--collective`, `--sort`,
`--bulk`, `--batch`, `--pipeline`, `--checkpoint`, `--pipeline-profile`,
`--recurrence`, `--window-repeat`, `--plan-memory`, `--prepare-memory`, and
`--metal-icb-boundary`
modes are the deliberate exceptions to the installed-SDK route: they share one
incremental current-source Release build in `.cache/measure/compute/focus` for
the edit/measure loop and are not baseline evidence. This tree is disjoint from
the Debug exact-case tree in `.cache/focus/compute`; switching between a
semantic contract and a measurement therefore cannot invalidate the other
configuration's objects. Bulk uses one CPU oracle sample and fifteen native
measured samples; execution modes require their documented graph/output parity
and zero-warm contracts. The selected accelerator receives fifteen timed
samples in sort and bulk mode; the CPU oracle receives one unless it is the
selected backend. Bulk reports value traffic separately from logical
coefficient reads. Product-scale preparation measurement is plan-only under
`--plan-memory`: its sole `prepare()` attempt uses a one-byte-short budget and
must reject before allocation. Only the explicitly named `--prepare-memory`
mode may perform the large materialization.
The Apple-only `--metal-icb-boundary` mode is excluded from every default
target and CTest route. It builds one explicit native scale probe, encodes
65,537 real ICB dispatches as a 65,536-command full chunk plus a one-command
tail, writes immediately before that boundary, reads immediately after it, and
requires one direct boundary barrier, one outer submission, and the exact
written result.
Both preparation modes use two `Max=516096, Tile=8192` window groups (`N=64`,
then `N=1`)
with a publication-to-later-read consumer, an ordinary 64-iteration
recurrence, and final transactional publication. This public reduced shape is
140 templates/4,413 commands. The issue reports 243/4,454 for its full graph;
the checked-in harness covers the named interactions without claiming exact
graph equivalence or manufacturing unpublished work to match those counts.
Its accelerator row additionally prints the frozen and consumed semantic
fingerprint, logical host/native/source reservations, command/descriptor/native
object counts, recurrence terminal/history route counts and immutable-template
group capacities, and the primary/alternate/registry backend-memory subset.
The logical reservation columns include source-specialization and backend
finalizer host-transient high-water separately; those serialized workspaces are
not retained-current bytes.
Successful materialization also reports the largest current
`memory_snapshot()` retained group and its explicit Pipeline lifetime;
plan-only/failure rows keep that observation unavailable and retain the
separate planner workspace coordinates.
The materialization row additionally reports C++ `operator new` call count and
cumulative requested bytes during `prepare()`. These diagnose allocation
topology and requested payload only; they are not a retained high-water and do
not include C, Objective-C, or driver-private allocation.
It also reports compact step/status/telemetry description counts separately
from physical status/telemetry command counts and occurrence-local parameter
bytes. The harness requires every consumed structural field, recurrence
capacity, descriptor count, and template-native allocation count to equal the
matching frozen limit. These gates are
checked independently from physical Device allocation granularity and process
RSS attribution. The CSV keeps logical `peak_bytes`, exact CPU
`arena_extent_bytes`, and page-rounded Device `committed_peak_bytes` as separate
columns. The explicit materialization row requires its ending current RSS and
any newly established maximum RSS to fit preparation-start current RSS plus
`committed_peak_bytes`. Caller Buffers and compiled Programs already exist at
that baseline, so `persistent_bytes` is not added again. An earlier unchanged
process maximum is reported but is not misattributed to preparation. Allocator
metadata, runtime stacks, and opaque backend-private storage remain separately
named: if the whole observed preparation crosses the committed envelope, the
product row reports `process_rss_contract_failed` rather than laundering those
bytes into an exact plan component. `prepare_contract_failed` is reserved for
a failed plan-identity, memory-owner, backend-reservation, or telemetry
contract. A literal deployment hard cap remains an OS process/container
policy, not a second runD memory estimate.

## Ownership

- [Verification](../docs/architecture/verification.md) owns command semantics,
  source identity, mutable-state rules, and evidence closure.
- [Workflow](../docs/process/workflow.md) owns temporary paths and execution
  discipline.
- [Layout](../docs/architecture/layout.md) owns admitted roots and generated
  path policy.
- [Performance Method](../docs/reference/performance/method.md) owns workload
  schemas, comparisons, and performance claims.
- Subsystem docs and tests own product behavior; tools only route to them.
