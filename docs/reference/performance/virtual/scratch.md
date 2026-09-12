# Graph execution scratch

This current-source Release diagnostic covers Graph execution's scratch and
lease ownership. The behavior owner is the
[Graph residency contract](../../../../node/docs/contracts/compute/residency/graph.md).
It changes no installed Release baseline and makes no GPU kernel timing claim.

## Reproduced problem and implementation

At clean `6b2ce3081af21c920d3000cded29c3506b3a2af7`, public prepared
`VirtualPipeline::run()` succeeded on the main thread but crashed with SIGBUS
on both 1 MiB and 512 KiB pthread stacks. Preparation remained on the main
thread; only execution ran on the worker. The same fixture now completes all
14 runs on each stack size with exact output and counters.

Four related duplicate-storage paths caused the excessive stack and serial work:

- Each of two Tickets copied Authority-owned lease tables. Tickets now borrow
  complete `EpochLease` views; terminal close clears the view and its identity
  together. Effects and output retention finish before close.
- Prefix and Terminal retained separate maximum-size projection tables.
  One Ticket-owned table now serves successive stages. Projection writes only
  the active prefix and commits metadata after validation; unused Prefix
  source/output mirrors were removed.
- Middle input evidence built a complete dummy Ticket. Evidence now consumes
  typed spans, preserving existing fetch/hit/byte/eviction meaning.
- Graph Promote retained up to seven identical whole-stage projections and
  rebuilt each one. A group now projects once and validates each source's
  range, owner, coordinate, ordered pages, and physical rows independently.

No heap workspace, physical frame, or memory-budget authority was added.
Authority remains the sole lease table owner. These are execution scratch
reductions, not reductions in allocated GPU payload or measured process RSS.

## Size and stack evidence

Host: Apple M4 Pro, Darwin ARM64, macOS 26.3, AppleClang 21; measured
2026-09-13 KST. Both binaries use the repository's Release closure and compiler
flags. Static frame sizes come from compiling the actual owners with
`-fstack-usage`; they are not complete nested call-stack peaks.

| Object or Release frame | Before (bytes) | After (bytes) |
| --- | ---: | ---: |
| `sizeof(Ticket)` | 572,096 | 139,568 |
| `sizeof(StageScratch)` | 43,832 | 88 |
| `execute_tiled_graph` | 1,166,192 | 301,072 |
| `project_ticket_impl` | 87,920 | 304 |
| `project_stage_scratch` | 44,080 | 432 |
| Middle evidence adapter | 572,288 | removed |
| `issue_graph_promote_group` | 287,552 | 41,488 |

The coordinator frame falls 74.2%; the Ticket object falls 75.6%. Debug
coordinator frames are 1,167,056 and 301,696 bytes. An intermediate version
that only removed Ticket lease copies passed 1 MiB but still failed 512 KiB.
LLDB located that remaining failure in Graph Promote group allocation. The
final shared projection removes that remaining reproduced failure.

## Repeated execution diagnostic

The fixture uses native Metal, four U64 inputs, 73 elements, 16 elements per
page, five stages and three K=2 batches. Input branches exceed the fusion
bound. The output implements two disjoint write lanes and naturally selects
the Host Graph route. Input invalidation forces the same backing reads before
every run. Preparation, invalidation and output/counter checks are outside
timing; synchronous public `run()` through publication is inside.

For each callback delay, three ABBA cycles run frozen before/after binaries.
Each process checks two conditioning runs and retains twelve timed runs:
72 timed samples per side per delay. No build or test overlaps measurement.
All 336 runs, including conditioning, produce exact output, 15 submissions,
20 input fetches, 2,336 backing-read bytes and 15 epochs. Every sample is
retained; there is no outlier filtering. The comparison runs on the main
thread to exclude worker creation and joining from execution timing.

| Input callback delay | Wall before / after (ms) | Coordinator CPU before / after (ms) |
| --- | ---: | ---: |
| None | 8.701 / 5.456 | 2.157 / 1.095 |
| 5 ms per non-prefix page | 79.741 / 71.679 | 2.945 / 2.233 |

These medians show 37.3% less wall time and 49.2% less coordinator CPU time
without callback delay. With delay, they show 10.1% and 24.2% reductions.
Each of the three ABBA cycles has the same direction for both metrics.
Wall samples vary substantially: 3.273–64.855 ms before and 2.284–57.808 ms
after without delay, and 60.047–165.197 / 62.414–170.186 ms with delay.
This diagnostic does not establish statistical significance or a universal
speedup. Coordinator CPU uses `CLOCK_THREAD_CPUTIME_ID` and excludes workers
and driver threads; wall uses `steady_clock`. GPU stages still submit/wait
individually. No GPU duration, GPU idle time, or kernel speedup is inferred.

## Verification and reproduction

Both focused commands passed on the final implementation:

- `tools/test/run --fresh compute.pipeline-residency`
- `tools/test/run --fresh compute.virtual-graph-residency-product`

The Authority contract keeps one Computing lease while admitting, activating
and rolling back another slot three times; it compares borrowed table
addresses, exact generation, and all binding/transition/port fields. The
maximum seven-source Promote contract bounds Group storage. Product tests
bound Ticket/view storage, validate input evidence and quiescent scratch
reuse, and run public CPU and native accelerator Graphs on a 1 MiB worker
stack. Existing four-input and shared-input tests also execute their controlled
failure/retry sequences on that worker stack. Stack limits here describe the
checked fixtures, not arbitrary user callbacks or every platform's ABI.

Local raw evidence is under `.cache/root-graph-scratch/`: preserved `before`
and `after` executables, `identity.json`, `measured-code.diff`, `*-before.su`,
`*-after.su`, `before-stack-results.json`, `after-stack-results.json`,
`after-unified-backtrace.log`, `abba/`, `abba-raw.json`, `abba-summary.json`,
and `counter-validation.txt`. `build-probe.py after .cache/release` links the
current public probe against the selected real Release closure. `abba.py`
replays the comparison; rebuilding `before` from current source would destroy
the original baseline. Full closure verification results are recorded in the
same packet's `verification-closure.json` and route logs.
