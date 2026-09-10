# Memory Observation

## Pipeline Job attribution — 2026-09-09

This current-source diagnostic measures the Job attribution portion of the
Pipeline memory observer, which runs inside the caller's Pipeline gate. It is
not an installed Release baseline, GPU execution timing, or an end-to-end
simulation speedup claim. The behavior owner is the
[Compute Memory contract](../../../node/docs/contracts/compute/memory.md).

The host is Darwin ARM64 with AppleClang 21. The pre-edit source is the dirty
`ac79a09561fd7a04538b1b79b88110751bab4a19` worktree after the Graph planner
allocation refactor. Both executables compile the real `run/memory/jobs.cpp`
with `-std=c++20 -O3 -DNDEBUG`, link the same local closure libraries, and call
`measure_pipeline_jobs` directly. Fixtures have 1, 64, or 1,024 steps, empty
Job payload arrays, one shared Workspace, and either distinct Jobs per step or
one shared Job across all steps. Each alternate Job aliases its primary Job.
The observer receives a full step-profile destination.

Each process performs 1,000 observations per fixture. Three ABBA cycles run
after compilation finishes; these are medians of six process means per binary.
Every row verifies zero observed heap allocations, equal aggregate metadata,
and the same declaration-ordered profile digest before and after.

| Steps | Job ownership | Before | After |
| ---: | --- | ---: | ---: |
| 1 | distinct | 421.30 ns | 45.35 ns |
| 1 | shared | 391.40 ns | 45.85 ns |
| 64 | distinct | 2,633.15 ns | 2,267.75 ns |
| 64 | shared | 1,566.15 ns | 1,369.20 ns |
| 1,024 | distinct | 182,509.75 ns | 41,233.15 ns |
| 1,024 | shared | 21,243.80 ns | 20,868.60 ns |

The compiler's `-fstack-usage` output reports the observer's static frame as
50,320 bytes before and 18,032 bytes after. This is a compiler-specific frame
measurement, not peak process RSS or the entire nested call stack; the new
sort also has bounded helper frames. Retained Pipeline memory and Job/Workspace
attribution do not change. The fully shared 1,024-step row remains approximately
equal; the main scaling improvement is the distinct-Job case.

The removed work is the repeated scan of all prior Job pointers and full
initialization of maximum-capacity pointer arrays. A compact uint16 ordinal
scratch array groups identical pointers, retaining the minimum declaration
ordinal in a bitmap. Adjacent shared occurrences collapse before sorting.
Workspace attribution and saturating counters retain their existing behavior.

Local reproduction artifacts are under `.cache/memory-observer-refactor/`:
`bench.cpp`, `build.py`, `jobs-before.cpp`, `before.su`, `after.su`, `abba.txt`,
`result.json`, and `identity.json`. Rebuild the current diagnostic with
`python3 .cache/memory-observer-refactor/build.py .cache/memory-observer-refactor/after`.
The preserved old binary is the pre-edit comparison; rebuilding it from current
source would erase that baseline.

Verification passed: `tools/test/run --fresh compute.pipeline --backend cpu`
and `tools/test/run --fresh compute.pipeline`. The checked-in attribution
contract covers null, alternate-only, nonadjacent shared, maximum 4,102 Job
occurrences, exact per-step ownership, summary-only observation, and zero heap
allocation. An isolated ASan/UBSan executable additionally compares 100 mixed
ownership fixtures against the preserved pre-edit observer, checking every
step's MemoryStats and the complete aggregate. Full Release and repository-wide
sanitizer matrices were not rerun for this private observation-only change.

## Arena tables and shared observation — 2026-09-10

Two further changes use the existing Pipeline owners. Arena planning now seals
checked offset/owner table sizes in its validation pass and materializes each
table once. Shared observation charges an Arena when collecting its first
Workspace, retaining Arena deduplication across distinct Workspaces; it removes
the second traversal over primary/alternate Jobs. Both fixed pointer scratch
arrays read only their initialized prefixes instead of clearing maximum
capacity at every observation.

This is a current-source Darwin ARM64 / AppleClang 21 diagnostic compiled with
`-std=c++20 -O3 -DNDEBUG`, after the preceding Graph and memory-observer
refactors on dirty base `ac79a09561fd7a04538b1b79b88110751bab4a19`. It is not
installed Release evidence, GPU timing, retained payload reduction, or peak RSS.

Arena fixtures have 1, 64, or 1,024 CPU steps sharing a four-chunk Program with
16/32/48/64 words. Each process constructs 200 complete plans and verifies the
same ordered owner/offset digest. Global ordinary `operator new` measures
allocation calls and cumulative requested bytes, including discarded tables.
Shared-observer fixtures have 1 or 1,024 steps whose primary/alternate Job
shares one Workspace and one Arena. This is an internal backend-neutral host
observation fixture with a Metal tag and no DeviceOps/native objects; it does
not exercise a GPU. Each process makes 20,000 observations, requiring identical
MemoryStats and metadata and zero observed heap allocation.

Three ABBA cycles run after compilation finishes. Values below are medians of
six process means per side using preserved pre-edit and post-edit binaries.

| Fixture | Before | After |
| --- | ---: | ---: |
| 1,024-step Arena plan allocations | 27 | 7 |
| 1,024-step Arena plan allocated bytes | 155,616 | 90,144 |
| 1,024-step Arena plan time | 40,951.7 ns | 31,732.1 ns |
| 64-step Arena plan allocations | 19 | 7 |
| 64-step Arena plan allocated bytes | 9,696 | 5,664 |
| 64-step Arena plan time | 2,992.6 ns | 2,153.6 ns |
| Single-step Arena plan time | 244.7 ns | 238.2 ns |
| Single-step shared observation time | 320.9 ns | 27.8 ns |
| 1,024-step shared observation time | 4,618.25 ns | 3,397.85 ns |

The single-step Arena plan still allocates seven times / 120 bytes; it has no
repeated table growth to remove. The shared observer still retains its existing
Workspace identity checks and shared-buffer accounting; the optimization does
not replace those authorities with cached counters.

`tools/test/run --fresh compute.pipeline`, `tools/test/run --fresh
compute.window`, and `tools/test/run --fresh compute.virtual-graph-residency-product`
passed, including native Metal/Vulkan product paths. Checked-in memory contracts
verify every repeated-route prefix/owner/offset against the single-step layout,
and exact shared/distinct Arena attribution across repeated primary/alternate
Jobs. An isolated ASan/UBSan executable compares 200 layouts and 100 shared-owner
cases with the pre-edit owners, including typed failure and invalid-owner
saturation parity. Full Release and repository-wide sanitizer matrices were
not rerun for these private planning/observation changes.

Sources, raw `abba.txt`, `results.json`, source/compiler `identity.json`, old
owners, binaries, and verification logs are under
`.cache/pipeline-cost-refactor/`. The current diagnostic rebuild is
`python3 .cache/pipeline-cost-refactor/build.py .cache/pipeline-cost-refactor/after`.

## Empty Job attribution rows — 2026-09-10

The next diagnostic isolates counter assembly after first-Job attribution in
`pipeline/run/memory/jobs.cpp`. Steps containing only already attributed or
null Jobs previously merged seven zero counter categories while holding the
Pipeline observation gate. They now skip those merges and still overwrite a
supplied profile row, preserving first-owner identity and stale-row clearing.
The behavior owner remains the [memory contract](../../../node/docs/contracts/compute/memory.md).

The immediate pre-edit source is the dirty
`ac79a09561fd7a04538b1b79b88110751bab4a19` worktree after the Arena-table and
shared-observer changes above. On Darwin ARM64, AppleClang 21 compiles the real
owner with `-std=c++20 -O3 -DNDEBUG` against the same local closure libraries.
The fixture has empty Job payload arrays, one Workspace, primary/alternate
aliasing, and a complete profile destination. Each process makes 1,000 calls;
three ABBA cycles yield the median of six process means per executable.

| Steps | Job ownership | Before | After |
| ---: | --- | ---: | ---: |
| 1 | shared | 46.45 ns | 45.85 ns |
| 64 | shared | 1,368.70 ns | 636.60 ns |
| 1,024 | shared | 21,105.35 ns | 10,400.95 ns |
| 1,024 | distinct | 41,112.85 ns | 41,204.90 ns |

All observations allocate zero heap objects; aggregate metadata and ordered
profile digests agree. The improvement is serial observation work, not retained
memory, process RSS, accelerator execution, or end-to-end throughput. Distinct
Job timing is effectively unchanged. Checked-in attribution cases verify
truncated and reused profile destinations, null/repeated rows, saturated totals,
nonzero budgets, and allocation-free observation. A focused ASan/UBSan harness
compares all summary fields, metadata, and profile rows against the immediate
pre-edit owner in 400 cases. Raw samples, source snapshots, and build scripts
live under `.cache/job-empty-refactor/`.

## Scratch growth and ordered schedule — 2026-09-10

Two further cold-planner changes remove transient allocations. Scratch page
projection reserves the computed page count once. Schedule planning tests node
order before stable sorting; out-of-order publication accesses retain the same
stable sort. Behavior is owned by the [memory contract](../../../node/docs/contracts/compute/memory.md)
and [Pipeline contract](../../../node/docs/contracts/compute/pipeline.md).

Immediate pre-edit sources are preserved in `.cache/plan-growth-refactor/` from
the dirty `ac79a09561fd7a04538b1b79b88110751bab4a19` worktree after empty Job
attribution optimization. AppleClang 21 on Darwin ARM64 compiles the real two
owners with C++20, O3, and NDEBUG against identical local closure libraries.
Three ABBA cycles report the median of six process means. Scratch uses a fake
backend callback describing 4,096-byte pages without native allocation; each
process plans 1,000 fresh layouts. Schedule uses 1,024 ordered read/write steps
over one four-byte resource and makes 100 fresh plans per process.

| Fixture | Before allocations / requested bytes | After allocations / requested bytes | Before / after time |
| --- | ---: | ---: | ---: |
| Scratch, 64 pages | 7 / 2,032 | 1 / 1,024 | 436.58 / 281.13 ns |
| Scratch, 1,024 pages | 11 / 32,752 | 1 / 16,384 | 4,287.52 / 3,750.19 ns |
| Schedule, 1,024 steps | 4,116 / 1,372,744 | 4,115 / 1,258,056 | 9.766 / 9.842 ms |

Requested bytes are cumulative allocation requests, not peak live memory or
RSS. Schedule removes one 114,688-byte temporary allocation; its whole-planner
time is effectively unchanged, so no schedule throughput improvement is claimed.
Scratch page digests and schedule summary witnesses agree. Focused ASan/UBSan
old/new-owner comparison checks scratch descriptors and summaries plus schedule
step views, dependencies, barriers, lifetimes, and summaries at 1, 64, and 1,024
steps/pages. Pipeline and Window contracts cover native preparation and
publication ordering. These are local diagnostics, not installed Release or
end-to-end GPU performance evidence.

## Sealed ownership inventory — 2026-09-10

The [memory contract](../../../node/docs/contracts/compute/memory.md) now owns
one preparation-time inventory. Job/Workspace first attribution and immutable
metadata move out of observation. Shared Workspace/Arena/Buffer projection is
sealed in inclusive and Pool-exclusive forms. Warm observation retains live
native counters, bound Job payloads, frame/transfer/staging counters and indexed
CPU ownership validation. There is no warm reconstruction fallback. Builder and
complete preparation-plan definitions are no longer exposed by the runtime
state header; that compile-time cut preserves the existing consumed-Builder
lifetime rather than claiming a new runtime plan-release saving.

On Darwin ARM64, AppleClang 21 compiles the real owners with C++20, O3 and
NDEBUG. Immediate pre-edit owners from the dirty
`ac79a09561fd7a04538b1b79b88110751bab4a19` worktree are retained in
`.cache/ownership-cut/`. `observe.cpp` compiles both old and current owners
against the same current types/libraries and runs three ABBA cycles of 2,000
calls, reporting medians of six means. Fixtures have 1,024 steps, one Workspace,
empty Job payload arrays, and alternate Jobs aliasing primaries. A Metal backend
tag supplies no native objects or callbacks. Cold sealing is outside timing.

| Observation | Ownership | Before | After |
| --- | --- | ---: | ---: |
| Job summary only | One reused Job | 4,428.94 ns | 28.34 ns |
| Job summary only | 1,024 distinct Jobs | 35,240.85 ns | 17,929.01 ns |
| Shared ownership | One reused Job | 3,399.50 ns | 17.20 ns |
| Shared ownership | 1,024 distinct Jobs | 3,424.90 ns | 1,139.49 ns |

The separate complete-profile fixture uses CPU-tagged empty Jobs and 1,000
calls per process with the same three ABBA cycles. At 1,024 steps, reused-Job
attribution changes from 10,272.70 to 6,155.60 ns; distinct-Job attribution from
40,986.60 to 32,527.00 ns. Producing every profile row still requires writing
every requested row; the summary-only result must not be generalized to that
surface. Every timed fixture observes zero heap allocations.

The cost is **392 inline bytes per Pipeline plus 16 bytes per unique Job** on
this platform, and one cold allocation for a nonempty owner inventory. Thus
one reused Job adds 408 retained bytes, while 1,024 unique Jobs add 16,776 bytes.
Admission and public Host/Metadata include this cost. These are local observer
measurements, not end-to-end execution, GPU speed, or process RSS claims.

Validation compares 400 old/new Job cases and 400 shared-memory combinations
covering Pool inclusion, dynamic epochs and invalid-owner saturation under
focused ASan/UBSan. Metadata comparisons explicitly account for the new owner
vector. Checked-in contracts also cover first-owner profiles, stale and
truncated destinations, saturation/budgets, live prepared counters, corrupt CPU
references, and runtime-header exclusion of Builder/complete plan types.
The final-source Debug matrix passed 35 of 36 tests; `tools.measure` hit its
60-second parallel-run timeout and passed alone in 44.25 seconds. All 36 tests
are therefore verified, including native Pipeline/Window/Virtual Graph and
memory contracts. Full Release and whole-repository sanitizers were not rerun.
