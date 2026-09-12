# Graph input Forecast window

This record covers a dependency-driven two-worker Host input window and its
completion notification. The behavior owner is
[Forecast residency](../../../../node/docs/contracts/compute/residency/execution/forecast.md#ready-horizon).
It is a current-source Debug mechanism diagnostic, outside installed Release
admission. It changes no performance baseline.

## Root cause and bounds

The previous eligibility and issue paths encoded three inputs and four stages,
with one fixed middle-stage pair. A graph with three independent middle inputs
could not refill the first available worker through that path. The coordinator
also repeatedly queried worker readiness and yielded while callbacks were
outstanding. Callback latency therefore consumed coordinator CPU time.

`graph/reduce/parallel.cpp` now proves independence from the sealed graph's
same-batch predecessors; `prefetch/stage/refill.cpp` admits the next ready
input into each free lane. Logical batch/stage/resource identity is separate
from the physical worker slot. Every receipt still closes its Authority lease
before the worker is reused. The same backing cannot own two simultaneous
reads. The scan also skips unavailable backings instead of letting one shared
input hide later independent work. Completed Host frames remain pinned until
their Ready rows are consumed; callback return alone cannot authorize reuse.
There remain two workers, two Forecast slots, and the existing Host
frames. A fixed atomic wake sequence and two borrowed pointers add 24 bytes
of owner fields on this ARM64 host; they add no payload queue or per-run
allocation. This is a field-size bound, not an RSS reduction claim.

The new failure/retry fixture also exposed an independent recovery defect:
accelerator tickets were considered unclean because they did not own a
CPU-only receipt book. Recovery now requires that book only for CPU tickets;
native, Forecast, callback-return, and quarantine checks still apply.

## Method

Host: Apple M4 Pro, ARM64, 12 CPU cores, 24 GiB, macOS 26.3 (25D125),
2026-09-12 KST. Both sides use native Metal, Debug, the public prepared
`VirtualPipeline`, 73 U64 elements, 16 elements per page, five pages and
three K=2 batches. The three-input program is the existing
`graph_wavefront_host` fixture; the four-input program is the new
`graph_forecast_window` fixture. Each input branch exceeds the fusion bound.
The latter output honestly implements two disjoint write lanes, so both old
and new implementations naturally admit the Host route without a private
route override. The functional regression separately uses an ordinary output.

The before closures were frozen from `ab78292f1ad7cf1c46e9fd843ab9353808f507c4`
before rebuilding the development tree. The after closures link the rebuilt
compute-focus tree including recovery and shared-input pin protection. Per shape/delay, process order
is before/after/after/before. Each process prepares once, checks two
conditioning runs, and retains twelve timed runs. Values below are medians
of all 24 timed samples per side, without filtering. Public input invalidation
before each run forces the same backing reads while retaining the prepared
pipeline. Timing includes synchronous `run()` through publication; preparation,
invalidation and exact output/counter checks are outside the interval.
No build or test overlaps the measurement.

Non-prefix input callbacks inject either 5 ms per page or no delay. Wall time
uses `steady_clock`; coordinator CPU time uses `CLOCK_THREAD_CPUTIME_ID` and
excludes worker/driver threads. Every run validates exact output, 12 or 15
submissions and stage epochs, 15 or 20 reads, and 1752 or 2336 read bytes.
No GPU kernel duration or GPU-idle interval is inferred from Host timing.

| Inputs | Callback delay | Wall before / after (ms) | Coordinator CPU before / after (ms) |
| --- | --- | --- | --- |
| 3 | 5 ms/page | 39.741 / 36.836 | 31.316 / 2.432 |
| 3 | None | 3.934 / 3.949 | 1.729 / 1.761 |
| 4 | 5 ms/page | 106.703 / 67.569 | 4.287 / 3.654 |
| 4 | None | 6.085 / 5.603 | 2.722 / 2.562 |

The delayed three-input case reduces coordinator CPU time by 92.2% and
wall time by 7.3%. The delayed four-input case reduces wall time by 36.7%
(1.58x throughput for this synchronous workload). No-delay wall medians are
0.4% higher and 7.9% lower. Parking avoids busy waiting but does not prove
lower wakeup latency for every run or shape.

All diagnostic rounds remain visible. The initial round before ordinary-worker
notification cleanup is retained in `abba-initial/` and `*.initial`: it observed
90.5% coordinator CPU reduction and 34.3% four-input wall reduction, with
no-delay wall increases of 10.6% and 3.4%. The intermediate `aed67462` round is
retained in `abba-aed67462/` and `*.aed67462`: it observed 87.0% coordinator CPU
reduction, a 10.6% delayed three-input wall increase, 38.0% delayed four-input
wall reduction, and no-delay wall changes of -0.2% and +5.3%. The table uses the
last source state, not a best-of selection; this cross-round variability limits
small wall-time claims.
This limited diagnostic does not establish statistical significance,
Release performance, other hardware, generic GPU acceleration, or 100x.
Host graph stages still submit and wait individually.

Local raw evidence is under `.cache/root-forecast/`: `probe.cpp`,
`probe4.cpp`, `build-probe.py`, `build-probe4.py`, `run-abba.py`, `abba/*.csv`,
`abba-summary.json` and `abba-binaries.json`. The first preliminary after
binary accidentally linked the old development tree; its `stale-after*`
artifacts are excluded. The final ABBA sequence uses the rebuilt focus closure.

| Executable | SHA-256 |
| --- | --- |
| Three-input before | `2eb153ad4d24298e0081c02db41a44e7252d0349f3bae512e25d16fdcd58eb68` |
| Three-input after | `30e30566b49e0ab5611e0b19cf61cb2e6becac750e3ee7dcd5e6451be77c73ea` |
| Four-input before | `889de41a3bbe7088e558854f8dd80eefcc1ddf2b489400c8487a6f36c70126bd` |
| Four-input after | `5445b5ae9328afc1e33d44bde3b4b8ab521e71d47d6b0cb43c957e4e6c50c705` |

## Semantic evidence

`tools/test/run --fresh compute.virtual-residency-product` passes on CPU,
native Metal and MoltenVK. The new four-input/five-stage fixture blocks one
middle input until a third middle input starts: waiting for the original pair
cannot pass. It checks two active callbacks at most, exact output/tail/read
bytes, fixed retained Pool sizes, quiescent workers, one version publication,
known input failure with unchanged bytes/version, and reuse of the same
pipeline after failure. The old recovery code failed that retry with
`PipelinePoisoned`; the corrected implementation passes.
An isolated negative-control binary that restores a pair-wide barrier fails
the new fixture as expected (`negative.log`, exit 8); it never changes the
production source or the admitted timing samples.

`tools/test/run --fresh compute.pipeline-residency` passes the shared
notification contract: reversed completion, completion before parking,
128 exact-token slot reuses with interleaved failures, and rejection of
rebinding a configured worker to another notification owner. A deadline makes
a lost notification fail the test instead of hanging indefinitely. These are
contract assertions, not timing thresholds or synthetic performance counters.

The shared-input variant reuses one logical input in two middle stages. Before
the fix it blocked the later independent input (`shared-input.log`, exit 5).
It now passes the same failure/retry, output, capacity and publication checks
with 15 reads / 1752 bytes. Removing only the Ready-pin protection in an
isolated negative-control binary fails as expected (`negative-ready.log`, exit
8). The fixture's handshake accepts either legal callback start order.

Full TSan execution also exposed a separate timed-observation defect: Compute
status was complete before the Scheduler had published its terminal cell.
`Submission::wait_for` could therefore return Pending before its requested
timeout. The [Runtime owner](../../../../node/docs/contracts/runtime.md)
now delegates timed observation to the same CompletionPool used by `poll`,
using its existing generation, mutex and condition variable. No second
completion state or public observer field is added. The controlled
`runtime.compute-host` regression holds the real Scheduler cell in Committing
after Compute completion, checks the timeout lower bound, then checks delayed
success and exact failure publication. Restoring the old early-return
predicate fails that regression (`negative-observation.log`, exit 1).
