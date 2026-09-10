# Metal Histogram locality and contention

## Authority and change

The semantic and execution authority is
[`node/docs/contracts/accel/histogram.md`](../../../node/docs/contracts/accel/histogram.md).
Implementation belongs to `node/src/accel/metal/histogram/{local.hpp,source.cpp,
encode.mm,pipeline.mm,resources.mm}` and the histogram source recipe in
`node/src/accel/metal/kernel/run/source/recipe.mm`.

Previously every valid input issued a device atomic increment. For at most
256 bins, the count pass now uses a 1 KiB threadgroup-local array and a bounded,
coalesced grid-stride input walk. It flushes only nonzero local counts. At most
1024 groups issue one device atomic per occupied local bin. For larger bin
counts, uniform active SIMD cohorts share one device atomic; mixed cohorts
retain direct increments. Broadcast/vote proves cohort equality, with active
prefix/sum electing the writer and its count. No global partial matrix is added.

Cold specialization removes the local array and barriers from the larger-bin
executable. Count pipeline keys and source recipe identities distinguish both
modes; the clear pipeline remains shared. Source materialization and exact byte
accounting use the same emitter. This changes local memory usage, not the
retained status buffer or the public two-pass execution contract.

## Measurement boundary

This is a current-source Release diagnostic on Apple M4 Pro / arm64, measured
2026-09-10 KST. It is not installed-SDK admission and does not update
`baseline.tsv`. The preserved pre-edit executable and final executable each
link the repository Release closure. The workload uses the public resident
Pipeline API, prepares once, uploads once, performs 60 conditioning runs, then
60 timed `Pipeline::run` calls. Timing includes submission, completion, and
publication; preparation, upload and final verification download are excluded.

Twelve scenarios combine `N={257,1048579}`, `B={4,256,4096}`, and either all
indices equal to `B-1` (hot), or `uint32_t(i*2654435761u)%B` (uniform).
Every output bin is compared with the host integer oracle after each scenario.
Dispatch count (2), submission count (1), graph fingerprint and output hash must
match across executables. Three sequential ABBA cycles give six process
summaries per executable; the table reports the median of their p50 values.
All 60 raw timings per scenario/process are retained. No samples are filtered
or retried. Agent-owned builds/tests did not overlap the measurements; other
host activity was not controlled, so small differences are not broad guarantees.

## Observed large-input results

`N=1048579`; latency in microseconds. Speedup is before / after.

| Bins | Input | Before p50 | After p50 | Speedup |
| --- | --- | ---: | ---: | ---: |
| 4 | Uniform | 318.594 | 132.771 | 2.40x |
| 4 | Hot | 913.417 | 157.177 | 5.81x |
| 256 | Uniform | 263.760 | 119.156 | 2.21x |
| 256 | Hot | 913.552 | 159.260 | 5.74x |
| 4096 | Uniform | 206.500 | 178.750 | 1.16x |
| 4096 | Hot | 925.542 | 150.156 | 6.16x |

The small-input diagnostic measured 1.10–1.20x p50 ratios, but host noise and
fixed submission cost limit interpretation. These results support improvement
for the measured Histogram workloads, not all GPU programs or other devices.
There is no Vulkan speedup claim. Declared shared memory and atomic-count bounds
are source-level evidence, not measured occupancy or physical bus transactions.

An earlier Min/Max equality reduction regressed the dispersed 4096-bin case
and was rejected. The final broadcast/vote implementation was remeasured for
all scenarios; rejected results remain in the local evidence packet.

## Verification and evidence

`tools/test/run --fresh accel.kernel-core` owns semantic verification, including
one bin, 255/256/257-bin boundaries, hot and mixed cohorts, partial tails,
invalid first/last indices, repeated clearing, and inputs exceeding the capped
grid. Metal source-authority tests check exact source bytes for both modes and
cold manifest accounting. `tools/check/run` is the repository-wide gate.

The local reproducibility packet is `.cache/histogram-next/`: `bench.cpp`,
`build-bench.py`, preserved `before` and final `after`, `compare.py`,
`comparison.json`, `comparison.txt`, and twelve `abba-*.csv` files. The JSON
records binary/harness/comparator SHA-256 values. Rejected comparison evidence
is `comparison-minmax.json` and `minmax-abba/`. The final local `report.md`
records verification command results and source-manifest identity. Scratch
artifacts are local evidence, not a portable admitted baseline.
