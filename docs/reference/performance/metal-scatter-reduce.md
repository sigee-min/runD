# Metal Scatter Reduce cohort aggregation

## Scope and authority

The semantic owner is the Scatter Reduce section in
[`kernel/docs/contracts/compute/primitives.md`](../../../kernel/docs/contracts/compute/primitives.md).
The native execution owner is
[`node/docs/contracts/accel/scatter.md`](../../../node/docs/contracts/accel/scatter.md).
Metal changes belong to
`node/src/accel/metal/scatter/reduce/{source.cpp,source/parallel.hpp}`.
The expanded parity test also exposed incorrect ordered U64 conflict reporting
on the local Vulkan route (31 hot inputs reported 1 instead of 30).
`node/src/accel/vulkan/scatter/reduce/source.cpp` now accumulates ordered-fold
statistics privately and publishes once. This has no measured Vulkan speedup
claim and does not change the Vulkan parallel algorithm.

Uniform active SIMD cohorts combine their count/value atomics. All associative
32-bit cohorts sum collision statistics before updating their common counter.
The ordered fold accumulates statistics in its sole writer's register, without
reordering 64-bit or Fixed saturating arithmetic. Preflight's shared minimum
uses one atomic word and two barriers instead of a 256-word array and nine
barriers. The associative source emitter omits the unused ordered-fold helper.

Declared preflight threadgroup storage changes from 1024 to 4 bytes. Retained
device scratch, three dispatches, one submission and public graph identity do
not change. These are source/contract bounds, not measured occupancy or total
application memory reductions. One-workgroup input preflight remains and limits
scaling; no claim is made that this change removes all serialized GPU work.

## Measurement method

This is current-source Release diagnostic evidence, not installed Release
admission and not a `baseline.tsv` update. Host: Apple M4 Pro / arm64,
2026-09-10 KST. A preserved executable was built before production edits;
the measured after executable links the same Release closure after the Metal
shader changes. The subsequent Vulkan-only correction leaves these measured
Metal source files unchanged; its correctness is verified separately, and the
measured executable is not used as evidence for corrected Vulkan behavior.

The public Pipeline is prepared once with resident input/output buffers.
Sixty conditioning runs precede sixty timed `Pipeline::run` calls. Timing
includes claims, submission, completion and publication; preparation, uploads,
and the final verification read are excluded. There are 18 scenarios:
U32 Sum/Min/Max, N=257 or 1048579, with uniform, hot, or unique destinations.
Values are `uint32_t(i*2654435761u)`. Uniform destinations are that same value
modulo 4096; hot destinations all equal 4095; unique destinations equal i and
have output extent N. Every output slot is checked against the host oracle.
Conflict count is checked against N minus the number of occupied targets.
Output hash, conflict count, graph fingerprint, three dispatches and one
submission must match between binaries.

Three sequential ABBA cycles retain six process summaries per side, with every
raw sample preserved. Reported p50/p95 values are medians of those process
quantiles. No sample filtering or retries. Builds/tests did not overlap the
comparison; other host activity was uncontrolled. Small differences are not
interpreted as reliable speedups or regressions.

## Results

N=1048579, latency in microseconds, speedup = before / after.

| Operation | Destinations | Before p50 | After p50 | Speedup |
| --- | --- | ---: | ---: | ---: |
| Sum | Uniform | 1193.573 | 1214.469 | 0.983x |
| Sum | Hot | 1941.865 | 1080.125 | 1.80x |
| Sum | Unique | 1280.271 | 1252.646 | 1.02x |
| Min | Uniform | 1208.114 | 1188.532 | 1.02x |
| Min | Hot | 1976.218 | 1084.062 | 1.82x |
| Min | Unique | 1262.385 | 1237.833 | 1.02x |
| Max | Uniform | 1197.333 | 1184.062 | 1.01x |
| Max | Hot | 2008.292 | 1066.990 | 1.88x |
| Max | Unique | 1251.812 | 1233.396 | 1.01x |

Hot-input p95 improves from 2688/2602/2650 to 1518/1578/1586 us for Sum/Min/Max.
Small-input ratios are 0.986–1.023x. The measured gain is concentrated-target
throughput/latency; dispersed and unique cases show no material improvement.
There is no Vulkan, other-device, 64-bit latency, or universal GPU speedup claim.

## Reproducibility and verification

The local `.cache/scatter-next/` packet contains `bench.cpp`, `build-bench.py`,
`before`, `after`, `before-source.cpp`, `compare.py`, `comparison.json`,
`comparison.txt`, and twelve `abba-*.csv` files. Binary/harness/comparator hashes
are recorded in the JSON. `report.md` records source identity and verification.
The pre-edit source is also recorded in the complete prior passing snapshot
`.cache/evidence/check/20260910T014732Z/source-manifest.tsv`.

`tools/test/run --fresh accel.kernel-core` checks typed Sum/Min/Max, signed
extremes, modular overflow, Fixed saturation, mixed and uniform cohorts,
partial tails, invalid indices across cohorts and repeated clearing.
`tools/test/run --fresh compute.pipeline --backend metal` checks public output
and exact U32/U64 collision statistics for 31/32/33/257/1027 inputs,
repeated runs, and the first failing ordinal across invalid cohorts.
`tools/test/run --fresh compute.pipeline --backend vulkan` verifies the
ordered conflict-count correction. `tools/check/run` owns the complete
repository regression gate.
