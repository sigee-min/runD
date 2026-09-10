# Virtual Window Performance

This page owns `tools/measure/compute/run --virtual-window`. It measures the
public Direct pointwise VirtualPipeline shape admitted by the bounded Q2--Q4
execution window. Ordinary nonresident callback-backed accelerator rows use
Persistent fixed-W Forecast/Promote/Native/Drain/Persist when the backend
admits it, with rolling as the capability fallback; the receipt is not a
GPU-generated DeviceVsm recurrence claim.

The shared seed/expected-value fixture is owned by
`tools/measure/compute/virtual/window/support.cpp`. Measurement preparation is
owned by `virtual/window/prepare.cpp`; profile validation, paired sampling, and
CSV row production are owned by `virtual/window/measure.cpp`; and
`virtual/window/entry.cpp` owns the public three-cell coordinator. The
declarations/value seam is `virtual/window/local.hpp`. CSV schema publication
and the independent three-packet consensus parser are owned by
`virtual/window/report.cpp`; that reporting path carries no measurement state
and is not a second result authority.

## Workload

One process packet creates one CPU owner and one native Metal owner, each with
the same I32 map `(x + 5) * 3`, 4,096-element pages, three Device frames per
bank, and a logical capacity of 49,151 elements. The paired active counts are
24,575, 36,863, and 49,151 elements. They have exactly 6, 9, and 12 active
pages and therefore Q=2, Q=3, and Q=4.

Each owner is prepared once and reused for all three cells. A cell performs
one untimed conditioning run per backend, opens one sample epoch, and records
thirty `CPU,Metal,Metal,CPU` cycles. This produces exactly sixty unfiltered
wall samples per backend. One terminal output-backing read and one terminal
Profile projection per backend follow the sample epoch. The route runs three
complete process packets and publishes a robust winner only when all three
packet labels agree.

The row fields `terminal_reads=2` and `profile_projections=2` are paired
CPU-plus-Metal harness totals: each backend contributes exactly one of each.

Acceptance requires matching graph and output hashes, exact prefix output and
poisoned tail, unchanged plan geometry and fixed memory shape, and 60/60
runD-owned allocation-free terminals. CPU must report zero compute and
transfer submissions, zero uploaded/downloaded bytes, zero native in-flight
depth, zero directional overlap, and a zero window receipt. Metal must report
one compute submission, one public window handoff, one queue call,
`window_batch_count` equal to the sealed epoch count, bounded callback service,
zero uploaded/downloaded bytes, and nonzero measured stall. These are
Persistent fixed-W counters, not Q GPU-generated DeviceVsm epochs. Both
backends must report exact directional-overlap accounting and exact active
page/dispatch/output totals.

## Receipt Boundary

Every row carries
`route_shape=direct_pointwise_q2_4_persistent_fixed_w` and
`receipt_scope=public_persistent_fixed_w_receipt`. The public Profile
separately proves the Direct shape, zero-transfer path, one compute submission,
one bounded-window handoff, epoch-sized service batch and one queue call,
allocation result, and end-to-end wall time. The counters identify the
Forecast/Promote/Native/Drain/Persist route rather than a Q-submit DeviceVsm
schedule.

Q1, rolling, CPU, and a zero-submit pre-native fallback publish a zero window
receipt. A row is therefore admitted as actual bounded-window evidence only
when its backend-specific receipt is exact; command submissions alone are not
accepted as a proxy.

The existing [crossover surface](./method.md#crossover-surface) authors a
Clamp Window Sum graph and is structurally ineligible for the Direct
pointwise window. Its rows remain rolling-path and CPU no-regression reference
evidence rather than a W4 measurement.

## Result Admission

The raw three-packet stream belongs under `.cache/evidence/perf/window/`.
Results may be added here only after all three packets pass the semantic gates
above. Comparisons with older crossover packets are source-identity-mismatched
historical context, not a CPU no-regression decision. Current paired ABBA
samples own the only same-source CPU/Metal comparison. Accepted observations
belong in [Virtual Performance Results](./result.md), not in this procedure
owner.
