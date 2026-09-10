# Indexed execution and route ownership

## Scope and authority

This record covers CPU Scatter Reduce scratch, Metal and Vulkan indexed
preflight, Virtual execution ownership, and the measured decision to retain
the Device claim gate. These are current-source Release diagnostics, outside
installed Release admission. They do not update `baseline.tsv`.

The semantic and planning owner is
[Kernel primitives](../../../kernel/docs/contracts/compute/primitives.md),
implemented by `kernel/program/compute/scatter/reduce/cpu/{plan,execution}.hpp`
and `kernel/program/compute/indexed/preflight.hpp`. Prepared CPU execution and
the legacy CPU adapter consume the same strategy. The native API owners are
[Gather](../../../node/docs/contracts/accel/gather.md) and
[Scatter](../../../node/docs/contracts/accel/scatter.md).
[Virtual routes](../../../node/docs/contracts/compute/residency/execution/routes.md)
owns the topology/proof hard cut. No public operation, arithmetic law, graph
fingerprint, failure precedence, or ownership transaction was changed.

## Execution and memory model

For admitted input capacity N and output extent O, the CPU strategy uses
`ceil(O/32)` U32 bitmap words when that count is at most N; otherwise it retains
N sorted U32 keys. Dense preflight is `O(N + ceil(O/32))` rather than the old
`O(N log N)` key sorting. Consecutive accesses to one bitmap word accumulate in
registers; each newly set bit contributes once to occupied-target count.
Conflicts remain exactly `logical_count - occupied_targets`. The arithmetic
fold still visits source ordinals in order, including Fixed saturation.
No caller output is changed before complete index validation. Every nonempty
bitmap run clears its full admitted scratch span, including after a rejected
run; warm inputs cannot inherit earlier target membership.

At N=1048579 and O=4096 the scratch payload is 4194316 bytes before and 512
bytes after. At O=N it is 131076 bytes after. These are exact planned payload
bounds, excluding arena alignment and other live resources; they are not RSS
or total application memory measurements. Sparse output shapes keep the
bounded sorted policy and have no measured sparse-case speedup claim.

Native preflight keeps one group for capacity at most 65536. Above that bound,
`G = min(ceil(N/4096), 1024)` groups write independent first-invalid minima into
the existing status buffer's tail. An API buffer barrier precedes one terminal
group that consumes all G minima and alone publishes status and payload
dispatch arguments. Group arrival order cannot change the minimum. There is
no cross-group spinning, extra pipeline, descriptor set, or buffer allocation.
The additional status payload is `4G` bytes, bounded by 4096 bytes. At the
measured N=1048579, G=257 and the additional payload is 1028 bytes. Large Gather
uses three passes instead of two; large Scatter Reduce uses four instead of
three. Both retain one submission. Count overflow still precedes index errors,
and invalid preflight cannot mutate output.

## Measurement boundary

Host: Apple M4 Pro, arm64, 12 CPU cores, 24 GiB, macOS 26.3 (25D125),
2026-09-10 KST. The before executables were preserved before this structural
change; the after executables link the rebuilt `runD-compute-focus` Release
closure. This is a same-host diagnostic comparison, not a sealed installed-SDK
baseline or evidence for another machine.
The measured implementation point is `24aaf1d7`. Subsequent fixes cover Runtime
portability, SDK-disabled interface closure, and DeviceVsm completion lifetime.
The tables below remain observations of that measured point, not newly sampled
timings. No latency improvement is inferred from the completion lifetime fix.

Each public Pipeline is prepared once with resident buffers. Timing covers
`Pipeline::run()` through completion/publication. Preparation, input upload,
and the final output read are excluded. CPU uses 20 conditioning and 20 timed
runs per scenario; Metal uses 60 conditioning and 60 timed runs. Three
sequential ABBA cycles retain six process summaries per side. Reported p50 and
p95 are the medians of those process quantiles, not a confidence interval.
No samples are filtered or retried. Builds, tests, and compiler audits do not
overlap the comparison; unrelated host activity is uncontrolled.

Scatter scenarios are U32 Sum/Min/Max at N=257 and N=1048579. Values are
`uint32_t(i*2654435761u)`. Uniform targets are that value modulo 4096, hot
targets are all 4095, and unique targets are i with O=N. Gather uses U32,
source/output extent N, and N=257/65536/65537/1048579. Its uniform targets are
the same wrapped value modulo N; hot targets are N-1; unique targets are i.
Every output slot is compared to the host oracle. The comparator requires
unchanged output hash, graph fingerprint low word, submission count, and
conflict count; Scatter also checks exact conflicts against occupied targets.
Dispatch counts must match the frozen plan, including the extra large-input
native preflight pass.

## Measured results

Latency is in microseconds; speedup is before p50 divided by after p50. All
scenario summaries are retained below, including slower observations.

### CPU Scatter Reduce

| Operation | Targets | N | Before p50 | After p50 | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Sum | Uniform | 257 | 1.729 | 0.709 | 2.441x |
| Sum | Uniform | 1048579 | 9238.937 | 739.896 | 12.487x |
| Sum | Hot | 257 | 1.042 | 0.792 | 1.315x |
| Sum | Hot | 1048579 | 3033.333 | 1856.635 | 1.634x |
| Sum | Unique | 257 | 0.542 | 0.333 | 1.628x |
| Sum | Unique | 1048579 | 1593.740 | 762.416 | 2.090x |
| Min | Uniform | 257 | 1.730 | 0.698 | 2.479x |
| Min | Uniform | 1048579 | 9232.156 | 748.125 | 12.340x |
| Min | Hot | 257 | 0.958 | 0.771 | 1.243x |
| Min | Hot | 1048579 | 3204.812 | 2030.865 | 1.578x |
| Min | Unique | 257 | 0.542 | 0.333 | 1.628x |
| Min | Unique | 1048579 | 1599.958 | 738.896 | 2.165x |
| Max | Uniform | 257 | 1.750 | 0.667 | 2.624x |
| Max | Uniform | 1048579 | 9282.688 | 769.885 | 12.057x |
| Max | Hot | 257 | 0.979 | 0.791 | 1.237x |
| Max | Hot | 1048579 | 3215.896 | 1992.969 | 1.614x |
| Max | Unique | 257 | 0.573 | 0.333 | 1.720x |
| Max | Unique | 1048579 | 1594.125 | 734.511 | 2.170x |

### Metal Scatter Reduce

| Operation | Targets | N | Before p50 | After p50 | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Sum | Uniform | 257 | 169.344 | 199.281 | 0.850x |
| Sum | Uniform | 1048579 | 1265.562 | 391.260 | 3.235x |
| Sum | Hot | 257 | 142.667 | 155.323 | 0.919x |
| Sum | Hot | 1048579 | 1129.833 | 231.865 | 4.873x |
| Sum | Unique | 257 | 133.750 | 123.730 | 1.081x |
| Sum | Unique | 1048579 | 1340.875 | 338.167 | 3.965x |
| Min | Uniform | 257 | 143.094 | 136.156 | 1.051x |
| Min | Uniform | 1048579 | 1224.302 | 418.688 | 2.924x |
| Min | Hot | 257 | 146.385 | 124.458 | 1.176x |
| Min | Hot | 1048579 | 1138.375 | 245.146 | 4.644x |
| Min | Unique | 257 | 144.636 | 122.584 | 1.180x |
| Min | Unique | 1048579 | 1344.292 | 328.250 | 4.095x |
| Max | Uniform | 257 | 135.291 | 130.114 | 1.040x |
| Max | Uniform | 1048579 | 1248.906 | 424.427 | 2.943x |
| Max | Hot | 257 | 141.677 | 123.469 | 1.147x |
| Max | Hot | 1048579 | 1125.500 | 224.031 | 5.024x |
| Max | Unique | 257 | 144.229 | 121.500 | 1.187x |
| Max | Unique | 1048579 | 1331.958 | 328.406 | 4.056x |

### Metal Gather

| Operation | Targets | N | Before p50 | After p50 | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Gather | Uniform | 257 | 156.448 | 172.114 | 0.909x |
| Gather | Uniform | 65536 | 287.333 | 293.021 | 0.981x |
| Gather | Uniform | 65537 | 293.125 | 179.761 | 1.631x |
| Gather | Uniform | 1048579 | 1276.677 | 244.010 | 5.232x |
| Gather | Hot | 257 | 142.594 | 124.062 | 1.149x |
| Gather | Hot | 65536 | 198.740 | 182.781 | 1.087x |
| Gather | Hot | 65537 | 166.125 | 133.614 | 1.243x |
| Gather | Hot | 1048579 | 1192.896 | 197.386 | 6.043x |
| Gather | Unique | 257 | 139.750 | 111.073 | 1.258x |
| Gather | Unique | 65536 | 166.541 | 209.896 | 0.793x |
| Gather | Unique | 65537 | 163.625 | 156.531 | 1.045x |
| Gather | Unique | 1048579 | 1201.604 | 252.698 | 4.755x |

Large CPU Scatter Reduce improves by 1.578–12.487x across the measured
operations and target patterns. Large Metal Scatter Reduce improves by
2.924–5.024x; large Metal Gather improves by 4.755–6.043x. Large Sum p95 changes
from 9461.062 to 798.896 us on CPU uniform input, and from 1510.250 to 504.812 us
on Metal uniform input. Large uniform Gather p95 changes from 1496.312 to
353.313 us. These end-to-end observations support the large-input improvement;
they do not measure occupancy or attribute every microsecond to the shader.

Small native observations span 0.793–1.258x. In particular, N=257 uniform Metal
Scatter Sum is 17.7% slower and N=65536 unique Gather is 26.0% slower in the
aggregate. Those before/after process medians span 145.562–218.834 /
134.500–268.979 us and 161.688–199.270 / 165.292–408.459 us, respectively.
The retained observations do not isolate shader cost from submission and host
scheduling. No small-input improvement or regression-free behavior is claimed;
these slower observations are not discarded or replaced by selected reruns.

## Virtual structure and claim gate

`VirtualRunProjection` now carries one exclusive logical topology instead of
seven independent booleans. Pipeline selection consumes that topology directly
and no longer constructs a synthetic 2136-byte projection. The selected
physical route is an inline variant whose active payload alone owns endpoint,
page count, frame capacity, and optional Window preflight. Prepared execution
no longer mirrors page count. Route stamps retain the existing serialization
order, and the same admission proof is consumed by cold and warm execution.

On this host, `VirtualDeviceVsmRouteProof` changes from 136 to 120 bytes with no
heap allocation. `VirtualRunProjection` remains 2136 bytes because of its
alignment and other fields; removing flags is not a total-size reduction claim.
These changes do not add previously unsupported native Graph routes, and no
Virtual latency speedup is inferred from the size comparison.

The claim probe used one CPU Device and disjoint prepared 16-element U32 Map
pipelines, with 1/2/4/8 callers. Each of three observations per caller count
used 1000 conditioning runs and 20000 runs per caller. Mean public
`pipeline.claim_ns` ranged from 35.676 to 82.342 ns/run. This did not establish
a dominant claim-gate bottleneck, so the conditional partition proposal was
not activated. The existing all-or-nothing transaction gate remains the single
claim authority. This diagnostic does not prove contention is absent in other
workloads, and no claim-sharding improvement is reported.

## Reproducibility and verification

The task-local `.cache/structural-next/` evidence contains the three preserved
before binaries, final after binaries, harnesses, build scripts, comparator,
36 unmodified ABBA CSV logs, and `cpu/gpu/gather-comparison.json`. The JSON
records per-process quantiles, semantic identities, dispatch expectations,
and SHA-256 identities of each binary, harness, and comparator. The earlier
`cpu-after` executable linked stale archives and is excluded; only the rebuilt
`cpu-after-final` participates in the final comparison. Claim observations and
the size probe are retained separately. `report.md` names the exact closure
commands, results, source identity, and remote CI run.

Kernel contracts cover all six CPU numeric modes, Sum/Min/Max, bitmap-word
boundaries, sparse strategy, dirty scratch reuse, canaries, failure precedence,
and forged/short plans. Native and public Pipeline contracts cover both sides
of the 65536 threshold, warm reuse, earliest errors, and G=257 with the first
invalid ordinal in partial slot 256. Virtual tests exercise selected proof
payloads, invalid capacities, Window overflow, stamp mutation, route matching,
and existing cold/warm product paths. The complete repository gate is
`tools/check/run`; native focused commands are `accel-device --case
accel.kernel-core` and `node-compute-accel --backend metal|vulkan --case
compute.pipeline` from `.cache/dev/node/`.

Linux CI owns actual Clang Debug contracts, GCC Release installed-SDK
consumption, and the no-device platform boundary. The local GCC audit compiles
the changed C++ translation units at O3 with warnings as errors and optional
GPU SDKs disabled; it is a compiler/SDK-boundary check on macOS, not Linux
execution evidence. Vulkan functional tests on this host use MoltenVK. They
verify generated SPIR-V, command ordering, barriers, and output semantics on
that path; they do not establish native Vulkan throughput. No NVIDIA/AMD,
64-bit GPU latency, or universal speedup is claimed.
