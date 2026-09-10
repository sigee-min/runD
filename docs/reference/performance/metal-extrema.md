# Metal SIMD-group Min/Max

This is a current-source Release diagnostic on Apple M4 Pro, Darwin arm64,
AppleClang `-O3 -DNDEBUG`, native Metal. It is not installed-Release baseline
admission and does not claim a GPU-wide or Vulkan speedup. The implementation
contract is [Accel Reduce](../../../node/docs/contracts/accel/reduce.md).

## Workload and method

The public API builds an input-only U32 or U64 Flow followed by Min or Max,
compiles it, uploads input, and prepares one Pipeline. Inputs are the stored
width projection of `uint64(i) * 11400714819323198485 + 0x7123456789abcdef`
modulo 2^64. Every terminal result is checked against the host's independent
integer `min_element` or `max_element` oracle outside timing.

Each process runs the same twelve scenarios in a fixed order. Each scenario
has 60 untimed conditioning runs followed by 60 consecutive timed
`Pipeline::run()` calls. Preparation, upload and terminal read are excluded;
claims, command submission, GPU execution, completion wait and publication are
included. All individual samples are retained; none are retried or removed.
Three sequential ABBA cycles compare preserved before/after executables with
no concurrent build or test. Each side has six process summaries per scenario.
The table reports the median of six per-process p50/p95 values, not pooled
percentiles or confidence intervals. Each process p50 averages sorted samples
29/30 and p95 uses sample 56 (zero based).

| Type/op | Elements | Before p50 us | After p50 us | Ratio | Before p95 us | After p95 us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| u32/min | 257 | 139.041 | 141.688 | 0.98x | 205.938 | 218.020 |
| u32/min | 1,048,576 | 208.250 | 149.989 | 1.39x | 259.833 | 212.750 |
| u32/min | 16,777,216 | 1184.260 | 417.000 | 2.84x | 1896.541 | 683.333 |
| u32/max | 257 | 137.917 | 135.667 | 1.02x | 202.167 | 193.521 |
| u32/max | 1,048,576 | 205.438 | 139.510 | 1.47x | 272.708 | 196.750 |
| u32/max | 16,777,216 | 1204.750 | 412.406 | 2.92x | 1905.688 | 626.145 |
| u64/min | 257 | 138.291 | 118.500 | 1.17x | 191.416 | 201.855 |
| u64/min | 1,048,576 | 207.604 | 125.459 | 1.65x | 274.729 | 231.896 |
| u64/min | 16,777,216 | 1197.885 | 689.375 | 1.74x | 1895.688 | 1341.583 |
| u64/max | 257 | 139.323 | 138.302 | 1.01x | 191.938 | 205.145 |
| u64/max | 1,048,576 | 206.750 | 137.562 | 1.50x | 257.708 | 218.084 |
| u64/max | 16,777,216 | 1206.688 | 693.510 | 1.74x | 1902.750 | 1334.021 |

Large U32 extrema improve by 2.84–2.92x and U64 extrema by about 1.74x on this
host. The smallest shape does not show a consistent improvement and some p95
values regress. These observations do not establish a fixed-cost submission
improvement, a signed/fixed timing result, or a Sum, nonlinear Map, or Vulkan
performance improvement.

## Mechanism and invariants

One native SIMD group owns one complete logical reduction block. Lane-strided
loads cover that block once, native integer SIMD extrema produce one partial,
and partials retain the original offsets and pass order. At block width 256
and this pipeline's SIMD width 32, eight logical blocks share a physical
threadgroup. This removes all threadgroup barriers (formerly nine per logical
block) and all threadgroup arrays (formerly 3,072 declared bytes per group).
These are generated-source bounds, not measured occupancy or memory traffic.
The host queries the compiled pipeline's actual SIMD width; there is no
hard-coded 32-lane requirement.

U64 extrema use two U32 reductions with a winning-high-word mask; signed order
flips the high sign bit before comparison and restores it after selection.
The checked semantic cases cover both widths, signed/unsigned order, tied high
words, block widths 3/33/65/256, partial SIMD groups, tails, multiple physical
groups and multiple passes. Existing bounded-count, status and overflow
contracts remain required acceptance surfaces. Sum/CountNonzero arithmetic is
unchanged. The old Metal extrema tree and its unused overflow/source helpers
were removed rather than kept as a second execution authority.

Public graph fingerprints, terminal values, three large-shape dispatches (two
for 257 elements), and one command submission match before and after in every
measured process. Physical workgroup counts change as documented; logical
scratch and dispatch counts do not. No new warm allocation is introduced.

## Evidence and limits

Local packet: `.cache/gpu-next/bench.cpp`, `build-bench.py`, `compare.py`,
`extrema-before`, `extrema-after`, `abba-*.csv`, `comparison.json` and
`report.md`. All samples, output identities, counters and executable/harness
SHA-256 values are retained. The starting source is the tree verified in
`.cache/evidence/check/20260909T172630Z/source-manifest.tsv`. The before
executable was rebuilt with the original reducer source and source-helper
files verified against that manifest; the final comparator checks exact
public identities across both executables.

An exploratory 256-thread Map tuning regressed and was reverted. The existing
`--pipeline-profile metal` diagnostic failed its pre-edit topology assertion;
it supplies no timing evidence here. This independent public-API benchmark
validates its own exact outputs and counters. Installed Release admission,
sanitisers and Vulkan performance comparison are not part of this diagnostic.
