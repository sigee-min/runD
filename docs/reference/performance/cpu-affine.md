# Prepared CPU integer affine maps

This is a current-source diagnostic on Apple M4 Pro, Darwin arm64,
AppleClang Release `-O3 -DNDEBUG`, built-in 12-worker CPU pool. It is not an
installed Release baseline admission or a claim about all Pipeline workloads.
The implementation contract is [CPU SIMD](../../../node/docs/contracts/accel/cpu/simd.md).

## Workload and comparison

The public API builds one prepared `Pipeline` over a uint32 Map:

```
y = ((x * 3 + 7) * 5 - 11) * 9 + 13  (mod 2^32)
```

Inputs are `uint32(i) * 2654435761`, including wraparound. Shapes are 1,024,
65,536 and 1,048,576 elements. Each process performs 60 untimed conditioning
runs and 60 consecutive timed `Pipeline::run()` calls per shape. No sample is
retried or removed. Terminal output is read after timing; every element must
match the independent unsigned scalar expression, and the complete output
hash must agree across binaries. The timed interval includes Pipeline claims,
worker dispatch, execution and terminal publication, but excludes preparation
and the final read/verification.

Three ABBA cycles run the preserved before and after executables sequentially
on the same host after all compilation finishes. Each side therefore has six
independent process summaries per shape. Values below are medians of those six
process p50/p95 values; p50 averages order statistics 29/30 of 60 samples and
p95 uses statistic 56. They are diagnostic summaries, not a pooled percentile
or a confidence interval. Individual per-run timings are not exported by this
local harness; the six process summaries and binary hashes are retained.

| Elements | Before p50 | After p50 | p50 ratio | Before p95 | After p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 37.552 us | 36.761 us | 1.02x | 45.354 us | 46.917 us |
| 65,536 | 65.386 us | 32.313 us | 2.02x | 80.104 us | 43.521 us |
| 1,048,576 | 573.969 us | 47.407 us | 12.11x | 650.042 us | 56.834 us |

The smallest shape does not establish a useful improvement; its p95 increased
slightly. This result supports a large reduction in interpreter work for the
measured large dense affine map. It does not establish better fixed-point,
nonlinear, strided or GPU performance, or remove small-run scheduling costs.

## Implementation and cost

Cold preparation proves `a*x+b` forms using unsigned modular arithmetic,
snapshots aliased physical operand slots, and retains two coefficients plus an
admission flag. The warm dense path performs a SIMD multiply/add directly;
per-operation indirect calls and intermediate value commits disappear. A dead
Index emitted by public Map construction is allowed, while any consumed Index
retains the generic runner. Existing invocation overflow checks remain.

The same-width modular identity is exact for I32/U32/I64/U64. Fixed-point
rounding, saturation, nonlinear products, alternate widths and multi-output
ordering are not folded. Strided bindings use existing execution. Tail and
in-place ordering are checked against the existing runner and unsigned oracles.
Each prepared SIMD map retains a 24-byte affine record on this host; cold
integer admission uses temporary storage proportional to physical value slots.
The retained size is charged through the enclosing CPU program's actual size.
No warm allocation or extra native submission is introduced.

## Evidence and limits

Local artifacts: `.cache/perf-next/bench.cpp`, `build-bench.py`, preserved
`before`/`after` executables, `abba-*.csv`, and `comparison.json` (including
binary and harness SHA-256). The before source is the dirty worktree verified
by `.cache/evidence/check/20260909T165929Z/source-manifest.tsv`; the current
change is not an installed-SDK baseline update. The preparatory
`--resident cpu` observations measure cold resident setup and are not used for
the warm speedup claim.

Checked-in semantic cases in `node/tests/contract/accel/cpu/vector/plan/affine.hpp`
cover both integer widths, wrapping values, parameter/constant arithmetic,
nonzero invocation begin, tails, dense/strided and in-place views, canaries,
vector/tail counters and forced generic-runner parity. Existing full Pipeline,
CPU SIMD and runtime contracts remain required acceptance surfaces.
