# Math64 Docs

`/math64` owns public deterministic 64-bit high-precision vector math law for
`rund::math64`.

## Boundary

Math64 owns signed 64-bit raw-lane arithmetic with 63 fractional bits, 64-bit
geometry, 64-bit nonlinear and turn approximations, widened stat/reference
helpers, quantized
high-precision AI/probability/RL primitives, prepared SoA range surfaces, and
SIMD contract evidence.

Public math formulas are vector-first. The canonical production law is the
SIMD lane law for this product width. Element scalar values are storage and
test literals only; they are not public formula contracts or fallback modes.

Math64 does not own 32-bit real-time range APIs, kernel packets, node runtime
policy, cluster policy, or physics semantics.

## Contract Map

| Page | Owns |
| --- | --- |
| [SIMD](./contracts/simd.md) | Two-lane 64-bit carriers, masks, load/store, selection, arithmetic, comparisons, and explicit reductions. |
| [Fixed](./contracts/fixed.md) | Raw-lane constants, vector clamp, saturation, wrap, multiply, divide, reciprocal, integer sqrt, and widened intermediates. |
| [Geometry](./contracts/geometry.md) | 64-bit lane carriers, masks, SoA views, and componentwise vector laws. |
| [SoA](./contracts/soa.md) | Prepared component-array range APIs, alias rules, in-place rules, and vector tail policy. |
| [Nonlinear](./contracts/nonlinear.md) | `Exp2` and `Log2` approximation laws over 63-fraction-bit raw lanes. |
| [Turn](./contracts/turn.md) | 64-bit turn-space constants, sin/cos, ratio, atan, and atan2 laws. |
| [Stat](./contracts/stat.md) | High-precision vector mean, variance, RMS, interpolation, smoothstep, and Hermite laws. |
| [Quant](./contracts/quant.md) | Target-width clamp and requantization formulas where `I8`, `I16`, `U8`, and `I64` are semantic operand/result widths. |
| [NN](./contracts/nn.md) | Deterministic activation, dot, RMS norm, and RoPE row primitives. |
| [Prob](./contracts/prob.md) | Logit/probability max, softmax, logsumexp, logsoftmax, softplus, logsigmoid, and cross-entropy primitives. |
| [RL](./contracts/rl.md) | Bellman, TD error, Q update, return, and GAE recurrences. |

## Verification

`test_math64` is created only by the math64 contract route. The default
development configure does not expose the target or register its CTest row.
The commands below are an owner-local subsystem development loop only; they do
not provide Release, source-stability, installed-package, or artifact evidence.
Use `tools/release/run` for those repository-wide claims.

```sh
cmake -S . -B .cache/build/math64-contract-route \
  -DRUND_ENABLE_MATH64_CONTRACT_TESTS=ON
cmake --build .cache/build/math64-contract-route --target test_math64
ctest --test-dir .cache/build/math64-contract-route \
  -R '^math64\.contract$' --output-on-failure --no-tests=error
```

## SDK Artifact

Release packaging for this subsystem is owned by
[`/package`](../../package/README.md). This subsystem owns behavior and tests;
`/package` owns artifact layout, CMake package export, and external
consumption policy.
