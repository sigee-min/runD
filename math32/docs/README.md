# Math32 Docs

`/math32` owns public deterministic 32-bit real-time vector math law for
`rund::math32`.

## Boundary

Math32 owns signed 32-bit raw-lane arithmetic with 31 fractional bits, 32-bit
geometry, 32-bit nonlinear and turn approximations, 32-bit stat helpers,
quantized AI/probability/RL
primitives, prepared SoA range surfaces, and SIMD contract evidence.

Public math formulas are vector-first. The canonical production law is the
SIMD lane law for this product width. Element scalar values are storage and
test literals only; they are not public formula contracts or fallback modes.

Math32 does not own 64-bit high-precision laws, kernel packets, node runtime
policy, cluster policy, physics semantics, or target-specific intrinsic
selection.

## Contract Map

| Page | Owns |
| --- | --- |
| [SIMD](./contracts/simd.md) | Four-lane 32-bit carriers, masks, load/store, selection, arithmetic, comparisons, and explicit reductions. |
| [Fixed](./contracts/fixed.md) | Raw-lane constants, vector clamp, saturation, wrap, multiply, divide, reciprocal, sqrt, and private widened intermediates. |
| [Geometry](./contracts/geometry.md) | Lane carriers, masks, SoA views, and componentwise vector laws. |
| [Nonlinear](./contracts/nonlinear.md) | `Exp2` and `Log2` approximation laws over 31-fraction-bit raw lanes. |
| [Turn](./contracts/turn.md) | Turn-space constants, sin/cos, ratio, atan, and atan2 laws. |
| [Stat](./contracts/stat.md) | Mean, variance, RMS, interpolation, smoothstep, and Hermite laws. |
| [SoA](./contracts/soa.md) | Prepared component-array range APIs, alias rules, in-place rules, and vector tail policy. |
| [Quant](./contracts/quant.md) | Target-width clamp and requantization formulas where `I8`, `I16`, `U8`, and `I32` are semantic operand/result widths. |
| [NN](./contracts/nn.md) | Deterministic activation, dot, RMS norm, and RoPE row primitives. |
| [Prob](./contracts/prob.md) | Logit/probability max, softmax, logsumexp, logsoftmax, softplus, logsigmoid, and cross-entropy primitives. |
| [RL](./contracts/rl.md) | Bellman, TD error, Q update, return, and GAE recurrences. |

## Verification

`test_math32` is created only by the math32 contract route. The default
development configure does not expose the target or register its CTest row.
The commands below are an owner-local subsystem development loop only; they do
not provide Release, source-stability, installed-package, or artifact evidence.
Use `tools/release/run` for those repository-wide claims.

```sh
cmake -S . -B .cache/build/math32-contract-route \
  -DRUND_ENABLE_MATH32_CONTRACT_TESTS=ON
cmake --build .cache/build/math32-contract-route --target test_math32
ctest --test-dir .cache/build/math32-contract-route \
  -R '^math32\.contract$' --output-on-failure --no-tests=error
```

## SDK Artifact

Release packaging for this subsystem is owned by
[`/package`](../../package/README.md). This subsystem owns behavior and tests;
`/package` owns artifact layout, CMake package export, and external
consumption policy.
