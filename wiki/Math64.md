# Math64 API

Entry header: `<math64/math64.hpp>`

Namespace: `rund::math64`

[Back to API Reference](https://github.com/sigee-min/runD/wiki/API)

## When To Use

Use this header for deterministic 64-bit raw-lane and vector math. Its fixed
formulas operate on signed Q1.63 SIMD lane bits; the module does not declare a
public fixed value type. Public stored fixed values use
`rund::compute::Fixed<I, F>`. Math64 also owns nonlinear and turn-space
approximations, statistics, geometry, prepared SoA ranges, quantization,
neural-row, probability, and reinforcement-learning primitives.

For the corresponding 32-bit real-time and quantized row families, see
[Math32](https://github.com/sigee-min/runD/wiki/Math32).

## Major Types

| Family | Public Names |
| --- | --- |
| Scalar aliases | `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64` |
| Fixed scale | `FixedMin`, `FixedMax`, `FixedScale`, `FixedHalf`, `FixedQuarter`, `SinCos`, `SinCosx` |
| SIMD lanes | `simd::LaneCount`, `simd::I64x`, `simd::U64x`, `simd::Mask64x`, `simd::SplatI64`, `simd::SplatU64` |
| Geometry | `geom::Vec2x`, `geom::Vec3x`, `geom::Mask2x`, `geom::Mask3x`, and view types |
| SoA ranges | `soa::I64View`, `soa::I64MutView`, `soa::U64View`, `soa::U64MutView`, `soa::Status`, `soa::StatusReason` |
| Quantization | `quant::RequantResult` with lane values and exact shift-validity mask |
| Neural rows | `nn::RowStatus`, `nn::RopePairResult` |
| Probability rows | `prob::MaxResult`, `prob::RowStatus`, `prob::LogSumExpResult`, `prob::CrossEntropyResult` |

## Function Families

| Family | Use For |
| --- | --- |
| SIMD | Lane load/store, masks, selection, comparisons, arithmetic, and reductions. |
| Raw fixed lane | Q1.63 lane wrap, saturating arithmetic, fixed multiply/divide, reciprocal, square root, clamp, min, max, abs, and sign. |
| Nonlinear | Vector `Exp2` and `Log2` approximations. |
| Turn | Turn-space constants, sine/cosine, tangent ratio, atan, and atan2 helpers. |
| Stat | High-precision mean, variance, RMS, interpolation, smoothstep, and Hermite helpers. |
| Geom | Componentwise 2D/3D vector load, store, add, subtract, clamp, comparisons, and masks. |
| SoA | Prepared component-array validation and range operations over geometry views. |
| Quant | I8/I16/U8 clamping and widened I64 requantization with explicit shift validity. |
| NN | Deterministic activation, dot, RMS normalization, and RoPE row primitives. |
| Prob | Logit maximum, softmax, logsumexp, logsoftmax, softplus, logsigmoid, and cross entropy. |
| RL | Saturating Bellman, TD-error, Q-update, return, and GAE recurrences over Q1.63 lanes. |

## Result Rules

- Scalar and SIMD pure value functions return their values directly.
- Exposed status helper families, including SoA helpers, expose `ok()` or
  status fields when validation matters.
- Geometry view and SoA status success derives from their reason enum;
  `NotEvaluated` is the default and no parallel success bit exists.
- `quant::RequantResult::ok()` requires every shift lane to be valid.
  `nn::RowStatus` and the probability result types derive success from their
  recorded shape, overlap, empty-input, and target-validity fields; those
  fields remain the sole detailed outcome evidence.
- Check the status or result before using output buffers written by those
  helpers.

## Example

```cpp compile
#include <math64/math64.hpp>

bool multiply_fixed_lanes() {
  namespace m = rund::math64;

  const m::simd::I64x half = m::simd::SplatI64(m::FixedHalf);
  const m::simd::I64x product = m::MulFixed(half, half);
  const m::simd::Mask64x positive = m::simd::Gt(product, m::simd::SplatI64(0));

  return m::simd::All(positive);
}
```
