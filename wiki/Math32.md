# Math32 API

Entry header: `<math32/math32.hpp>`

Namespace: `rund::math32`

[Back to API Reference](./API.md)

## When To Use

Use this header for deterministic 32-bit raw-lane and vector math. Its fixed
formulas operate on signed Q1.31 SIMD lane bits; the module does not declare a
public fixed value type. Public stored fixed values use
`rund::compute::Fixed<I, F>`. Math32 also provides helper families for
geometry, prepared SoA ranges, quantized rows, probabilities, neural-network
primitives, and reinforcement-learning recurrences.

For 64-bit high-precision math, see [Math64](./Math64.md).

## Major Types

| Family | Public Names |
| --- | --- |
| Scalar aliases | `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64` |
| Fixed scale | `FixedMin`, `FixedMax`, `FixedScale`, `FixedHalf`, `FixedQuarter`, `SinCos`, `SinCosx` |
| SIMD lanes | `simd::LaneCount`, `simd::I32x`, `simd::U32x`, `simd::Mask32x`, `simd::SplatI32`, `simd::SplatU32` |
| Geometry | `geom::Vec2x`, `geom::Vec3x`, `geom::Mask2x`, `geom::Mask3x`, and view types |
| SoA ranges | `soa::I32View`, `soa::I32MutView`, `soa::Status`, `soa::StatusReason` |
| Quant/NN/probability results | `quant::RequantResult`, `nn::DotAccumulator`, `nn::RowStatus`, `prob::MaxResult`, `prob::RowStatus`, `prob::LogSumExpResult`, `prob::CrossEntropyResult` |

## Function Families

| Family | Use For |
| --- | --- |
| SIMD | Lane load/store, masks, selection, comparisons, arithmetic, and reductions. |
| Raw fixed lane | Q1.31 lane wrap, saturating arithmetic, fixed multiply/divide, reciprocal, square root, clamp, min, max, abs, and sign. |
| Nonlinear | Vector `Exp2` and `Log2` approximations. |
| Turn | Turn-space constants, sine/cosine, tangent ratio, atan, and atan2 helpers. |
| Stat | Mean, variance, RMS, interpolation, smoothstep, and Hermite helpers. |
| Geom | Componentwise 2D/3D vector load, store, add, subtract, clamp, comparisons, and masks. |
| SoA | Prepared component-array validation and range operations over geometry views. |
| Quant | Clamp and requantize lanes to `i8`, `i16`, or `u8` target ranges. |
| NN | Deterministic activation, dot product, RMS norm, and RoPE row helpers. |
| Prob | Row max, softmax, logsumexp, logsoftmax, softplus, logsigmoid, and cross-entropy helpers. |
| RL | Bellman, TD error, Q update, return, and GAE recurrence helpers. |

## Result Rules

- Scalar and SIMD pure value functions return their values directly.
- Status and result helper families, including SoA, probability, and
  quantization helpers, expose `ok()` or status fields when validation matters.
- Geometry view and SoA status success derives from their reason enum;
  `NotEvaluated` is the default and no parallel success bit exists.
- Check the status or result before using output buffers written by those
  helpers.

## Example

```cpp compile
#include <math32/math32.hpp>

bool add_fixed_lanes() {
  namespace m = rund::math32;

  const m::simd::I32x lhs = m::simd::SplatI32(m::FixedHalf);
  const m::simd::I32x rhs = m::simd::SplatI32(m::FixedQuarter);
  const m::simd::I32x sum = m::AddSat(lhs, rhs);
  const m::simd::Mask32x in_range =
      m::simd::Le(sum, m::simd::SplatI32(m::FixedMax));

  return m::simd::All(in_range);
}
```
