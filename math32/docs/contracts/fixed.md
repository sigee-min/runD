# Math32 Fixed Contract

Math32's raw SIMD lane law uses a signed 32-bit lane with 31 fractional bits.
It does not declare a public fixed value type; the
public stored-value authority is `rund::compute::Fixed<I, F>`. Math32 formulas use
`math32::simd::I32x`, `math32::simd::U32x`, and `math32::simd::Mask32x`
carriers. Scalar element widths such as `i32` and `u32` are lane storage
types, constants, or cold detail helpers, not public fixed formula contracts.
`i64`, `u64`, `i128`, and `u128` are private implementation details under
`rund::math32::detail`. Because this product is header-only, detail helpers can
be transitively visible to C++ name lookup; they are still outside the public
formula surface and outside the supported API contract.

Four-lane widened intermediates use one ordered pair of 128-bit carriers.
Lane `0..1` belongs to the low carrier and lane `2..3` to the high carrier.
Every operation applies the same law to both halves, then narrows in that
order. This preserves the four-lane result bits without requiring a 256-bit
platform ABI, target flag, runtime dispatch, or scalar formula authority.

`math32/fixed/arithmetic.hpp` is the sole public grouping header for scalar and
lane fixed arithmetic. Public vector formulas are unsuffixed inside
`rund::math32`: `AddWrap`, `AddWrapUnsigned`, `SubWrap`, `MulLow`, `MulHigh`,
`AddSat`, `AddSatUnsigned`, `SubSat`, `Min`, `Max`, `Clamp`, `Select`,
`NegPositiveFixed`, `Abs`, `AbsMagnitude`, `Sign`, `MulFixed`,
`MulFixedScaled`, `MulUnsignedFixed`, `MulAddFixed`, `DivFixed`, `Recip`,
`Sqrt`, and `Rsqrt`.

The canonical production law is the SIMD lane law for this product width.
Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Evidence: `math32.contract`.
