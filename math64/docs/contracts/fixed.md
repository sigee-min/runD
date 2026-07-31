# Math64 Fixed Contract

Math64's raw SIMD lane law uses a signed 64-bit lane with 63 fractional bits.
It does not declare a public fixed value type; the
public stored-value authority is `rund::compute::Fixed<I, F>`. Math64 formulas use
`math64::simd::I64x`, `math64::simd::U64x`, and `math64::simd::Mask64x`
carriers. Scalar element widths such as `i64` and `u64` are lane storage
types, constants, or cold detail helpers, not public fixed formula contracts.
`i128` and `u128` remain private implementation details under
`rund::math64::detail`. Because this product is header-only, detail helpers can
be transitively visible to C++ name lookup; they are still outside the public
formula surface and outside the supported API contract.
Private widened vector helpers reinterpret signed and unsigned 128-bit carriers
with bit-preserving casts. They do not rely on value-converting vector casts
between signed and unsigned 128-bit element types, because GCC and Clang do not
share one portable value-cast contract for those private carriers.
The two widened lanes are stored as one ordered pair of 128-bit values.
Every operation applies the same law to both values, then narrows in lane
order. This preserves the two-lane result bits without requiring a 256-bit
platform ABI, target flag, runtime dispatch, or second formula authority.

`math64/fixed/arithmetic.hpp` is the sole public grouping header for scalar and
lane fixed arithmetic. Public vector formulas are unsuffixed inside
`rund::math64`: `AddWrap`, `AddWrapUnsigned`, `SubWrap`, `MulLow`, `MulHigh`,
`AddSat`, `AddSatUnsigned`, `SubSat`, `Min`, `Max`, `Clamp`, `Select`,
`NegPositiveFixed`, `Abs`, `AbsMagnitude`, `Sign`, `MulFixed`,
`MulFixedScaled`, `MulUnsignedFixed`, `MulAddFixed`, `DivFixed`, `Recip`,
`Sqrt`, and `Rsqrt`.

The canonical production law is the SIMD lane law for this product width.
Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Evidence: `math64.contract`.
