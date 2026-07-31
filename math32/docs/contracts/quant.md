# Math32 Quant Contract

Math32 quant owns target-width clamp and requantization formulas:
`ClampI8`, `ClampI16`, `ClampU8`, `RequantI32ToI8`, `RequantI32ToI16`, and
`RequantI32ToU8`. Widths in these names are semantic operand or result widths,
not namespace-width suffixes.

Public math formulas are vector-first. The canonical production law is the
SIMD lane law for this product width. Element scalar values are storage and
test literals only; they are not public formula contracts or fallback modes.
For a clamped shift `s`, requantization defines the rounding exponent as
`k = s == 0 ? 0 : s - 1` and the rounding term as
`r = s == 0 ? 0 : 2^k`. The zero-shift lane therefore never evaluates an
out-of-range unsigned shift.

Evidence: `math32.contract`.
