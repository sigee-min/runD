# Math64 Stat Contract

Math64 stat law owns high-precision `Mean`, `Variance`, `Rms`,
`DifferenceMagnitude`, `MulUnit`, `Lerp`, `ClampLerp`, `SmoothStep`, and
`Hermite`.

The canonical production law is the SIMD lane law for this product width.
Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Range APIs consume all input elements. Full chunks use `simd::LaneCount`
loads; non-multiple tails are copied into one padded vector carrier, computed
with the same vector law, and counted in `processed`. Tail computation must not
fall back to a separate scalar formula.

Evidence: `math64.contract`.
