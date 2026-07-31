# Math32 Prob Contract

Math32 probability law owns deterministic fixed-logit rows and vector helpers:
`Max`, `SoftmaxApprox`, `LogSumExpApprox`, `LogSoftmaxApprox`,
`SoftplusApprox`, `LogSigmoidApprox`, and `CrossEntropyLogitsApprox`.
`prob/model.hpp` is the one public owner for probability constants, row status,
and result records.

The canonical production law is the SIMD lane law for this product width.
Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Row APIs consume all input elements. Full chunks use `simd::LaneCount` loads;
non-multiple tails are copied into one padded vector carrier, computed with the
same vector law, stored only for live lanes, and counted in `processed`. Tail
computation must not fall back to a separate scalar formula.
Row alias admission consumes the single private byte-range law owned by the
[SoA contract](./soa.md); Probability owns no parallel overlap predicate.

Evidence: `math32.contract`.
