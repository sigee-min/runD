# Math32 NN Contract

Math32 NN owns deterministic vector activations, semantic-width dot products,
RMS normalization, and RoPE helpers. `DotI8I8To32`, `DotU8I8To32`, and
`DotI16I16To32` keep widths because those widths name operands and result
accumulators.
`nn/model.hpp` is the one public owner for NN row status and RoPE pair results.

The canonical production law is the SIMD lane law for this product width.
Element scalar values are storage and test literals only; they are not public
formula contracts or fallback modes.

Row APIs consume all input elements. Full chunks use `simd::LaneCount` loads;
non-multiple tails are copied into one padded vector carrier, computed with the
same vector law, stored only for live lanes, and counted in `processed`. Tail
computation must not fall back to a separate scalar formula.
Row alias admission consumes the single private byte-range law owned by the
[SoA contract](./soa.md); NN owns no parallel overlap predicate.

Evidence: `math32.contract`.
