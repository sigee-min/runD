# Math32 Contracts

Contracts are split by formula family and linked from
[`math32/docs/README.md`](../README.md). Width is carried by the top-level
`rund::math32` namespace; public formula names do not keep a namespace-width
`32` suffix unless that width names an operand or result type such as
`DotI8I8To32`.

Public math formulas are vector-first. The canonical production law is the
SIMD lane law for this product width. Element scalar values are storage and
test literals only; they are not public formula contracts or fallback modes.
