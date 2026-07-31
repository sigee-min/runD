# Math64 Contracts

Contracts are split by formula family and linked from
[`math64/docs/README.md`](../README.md). Width is carried by the top-level
`rund::math64` namespace; public formula names do not keep a namespace-width
`64` suffix.

Public math formulas are vector-first. The canonical production law is the
SIMD lane law for this product width. Element scalar values are storage and
test literals only; they are not public formula contracts or fallback modes.
Every public formula must use the vector-first carrier contract and is verified
by `math64.contract`. Focused compiler-output inspection may support a specific
performance investigation, but it is not a second formula registry or a
permanent acceptance gate.
