# Checked Arithmetic Contract

`kernel/core/checked.hpp` is the only implementation authority for exact
unsigned 64-bit capacity arithmetic shared by Kernel and Node. It is an
internal machine-bound safety contract, not a public numeric law.

For `M = 2^64 - 1`:

- `checked::add(a, b)` succeeds exactly when `a <= M - b`.
- `checked::sub(a, b, out)` succeeds exactly when `a >= b`.
- `checked::mul(a, b)` succeeds exactly when `b == 0` or `a <= M / b`.
- `checked::mul(a, b, c, out)` first proves `a * b`, then proves the product
  times `c`.
- `checked::ceil(n, d)` is `floor(n / d) + [n mod d != 0]` for `d != 0` and
  returns the fail-closed count `0` for `d == 0`.

The result overloads publish `out` only after the complete operation succeeds.
Failure preserves the caller's previous output, including failure of the
second multiplication in a three-factor product. An overflow query is the
negation of the corresponding predicate; there is no separately maintained
overflow implementation.

The ceiling expression never evaluates `n + d - 1`, so it cannot overflow.
The add and multiply predicates evaluate one subtraction or division before
the represented operation. Every function is header-only `constexpr`,
stateless, allocation-free, and independent of backend or workload size.

Typed host-size projection, alignment, saturating counters, fixed-point
arithmetic, and strict reduction are different contracts and do not reuse
these exact operations.

## Verification

`kernel.core` owns the algebra and boundary cases in
`kernel/tests/contract/core/checked.cpp`. Replay, Compute, and accelerator
tests verify their own admission and persistence semantics without repeating
the arithmetic law.
