# Counter Arithmetic

This page owns Node's domain-neutral arithmetic for bounded evidence counters.
It does not own Compute numeric policy, fixed-point arithmetic, semantic hashes,
or additions whose contract is to reject overflow.

## Counter Laws

`rund/counter.hpp` is the single non-product implementation authority for
unsigned 64-bit evidence-counter mutation. For `M = 2^64 - 1`, addition is

```text
sat(a, b) = min(M, a + b)
```

where the sum on the right is mathematical rather than wrapping machine
addition. The implementation compares `b` with `M - a` before evaluating
`a + b`, so no overflowing machine operation is executed. The law is
commutative, associative, monotone, has zero as its identity, and has `M` as
an absorbing value.

Releasing retained capacity and measuring an interval preserve the same
absorbing maximum:

```text
remaining(current, released) =
  M                              when current = M
  max(current - released, 0)     otherwise

delta(before, after) =
  M                              when before = M or after = M
  max(after - before, 0)         otherwise
```

Once observation has saturated, later subtraction cannot reconstruct the
unknown mathematical value. `M` therefore remains absorbing for `Remaining`,
`Release`, and `Delta`; wrapping or resetting it to a plausible finite value
would manufacture false evidence. `Accumulate` and `Release` are the sole
in-place mutations. `SaturatingAdd`, `Remaining`, and `Delta` are their
value-returning laws. Saturating multiplication remains the shared owner for
derived counter products.

Telemetry durations and byte counts, Replay evidence totals, Compute memory
evidence, accelerator execution evidence, Network rejection accounting, and
repository performance-measurement aggregation consume this owner directly.
They do not retain domain-local aliases or copies of the overflow formula. The
transitive header and its `rund::detail` namespace are support code rather than
a direct SDK entry or product surface.
Checked additions that must reject overflow remain separate because rejection
and saturation have different result contracts.

## Verification

`counter.saturation` is a header-only exact contract. It verifies addition and
multiplication identity, absorption, exact overflow boundaries, commutativity,
and associativity. It also exhaustively checks `Remaining` and `Delta` over a
representative set containing zero, interior values, and both sides of the
maximum boundary, including the absorbing-maximum law. Run it with:

```text
tools/test/run counter.saturation
```
