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

Telemetry durations and byte counts, the pure-evidence Replay semantic totals
`Spawned`, `Completed`, `Yields`, `Joins`, `Timers`, `ChannelSends`,
`ChannelRecvs`, `ChannelCloses`, `Observations`, and `ObservationDropped`,
Compute memory evidence, accelerator execution evidence, every additive Network
call/lifecycle/would-block/admission-rejection/byte slot, and repository
performance-measurement aggregation consume this owner directly.
The reactor uses it for every additive `StatStorage` mutation; maximum gauges
and non-telemetry identity sequences remain separate operations.
`Failed` is not yet in that Replay list: scheduler progress currently also
uses its value change as a host-replay activity signal. Saturating that slot
before separating the control responsibility could hide activity at
`UINT64_MAX`, so its raw producers remain an explicit contract blocker rather
than a claimed implementation of this law.
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

`runtime.task.net-ready-set` verifies the reactor consumer at the exact
`UINT64_MAX - 1` boundary, proves the reserved compatibility slots remain
unchanged, and proves both public alias getters read their canonical slots.
`runtime.task.net-stats` drives the production host-event recorder through all
15 Network event kinds at the same boundary, checks every non-selected call
slot and the opposite byte slot remain unchanged, and covers the existing
would-block and failed-event precedence. `runtime.task.net-limits` covers each
externally reachable capacity-rejection route. A source scan separately proves
that both admission-rejection mutation sites consume `Accumulate`; the
null-owner defensive branch is not claimed as dynamically exercised by that
target.
