# Storage Budget Contract

## Scope

This page owns the domain-neutral hierarchical storage accounting surface under
`rund::storage`. It defines concurrent capacity admission, reservation
lifetime, exact committed usage, refund, and reporting. Replay specializes
this mechanism for spill generations in [Replay](./replay.md) and
[Host](./host.md); those pages own Replay byte formulas and filesystem
lifetime, not a second budget state machine.

Public authority is `<rund/storage.hpp>`. Implementation authority is
`/node/src/runtime/storage.cpp`, and focused contract verification is
`/node/tests/contract/runtime/storage.cpp`. SDK consumers link the same
`runD::sdk` target as every other focused runD domain.

A Budget is an accounting capability, not a filesystem, allocator, artifact
store, or volume-global quota. The subsystem that commits a Reservation owns
the measurement represented by `Usage`. The caller that creates the root
Budget owns which producers share that aggregate capacity boundary.

## Public Surface

The complete public vocabulary is:

- `Usage{physical_bytes, allocated_bytes}`;
- `Report`;
- `Status`;
- copyable `Budget`;
- move-only `Reservation`.

`Budget(capacity_bytes)` creates one root whose capacity is measured in
conservative allocated bytes. `child(capacity_bytes)` creates a node in the
same hierarchy, `reserve(max_allocated_bytes)` attempts one capacity
reservation, and `report()` returns one synchronized snapshot.
`Reservation::commit(usage)` replaces the maximum reservation with measured
committed usage. `refund()` releases either an uncommitted reservation or a
committed usage. Destruction performs the same release when an owner has not
already refunded it.

Every fallible value has one `ReasonCode`. `Ok` is the sole success value;
truth conversion and `ok()` derive from it, `error()` is its stable text
projection, and `exit_code()` is `0` or `1`. A default Budget, a zero-capacity
root or child, and a default or moved-from Reservation fail closed.
`reserve(0)` is valid: a producer may need to account exact physical growth
that consumes no additional allocated capacity. The storage-specific failure
vocabulary is:

- `StorageBudgetInvalid`;
- `StorageBudgetAllocationFailed`;
- `StorageCapacityExceeded`;
- `StorageReservationInvalid`;
- `StorageCommitInvalid`.

## Hierarchy And Concurrency

Every node stores its own capacity and current counters. A child capacity must
be nonzero and no greater than its direct parent's capacity. Child creation
does not permanently partition or reserve parent capacity: siblings compete
for the actual reservations and committed allocations charged through their
common ancestors.

Let, for any hierarchy node `v`,

```text
C_v = capacity_bytes
A_v = allocated_bytes
R_v = reserved_bytes
U_v = A_v + R_v
```

The invariant is:

```text
0 <= U_v <= C_v
available_bytes = C_v - U_v
```

A request for `X` bytes through node `n` succeeds only when every node from
`n` through the root satisfies:

```text
X <= C_v - A_v - R_v
```

One mutex is shared by the complete hierarchy. Checking every ancestor and
charging them all is one critical section, so concurrent sibling Sessions
cannot each observe the same remaining parent capacity. Reservation, commit,
refund, counter updates, and `report()` use that same gate. This is a
thread-safe in-process hierarchy; it is not an interprocess filesystem lock or
a replacement for an operating-system quota.

For a Budget at hierarchy depth `D`, an accepted reserve or commit performs
exactly one read-only admission walk and one state-change walk. The
state-change walk updates current bytes, peaks, and the operation count
together; those fields are not maintained by a second counter traversal.
Refund is the exact inverse of an already admitted operation and therefore
performs one state-change walk. The ancestor-link visit bounds are consequently
`2D` for successful reserve and commit and `D` for refund, independent of byte
quantity. Rejection retains the same all-ancestor diagnostic count without
mutating byte or peak state.

Budget copies share the exact same node. A child retains its ancestor state, so
destroying a caller's Budget value does not invalidate live child Budgets or
Reservations.

## Reservation State

An accepted reservation for maximum allocated size `X` adds `X` to
`reserved_bytes` at its node and every ancestor. The Reservation is the sole
move-only lifetime owner of that charge.

Commit accepts:

```text
P = Usage.physical_bytes
A = Usage.allocated_bytes
```

only when:

```text
A <= X
```

Successful commit performs, at the Reservation node and every ancestor:

```text
reserved_bytes  := reserved_bytes - X
allocated_bytes := allocated_bytes + A
physical_bytes  := physical_bytes + P
```

The unconsumed upper-bound slack `X - A` therefore becomes immediately
available. Commit is one-way for that Reservation: a repeated commit fails
closed. `usage()` exposes the committed values only after success, and
`committed()` derives from the same state.

`P` and `A` are independent producer measurements. Capacity constrains `A`,
not `P`; a sparse-file producer can legitimately report `P > A`, while
allocation rounding can make another producer report `A > P`. Physical-byte
addition is still checked for overflow, but the generic Budget imposes no
ordering relation between the two fields.

Refund or destruction is exactly-once release:

```text
uncommitted: reserved_bytes  := reserved_bytes - X
committed:   allocated_bytes := allocated_bytes - A
             physical_bytes  := physical_bytes - P
```

After release the Reservation is invalid. Move construction and move
assignment transfer the one release obligation; the moved-from value cannot
commit or refund it again. Move assignment first releases any charge already
owned by the destination.

The implementation validates arithmetic before reservation and validates
`A <= X` plus checked physical and allocated addition before commit. It never
treats a rejection, failed commit, or diagnostic counter as a successful
allocation.

## Report Meaning

`Report` is one atomic snapshot of a Budget node:

| Field | Meaning |
| --- | --- |
| `capacity_bytes` | This node's configured hard accounting capacity. |
| `physical_bytes` | Current committed producer-measured physical bytes. |
| `allocated_bytes` | Current committed conservative capacity charge. |
| `reserved_bytes` | Maximum allocated bytes authorized but not yet committed. |
| `available_bytes` | Exact `capacity_bytes - allocated_bytes - reserved_bytes`. |
| `peak_physical_bytes` | High-water mark of current committed physical bytes. |
| `peak_allocated_bytes` | High-water mark of current committed allocated charge. |
| `peak_reserved_bytes` | High-water mark of outstanding reservations. |
| `peak_used_bytes` | High-water mark of `allocated_bytes + reserved_bytes`. |
| `reservation_count` | Accepted reservations at this node or its descendants. |
| `commit_count` | Successful commits at this node or its descendants. |
| `refund_count` | Explicit or destructor-driven releases at this node or its descendants. |
| `rejection_count` | Capacity or commit rejections at this node or its descendants. |

Counts saturate at `2^64 - 1`; current and peak byte fields remain exact
checked state. `physical_bytes` may exceed `capacity_bytes` because capacity
is denominated only in allocated bytes. Ancestor counts include descendant
operations because the ancestor is the aggregate authority. A child report
covers only that child subtree.

`physical_bytes` and `allocated_bytes` are deliberately distinct.
`physical_bytes` is the integrating producer's exact owned byte metric;
`allocated_bytes` is the conservative quantity admitted against capacity and
may include allocation rounding or another documented upper bound. The Budget neither probes a
filesystem nor verifies a producer's measurement.

## Responsibility Boundary

The root owner must include every in-process producer that is intended to share
one hard accounting limit. Producers outside that hierarchy are outside the
claim. In particular, the following require caller or platform authority:

- unrelated processes and writers;
- persisted artifacts written through caller callbacks;
- external or remote replicas;
- filesystem snapshots, copy-on-write retention, directory metadata, and
  journals;
- operating-system free-space or project-quota enforcement.

A filesystem free-space observation may be an additional admission condition,
but it is advisory because another writer can consume that space after the
observation. It does not strengthen the Budget into a volume-global capacity
reservation.

## Verification

Closure requires contracts that prove:

- root, child, invalid, allocation-failure, and result-code behavior;
- `A + R <= C` at every ancestor;
- concurrent sibling reservation never over-admits the shared root;
- exact reserve-to-commit conversion and immediate slack recovery;
- explicit, destructor, move-construction, and move-assignment refund exactly
  once;
- zero-size reserve/commit remains valid, including `physical > allocated`,
  while `Usage.allocated_bytes > Reservation::max_allocated_bytes()`,
  arithmetic overflow, and repeated commit fail closed;
- current, available, peak, and saturating operation-count reports.

Replay verification must additionally prove its producer-specific physical and
allocated measurements under the linked Replay and Host contracts.
