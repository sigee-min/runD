# Residency Plan

This page owns immutable demand storage, capacity derivation, dirty geometry,
and identity. It owns no mutable frame state or backend transition.

`ResidencyPlan` has two compressed representations under one policy. A
monotonic stream uses constant storage. A recurrent graph stores
`O(resources + stages + remap entries)` cold topology and projects one
`K`-bounded epoch into
caller-owned `PageUse` scratch. Neither representation simulates a frame cache
or stores runtime transitions.

## Source ownership

The backend-neutral execution `Plan` is split by immutable concern. `plan/seal.cpp`
owns checked request/materialization validation, canonical-window admission,
stable identity hashing, and sealing. `plan/project.cpp` owns node, forecast,
and predecessor projections. `plan/input.cpp` owns input byte/source/reuse and
live-row/materializability projections. `plan/window.cpp` owns canonical input
sources and exact Window footprint reconstruction. The shared
`plan/internal.hpp` contains only the single checked-add and checked-multiply
constexpr helpers; no leaf stores mutable state or forwards through an
aggregate implementation.

## Graph preparation ownership

Graph planning borrows the immutable request only for the duration of
`PlanResidency`. A bounded array of resource pointers supplies canonical ID
order without copying the input graph, stage ports, or remap vectors. The
sealed resources own one sorted copy of each remap; no caller storage is
retained in the returned plan. Liveness and coloring use invocation-local
arrays bounded by `TiledGraphResourceCapacity`; coloring reads the visited
prefix of its own order instead of allocating a second index list. Lookup comparators and active-invocation
validation borrow resource rows by const reference. Repeated resource lookup
and active projection allocate no memory, including nonempty remap templates.
The graph planner contract verifies caller-order preservation, canonical
identity, independent result ownership, and allocation-free warm projection.

## Epoch projection

A sealed remap is a complete bijection sorted by `target_local`, so epoch
projection indexes the target row directly and checks its target/source bounds.
It does not scan the remap vector once per page. Pin bounds, next-use ordinal,
and prefetch epoch depend only on the port and batch; they are checked and
computed once before that port's page loop. Only source-page selection, payload
bounds, and dirty-tail bytes vary per page. Thus `K` pages over `P` ports use
`O(P*K)` projection work, including remapped inputs. Full-page and active-tail
permutation contracts cover both Begin and End origins with exact PageUse
fields and no allocations.

## Stream

For page count `P`, requested/max frames, frame-local dirty extent, total dirty
bytes `B`, and prefetch distance `D`, preparation selects `K` once:

```text
dirty.offset = input_page_bytes + output_prefix_bytes
dirty.bytes = output_payload_bytes
K = min(P, requested_frames)
W = P == 0 ? 0 : ceil(P / K)
```

`P>0` with no requested frame and a request above `max_frames` are rejected.
Epoch `w<W` is projected by formula:

```text
first(w) = checked_mul(w, K)
count(w) = min(K, P - first(w))
pin(w) = [w,w]
ready(w) = w
prefetch(w) = max(0, w - D)
first_use(p) = p
next_use(p) = checked_add(P, p), overflow -> NeverUse
dirty(p).offset = dirty.offset
dirty(p).bytes = min(dirty.bytes, B - checked_mul(p, dirty.bytes))
```

For Backing output, `B` is the exact logical output extent. Reduction uses the
checked extent of fixed-width Transient partials; those partials never become
backing ranges. Runtime projects only `dirty(p).bytes` onto
`p*output_payload_bytes`; the frame-local prefix is not added to backing.

## Identity versions and remap templates

The no-remap graph schema is identity version 9 and remains byte-equivalent to
the ordinal plan. A nonempty external-input page template selects version 10
and hashes the Program graph fingerprint plus the canonical sorted
resource groups. Each group is hashed as the exact number sequence
`resource, remap_count, target_local, source_local, origin` for every entry.
The public `input` field is only a dense input ordinal; preparation resolves it
once through the compiled input-resource table and seals the resulting private
canonical resource ID. The public ordinal is not an identity tuple component.
The
template is deep-copied into fixed-capacity plan/Authority storage; a request
that retains the caller span cannot be admitted.

Only a full resource-level bijection is supported. For `K <= 32`, the active
tail uses `q = min(K, P - F)` and resolves source pages within that batch. The
planner seals bounds and uniqueness, and the Authority repeats the full-q
check before one token commit. Partial-byte remaps and cross-resource or
internal/output remaps are not represented by this plan.

## Graph

Every graph use freezes exact access, dirty extent, least later use,
single-epoch pin, ready epoch, and `max(0,e-D)` read prefetch. Duplicate access
is merged to read-write. Touching or overlapping dirty extents are merged;
fragmented same-page output that cannot be one exact contiguous extent is
rejected rather than widened. The recurrent consumer is owned by
[Graph](./graph.md).

## Identity

Checked arithmetic rejects page, epoch, counter, and byte overflow. The
completed plan has no mutation API or problem-size-dependent stream storage.

Identity domain `rund.compute.pipeline.residency` uses graph schema 9 for the
ordinal plan and schema 10 for a nonempty external-input remap, and covers:

- stream: `stream,page_bytes,P,K,dirty,B,D`;
- graph schema 9: `graph,page_bytes,K,D` plus canonical
  `(node,tile,PageKey,access,dirty,next_use,pin,prefetch,ready)` demand.
- graph schema 10 additionally includes the Program fingerprint and the
  canonical resource groups, each with the byte tuple
  `resource, remap_count, target_local, source_local, origin`.

`CacheKey::domain` distinguishes Backing and Transient identities inside the
one Authority. Execution counters are never identity inputs. Changing demand,
future timing, dirty extent, page width, or capacity changes identity.
