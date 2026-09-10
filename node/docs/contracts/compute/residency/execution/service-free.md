# Service-Free Direct Recurrence

This page owns the stronger product seam that may connect an Authority-proved
resident Direct recurrence to a fused Metal or Vulkan Map primitive. It is
not the service-aware persistent Sliding request: a service-free request has
no ready, wait, acknowledgement, Fetch, Promote, Drain, Persist, Project,
Release, or Returned operation.

Status: the backend-neutral capability, immutable owned proof, request,
aggregate Final, validation, and Q=5/9/257 contracts are implemented.
Common expanded Pipeline preparation mints one read-only proof from the same
canonical top-level `MapRecurrence`. The ordinary public
`Pipeline::repeat<Q>::run()` product route now registers the exact resident
views with Authority, admits one aggregate lease, submits the fused backend
primitive once, consumes one aggregate Final, and commits the Authority rows
and Pipeline terminal once. Actual Metal and Vulkan Q=5/9/257 tests exercise
that public route and require exact output, one native/queue submit, one
payload dispatch, zero per-iteration native submits, zero Host service turns
or epoch callbacks, and one Final/publication. They also cover Known
pre-submit rejection followed by same-Pipeline retry and sticky Unknown
quarantine without publication.

Each backend diagnostic publishes an explicit native Terminal/History mode.
Capability is fail-closed unless that mode equals the proof retention; an
unknown or unsupported mode cannot select the product route.

This implemented product route is the resident, service-free Direct shape for
both terminal output and `write_each` history. Fixed common repeat storage is
part of its authenticated proof and capability. History requires each output
reference to cover exactly Q authored slices, with a backend-checked per-slice
pitch and exact logical output bytes; the native lowering writes one slice per
GPU iteration. Large-Q evidence beyond 257 and replacing a backing-serviced
Virtual page recurrence remain open. The last case is a different semantic
recurrence and must not select this route merely because its page count equals
Q.

## Semantic boundary

The recurrence coordinate is an iteration of one resident state:

```text
x[k + 1] = F(x[k], c), 0 <= k < Q
```

It is not a sequence of unrelated Virtual backing pages. A Virtual plan with
page inputs `[1]`, `[10]` and `F(x)=x+1` must produce `[2]`, `[11]`; replacing
it with a two-iteration recurrence would produce `3` from the first page and
never read the second. Therefore the existing page-epoch Sliding route cannot
select this seam merely because its page count equals Q.

An admitted proof owns, either directly or through one exact strong RunState
owner:

- the strong semantic owner;
- the exact canonical Map artifact and Compute plan;
- normalized resident input/output references and strong handles;
- the complete fixed set of distinct physical resident allocations touched by
  any occurrence, with the union of read/write access;
- exact dispatch windows and parameter bytes;
- the recurrence iteration count and terminal/history publication mode; and
- one full 128-bit proof identity.

The current projection copies normalized resident rows and their storage
handles. Artifact, window, and parameter views remain immutable views into the
strongly retained RunState. This avoids an unplanned second copy of canonical
source/IR and keeps proof storage fixed in Q. Cold preparation scans the
authored occurrences, but after the backend has consumed them the common owner
compacts the repeated semantic-owner table to one row. The retained proof,
status, identity, and control layouts are fixed-capacity, so the proof and
backend capability both authenticate `fixed_common_storage=true`. Cold
Q-dependent preparation work remains transient and is not hidden by that
retained-storage fact.

The physical-state projection scans every cold occurrence but retains only a
fixed-capacity distinct allocation set. Each row is canonicalized to the full
allocation `(id, bytes, offset=0, element=1, stride=1, count=bytes)` while the
semantic input/output arrays preserve their typed subviews. Repeated ids must
have the same allocation bytes and strong handle. A recurrence that introduces
Q-distinct allocations, aliases one handle under different ids, or exceeds the
fixed state capacity is rejected; it cannot manufacture a fixed-storage claim
from a fused command stream.

Hash equality is diagnostic only. Authority binds the exact proof owner and
registered resident regions; it does not reconstruct program semantics from a
hash or backend object.

## Resident Authority

An ordinary resident Buffer view is registered with its complete resource id,
allocation bytes, offset, element width, stride, count, access, role, and one
process-unique registration nonce. Authority retains every geometry field in
its frame row and rechecks the opaque nonce plus the exact fields during
admission; no 64-bit geometry hash is an authority boundary.

The aggregate lease pins all registered input/output rows and strongly retains
the registration owner. Read-only, write-only, and loop-carried Read|Write rows
are respectively registered as Input, Output, and Intermediate. Known success
makes every written row Resident and invokes one no-throw publication callback.
Known pre-submit failure restores the rows for retry; a Known may-write failure
invalidates every written row before retry.
UnknownMayWrite invokes no publication callback and retains the proof plus
pinned rows as sticky quarantine. The focused Authority contract covers
Q=5/9/257, cross-Authority replay, forged geometry, callback count, Known
retry, Unknown quarantine, and the fixed registration-owner join. It does not
substitute for the actual public Metal and Vulkan product tests.

## Fixed-storage capability

Product selection requires all of:

```text
device_generated_recurrence = true
fixed_native_storage        = true
fixed_common_storage        = true
one_native_submit           = true
host_service_turns_zero     = true
host_epoch_callbacks_zero   = true
aggregate_terminal_once     = true
terminal_output             = true   # terminal retention
history_output              = true   # exact Q-slice history retention
```

`fixed_native_storage` covers backend commands and native recurrence state.
`fixed_common_storage` independently proves that retained common semantic-owner,
identity, profile, and control storage does not grow with Q. Product admission
requires both facts and rejects a backend capability that disagrees with the
immutable proof. Actual Q=5/9/257 contracts compare the complete current
common Host/Device/Staging memory tuple across Q, in addition to the backend
native and route-storage diagnostics. This remains an orthogonal storage fact;
it is not permission to fabricate Host epoch control, total user-visible
History output storage, or total scratch/storage. Those total-storage and
large-Q products remain unimplemented.

## Product transaction

The request contains one immutable proof, one Authority credential, one
backend lowering owner, and one Final callback. Its only backend operation is
`submit`; the type has no per-iteration callback or data-service table. Final
reports Q completed iterations, one native submit, zero epoch submits, zero
Host service turns/callbacks, and one aggregate terminal.

The registration State seals the wrapper's shared-owner control block and
object pointer as a weak credential. Authority requires the same shared
control block and the same wrapper pointer; an aliasing subobject or a
different control block is foreign even when its raw pointer compares equal.
It checks this credential before any pin or lifecycle transition. With no
active direct execution, DirectAbort is `Invalid`; for an active slot, foreign,
stale, wrong-phase, Pending, and Inflight direct aborts are `Busy` with no
mutation, publication, or quarantine. Only an authenticated Known close
clears the slot, while Unknown or contradictory evidence remains
Quarantined.

Successful publication requires mutation-free preparation across Authority,
Pipeline, and all output/history regions followed by one no-fail commit.
Known pre-submit rejection publishes nothing and permits retry. Unknown
MayWrite retains the proof, backend lowering, registered resident regions,
and Pipeline attempt under sticky quarantine.

The resident close is split into an Authority stage and wrapper-owned finish.
Stage restores or invalidates exact registered rows and enters Pending without
calling publication. Service-free Release then authenticates the same
immutable registration credential in `release_pending`; only `finish_pending`
performs the single callback. Busy or foreign credentials do not mutate or
quarantine the owner. Callback re-entry is blocked by Inflight, and a
post-callback verification failure preserves the already-published terminal
without a second publication.

That boundary is also physical in the runtime implementation:
`registry/direct_recurrence_owner.{hpp,cpp}` is the stateless Authority facet
and accessor; `direct_recurrence/begin.cpp` owns admission and lease minting;
`direct_recurrence/accept/registration.cpp` owns credential-to-row binding,
construction rollback, and retirement; `direct_recurrence/accept/validation.cpp`
owns lease and registered-row identity checks; `direct_recurrence/accept/final.cpp`
owns staged release and the one publication transition; and
`direct_recurrence/abort.cpp` owns rollback-frame restoration and quarantine.
Authority retains the single gate, frame table, and ExecutionSlot. Its only
Direct-facing surface is `direct_recurrences()`; the private registration
rollback remains reachable only by `DirectRecurrenceRegistration` and
`DeviceVsmRegistration`. There is no aggregate accept translation unit or
second lifecycle authority.

The focused Authority proof mirrors those lifecycle boundaries physically.
`service_free_direct/authority.cpp` owns only the ordered contract dispatch;
`authority/support.cpp` owns the canonical proof fixture and stage/finish
publication bridge; `authority/success.cpp` owns successful and Known retry
transitions; and `authority/terminal.cpp` owns Frozen/Inflight exclusion,
forgery rejection, and Unknown quarantine. The shared `authority/local.hpp`
contains the one fixture/publication model plus declarations only, with no
second registration state or terminal policy.

If the stage caller receives Busy after Final freeze, it invokes the frozen
abort path with the same State pointer, nonce, proof, token, generation, and
owner. A Known authenticated abort closes the generation; Unknown or
contradictory evidence quarantines it. Construction failures use the separate
ordered exact-binding rollback and never call generic `release_frames`.

## Acceptance

The implemented terminal-output and History product boundary is accepted by
actual public Metal and Vulkan tests for Q=5/9/257. The tests prove:

1. one public handoff, one queue submission, one payload dispatch, one Final;
2. zero per-iteration native submits, callbacks, and Host service turns;
3. fixed native storage and fixed retained common storage across Q=5/9/257;
4. exact terminal output or exact Q history slices and one Pipeline generation
   advance;
5. Known pre-submit retry and Unknown sticky quarantine; and
6. one no-throw Authority/Pipeline terminal publication.

The same boundary remains incomplete for a large-Q product run and arbitrary
backing-serviced Virtual VSM recurrence. The aggregate algebra is shared by
terminal and History retention; it is not a claim of GPU-generated fixed
storage for either Virtual page recurrence.
