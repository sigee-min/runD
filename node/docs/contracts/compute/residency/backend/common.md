# Accelerator Residency

Compute's private backend interface is composed once by `DeviceOps` in
`compute/backend.hpp`. Ordinary buffer, Job, and Pipeline operations remain
on that interface; `backend/residency.hpp` owns the native residency transport
and cold preparation callbacks as the value member `residency`.
`backend/virtual.hpp` owns the accelerator Virtual product and asynchronous
DeviceVSM callbacks as the value member `virtual_execution`.
Callers and test observers access the owning member directly. The outer
interface keeps no compatibility callbacks or forwarding copies, and the
accelerator catalog initializes each callback exactly once.
Transfer result and request values have their separate declaration owner in
`backend/transfer/model.hpp`; those values carry no transport policy. Pipeline
transfer declarations, transfer accounting, and Virtual cache declarations
include this model directly. Only implementations that invoke backend
callbacks require the complete operation interface.

`virtual/run/readiness.cpp` owns the mutation-free two-bank native-window
readiness query, including Pipeline-before-publication lock ordering and the
prepared-owner snapshot. Both staged DeviceVSM probing and persistent sliding
preparation consume it directly; neither carries a second readiness rule.
Only an unsupported result permits ordinary-route selection. Other backend
errors remain terminal with their original status, before native preparation.

Cold preparation proves the complete canonical resource set is
Pipeline-owned. Each private transfer or execution revalidates the retained
Buffer owner, resident identity, exact ref, and bank. Shared public transfers
keep ordinary Device claims and backend registry diagnostics; private routes
cannot fall back to unissued full-command execution.

Accelerator submission bypasses the Pool OS compute worker. One cold fixed
transaction retains selected locals and Pipeline lifetime until an exact
generation terminal. The callback validates GPU-written generation and
verified-prefix evidence before publishing a receipt. Stale or partial control
invalidates writable and fetched frames. CPU scheduling is unchanged.

The compute-private Q=1 boundary is separate from the selected epoch hook.
`prepare_residency_execution` is Plan-free: it aliases one already-accounted
prepared native owner and freezes only reusable topology. Compute's
`pipeline/execution/submit` binds the invocation `execution::Plan`, opaque
Authority token and generation, fixed selected-local storage, and one
native-only final evidence cell in caller-owned `Control`. It starts and
finishes the existing `PrivateResidency` Pipeline SOT around the backend's raw
`submit_residency_pipeline` evidence.

Arbitrary-Q uses a distinct whole-Schedule seam. Cold preparation receives
four immutable `(bank, publication parity)` role templates plus Q and the exact
tail. It returns a typed lowering, cold owner, tracked retained/transient byte
bounds, queue-call count, and asynchronous-callback proof. Compute reserves
and commits the exact retained charge before Authority begins. Submission then
adds only the run credential and final callbacks. Native Releases contain no
page, cache, backing, or Host-service policy; Compute projects every such fact
from the same sealed Plan and joins one aggregate terminal.

Native per-node callbacks are forbidden, and the backend cannot report Host
Input or Output service. Metal and Vulkan implement the raw exact-local Q=1
Dispatch boundary. The separate fixed window transport accepts up to four raw
batches in one public handoff. Direct pointwise Virtual execution consumes it
only when coherent Host views and fixed selected locals are available. For
Q>4 the preferred Schedule accepts all native batches up front; the fixed
window Stream remains capability/budget fallback. Neither adapter owns Compute
policy. CPU exposes no DeviceOps Schedule hooks and never constructs these
owners.

The finite `WindowRingFused` proof is a separate bounded DeviceVSM route for a
centered I32/U32 Window. It accepts bare forms and exact authenticated,
parameter-free `CanonicalTotalU32` Map chains of symmetric depth 1–3 before
and after the Window. It authenticates a Resident or fully pre-staged
input/output endpoint before Authority mutation, then uses one physical
dispatch/native submit for `Q` seed, `Q` compute, and `2Q` internal phases. It
has zero Host epoch control and zero transfer submissions/bytes, and one common
Final/publication/version. The proof is not a generic PagedLoop or I32 Map
fusion capability.
Common geometry derives the checked `2Q` internal-phase count once; Metal,
Vulkan, evidence validation, and terminal classification consume that value
without restating the multiplication.

A distinct fixed-state sliding transport owns at most four global-tagged
slots and advances one only from its exact asynchronous terminal. It accepts
immutable selected locals, active-local access masks, and authenticated
`(Plan, token, run generation, coordinate, stride, slot, turn, descriptor
generation)`. It contains no page, victim, next-use, or Graph formula and does
not reuse Q-proportional Schedule storage. Retained/transient bytes, actual
queue calls, and inflight peak remain explicit evidence.

Cold preparation also admits one State-private service owner with a single
coalesced ready cell. It is retained in the reported fixed byte charge and is
created before Submit; Wake performs neither allocation nor thread creation.
There is no global singleton lane, no Q-sized event queue, and no sharing that
can create cross-device head-of-line blocking. Exact submit-slot context and
Release callback handoffs remain live until their outer frames return.
Role identity and the complete backend capability are preflighted before that
State or service owner is allocated. A closed or malformed backend capability
therefore rejects with an empty Control and zero allocations; merely probing
the still-closed product path cannot create and tear down an OS thread.

The transport is not a backend capability by itself. A backend must prove the
complete descriptor row is GPU-visible after any required flush and before
selected work consumes it. Metal and Vulkan now have source-private physical
HostCoherent gates, but both report `descriptor_release_acquire=false`, so
production cannot admit them before the common Authority W2..4
Release/Final/publication boundary is proven. A CPU release/acquire mirror does
not satisfy this GPU memory-order law.
