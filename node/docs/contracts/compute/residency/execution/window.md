# Native Window

The backend-neutral native unit contains `1<=W<=4` epochs. Four is a fixed
storage bound, not a throughput claim. One raw handoff contains W native
batches; `queue_calls` independently records the actual backend queue shape.
Metal may report W queue calls while Vulkan reports one.

Each Release carries Plan identity, Authority token and generation, global
epoch, bank, monotonic backend sequence, status, and dispatch/completion/
may-write facts. Chunk storage is indexed by a local slot but authenticates
the stored global epoch. A later chunk cannot renumber epoch four to zero.

Authority consumes Release(e) before Output(e) or Input(e+2). Dispatch(e+2)
also waits for both service terminals. The Pipeline attempt must be published
and reseeded before a successful Release becomes reusable.

Input cache policy is global across the two fixed Host banks. Exact retained
pages may supply either Device bank. A Host miss may replace only an already
consumed page that is not demanded by the current node. If every Host row is a
future hit, an authenticated coherent mask directs backing bytes into the
exact Device target instead of evicting that future Host page; the ticket
records backing work and zero transfer work. Device targets remain bank-local,
so same-bank reuse still requires its prior Release. This policy preserves
warm Host hits while retaining zero physical H2D on an admitted coherent path.

The bounded controller has one serial Host-service lane. An input backing that
advertises more than one concurrent read therefore remains on the rolling
two-lane path; admitting it here would silently serialize the frozen prefetch
contract. A native window for parallel backing requires two authenticated
service lanes or a distinct transfer-queue DAG.

Duplicate, stale, wrong-bank, skipped-sequence, or out-of-prefix evidence is
invalid. Before native acceptance, an undispatched suffix is explicit known
no-dispatch. After acceptance, every batch is queued: failure opens remaining
no-write gates and reports dispatched, completed, `may_write=false`. It cannot
be relabeled undispatched. Terminal loss reports dispatched, incomplete,
may-write and quarantines the window.

The raw abort operation authenticates the same Plan/token/generation and opens
every unsignalled gate conservatively. Final evidence repeats the exact fixed
Release journal plus public handoff, native batch, queue-call, inflight peak,
and completion time. Final cannot unlock a bank and cannot fabricate Host
service completion.

Prepared command owners are copied into fixed strong storage. Signals are
authenticated before backend mutation and cross the backend callback boundary
without holding the common control mutex. A whole stream additionally holds a
deduplicated claim over both banks' primary and transactional alternate
prepared owners; intermediate chunk Finals cannot open a peer-submit gap.

Mutation-free readiness includes coherent Input/Output view availability and
pending view-fault disposition. A denied view or pending injected view fault
selects rolling before Authority/window mutation; rolling then consumes and
reports the physical fault. Readiness cannot erase the fault by treating it as
an ordinary capability miss after admission.

The Host-driven Virtual implementation keeps cold and warm ownership separate:
`virtual/run/execution/window/prepare.cpp` owns capability, prepared-owner, and
coherent-view admission before Authority mutation;
`window/service.cpp` owns Host Input/Output service plus native release/final
callbacks; and `window.cpp` owns bootstrap, the single final join, cache
publication, and statistics. The declarations-only local seam carries the one
borrowed `WindowWait` control shared by those phases. Neither side recomputes
the other's decision or retains a parallel execution authority.

## WindowRingFused

The finite WindowRingFused route is distinct from the Host-driven
fixed-`W` transport above. A typed immutable proof selects the Resident or
fully staged endpoint and authenticates the complete input/output footprint,
two scratch banks, range, and non-alias budget before the Authority lease. The
staged endpoint performs all page-in before that lease; no Host epoch service,
callback, or transfer submission runs during the recurrence. One physical
dispatch and one native submit own `Q` seed epochs, `Q` compute epochs, and
`2Q` internal phases. A single common Final then performs the output
writeback/publication/version transition. Known failure terminates accepted
work without fallback; Unknown quarantines the owner. The proof accepts bare
centered I32/U32 Window semantics, including checked signed-wrap and boundary
behavior, and exact authenticated, parameter-free `CanonicalTotalU32` Map
chains of symmetric depth 1–3 before and after the Window. It does not claim
I32 Map fusion or generic dynamic GPU↔Host demand/ack.

The DeviceVSM Window source recipes are direct-owned per backend:
`source/window/metal/` and `source/window/vulkan/` each keep `emission.cpp`
for scalar/direct/shared/counter fragments, `ring.cpp` for the two-bank
recurrence source, and `entry.cpp` for the ordinary public emitter. Existing
`metal_multipass.cpp` and `vulkan_multipass.cpp` remain the independent
multi-pass recipes. The root `source/window.cpp` only sequences validation,
identity, backend composition, and final artifact publication; `window/validation.cpp`
owns source-specific geometry/map admissibility, `window/identity.cpp` owns
the executable domain/API and artifact-key projection, and
`window/lifecycle.cpp` owns the checked artifact/plan/parameter/proof
materialization. `window/internal.hpp` is declarations-only for these owners
and source fragments. This split changes translation-unit ownership only;
artifact keys, source bytes, binding layouts, barriers, and status markers
remain canonical.
