# Execution Evidence

Implemented surfaces:

- immutable arbitrary-Q Plan projection and identity;
- O(1)-in-Q Authority admission and bounded circular journals;
- cache-aware Q1 Input/native/Output join with fixed tickets;
- native W1-W4 Release and Final transport;
- ordinary all-staged unary nonresident Pointwise Q>=2 selecting `StagedLoop`
  when exact mapped Host-visible/coherent input/output views are available;
  the route owns one native recurrence submit, Q GPU epochs, zero transfer
  submits/bytes, zero Host epoch submit/service/callbacks, and one
  Final/Authority/output publication/version;
- finite centered-I32/U32 `WindowRingFused` on Resident or fully staged
  endpoints, including bare forms and exact authenticated, parameter-free
  `CanonicalTotalU32` Map chains of symmetric depth 1–3 before and after the
  Window: one physical dispatch/native submit, Q seed epochs, Q compute
  epochs, 2Q internal phases, zero transfer submits/bytes and zero Host epoch
  submit/service/callbacks, followed by one common Final/publication/version;
  I32 Map fusion is not part of this evidence;
- clean pre-lease structural `BackendUnsupported` from that mapped probe
  declining to the legacy service, while any owner mutation/native acceptance
  or non-capability failure is terminal;
- legacy nonresident Persistent Pointwise R2 using two-coordinate
  `BackendChunked` chunks and `ceil(Q/2)` native submissions;
- dedicated bounded Window (`Q<=4`) or Stream (`Q>4`) fallback for unsupported
  Persistent spatial Window; old `OneSubmit` checks are lower-level evidence;
- Metal and Vulkan Direct pointwise coherent W2-W4 product execution;
- arbitrary-Q Direct pointwise coherent stream chaining with global epoch
  receipts, fixed chunk state, one caller handoff, and one caller Final wait;
- one O(1)-journal whole-schedule Authority terminal model, including
  Q=100,000 success and an arbitrary accepted Known/no-write failure suffix;
- production whole-schedule lowering for coherent Direct pointwise Q>4:
  Vulkan Q timeline records through one `vkQueueSubmit`, and Metal Q distinct
  pending MTL4 commands through Q queue operations;
- known abort, no-write drain, terminal-loss quarantine, and prepared-owner
  lifetime retention;
- CPU compile-time exclusion from accelerator execution owners.

The bounded resident GraphResident evidence additionally records the actual
number of successfully encoded physical controller dispatches. Existing
all-staged U64 Tile evidence remains the sealed `S=5,Q=5,C=2,B=3` case: one
dispatch in one native submit, with the GPU controller completing 15
`(batch,stage)` steps. The logical payload dispatch remains one, and Host
epoch service/callback counts remain zero on cold and warm runs. The bounded
shared `VirtualReadCohort` path covers the authenticated all-staged Q6 proof;
generic callback-backed cohort generalization remains unimplemented because
it lacks cross-input-wait-free reads, a cross-backing concurrency budget, a
callback no-reentry/cycle rule, and a dedicated bounded cohort owner/join.
TilePartial,
AddSat, S=7 outside planner-sealed static GraphResident/GraphPointwise shapes,
schedules at or above the tile bound, and dynamic
Forecast/Promote/Drain/Persist or nonresident GPU-native I/O remain
unimplemented/blockers.

Product evidence requires backend-produced counters. An admitted `StagedLoop`
reports one native recurrence submit, Q GPU epochs, zero transfer submits/
bytes, and zero Host epoch submit/service/callbacks. Its mapped-view route is
not reconstructed from Q. A legacy callback fallback is identified by one
Persistent handoff, `window_batch_count` equal to the sealed epoch count, and
the backend-specific physical submit count; it is not DeviceVsm-generated
recurrence. The admitted `BackendChunked` product uses two-coordinate chunks,
so Q=2/3/5 reports `ceil(Q/2)` native submissions while preserving one logical
handoff and one Final. `OneSubmit`/legacy fixed-W paths retain their own
one-submit contract as lower-level evidence only. Host Input and Output work
remains on admitted cold residency service lanes and is not evidence of a
persistent device scheduler.

These counters have distinct product interpretations: mapped all-staged
Pointwise Q2/Q3/Q5 is intended `StagedLoop` evidence; nonresident Pointwise R2
fallback uses `BackendChunked` for Q=2/3/5; and unsupported Persistent spatial
Window uses the bounded Window/Stream fallback, whose current Metal path may
make Q physical queue calls. The direct Q=5/9/257 backend checks are legacy
`OneSubmit` evidence rather than public Window or Pointwise R2 product evidence.
Neither interpretation claims same-submit GPU ownership of generic nonresident
backing data. The natural StagedLoop fixture is a passing functional E2E check
in the current tree, not a performance result or a 60-sample timing claim. It
verifies one native submit/handoff, Q GPU epochs, zero Host epoch
callbacks/transfers, and one Final/publication/version. Shared Persistent
Q2/Q3/Q5 `BackendChunked` and direct API Q5/Q9/Q257 checks remain separate from
public StagedLoop evidence. Range preflight rejects the reserved sentinel and
overflow request before native owner allocation; this does not attribute every
past IOGPU abort or establish safety for every finite-Q device/driver. Valid
finite-Q direct Metal O(Q) encoding remains blocked on authoritative native
command-capacity/ICB migration.

The separate WindowRingFused fixture exercises resident and staged bare-I32
Window Q2/Q3/Q5 cases, including signed boundary inputs and wrapping Sum
reference arithmetic. The same proof also admits the exact authenticated
parameter-free canonical-total U32 depth-1–3 prefix/suffix forms. Its route
proof, one-dispatch/one-submit counters, Q/Q/2Q
logical counters, zero Host/transfer counters, and one Final/publication/version
are checked independently of the legacy Window/Stream and Persistent oracles.
No new Known/Unknown fault case is claimed here; the existing terminal
contracts remain the authority for those dispositions.

The backend-neutral persistent-product seam now has a focused fake-backend
contract for finite Q=5, Q=9, and Q=257. Its request structurally excludes
per-coordinate `Project`, `Release`, and `Returned` callbacks and carries one
aggregate `Final`. This seam is the legacy `OneSubmit` mode: it reports one
fake native submit, zero Host epoch
and backend epoch callbacks/submits, Q backing signals, Q synchronous waits, Q
post-Drain/Persist acknowledgements, Q fake GPU completions, and one Final
only after the last acknowledgement. It also seals the final short-tail local
count and exact service identity. This proves the type and
accounting contract only.
The `UINT64_MAX` no-coordinate sentinel is not a workload size; the shared
range predicate rejects it and any control/descriptor generation overflow
before a backend owner is allocated.

The actual Metal and Vulkan persistent contracts additionally execute the
bounded chunked Q=2/3/5 cases through the public `VirtualPipeline::run()`
runner. Each fixture ends
with a physically short pointwise input page and proves that exact backing-read
bytes exclude the inactive suffix while physical promotion bytes cover the
fully materialized, zero-filled frame. A test-local
`DeviceOps` copy requires and wraps the exact non-null production product
operations for observation, delegates through them, and restores the original
table unchanged. The
actual backend counters report one public handoff, `ceil(Q/2)` physical native
submissions, zero epoch native submits/callbacks, Q GPU
completions/signals/waits/acks, one Final, exact output, one backing version
increment, and one generation/payload increment in each of the two physical
Pipeline banks. This is backend and
common-coordinator and active production-route evidence. Focused public-run
cases additionally prove Known first-unsent
failure drains a Q-sized no-dispatch suffix and retries, while Unknown emits
one terminal, publishes nothing, retains quarantine lifetime, and rejects a
retry without a second queue submit. The separate pause-hook observer contract
proves the success publication is atomic across both banks, backing, and
Authority admission.

The distinct DeviceVsm public product contract exercises the same public
Virtual runner without the service-aware persistent request. Its test-local
DeviceOps wrapper requires and delegates through the exact non-null production
prepare/execute entries, so fallback cannot satisfy the oracle. Metal and
Vulkan Q=5,9,257 each report one queue submit, one logical payload dispatch
(the bounded GraphResident five-stage/three-batch case additionally records one
physical controller dispatch and 15 GPU steps), zero epoch native submits, zero Host service turns/callbacks, Q GPU-generated and
completed page epochs, exact output, one aggregate Final, one Authority
accept, two Pipeline terminals, and one backing version publication. The
long public callback-backing case extends this exact oracle to Q=100001 and
also requires zero public resident rows, one whole-run-staged input, one
whole-run-staged output, and a false bounded-external-page-service bit. This
proves that the GPU-owned recurrence and the pre/post whole-run backing copy
are separate authorities; neither resident aliasing nor a hidden Host epoch
service can satisfy the case. Pointwise and `GraphPointwise` cases also
require a Q-invariant physical native
ring and three GPU result words that a Host-only completion counter cannot
synthesize. The ring contains `4W` turn-state bytes and
`resident_count*W*payload_bytes` scratch bytes. Each page atomically consumes
its exact prior slot turn, copies every resident input into a distinct scratch
frame, executes the payload with only scratch input/output bindings, and copies
the output scratch back to the resident destination. Final requires
`Q*resident_count` such physical round-trips, exactly `Q-W` reuse transitions,
and a modulo-2^32 checksum over each page ordinal, exact
`page mod W` slot, one-based slot turn, and active element count. Common Final
computes the expected receipt and bytes in O(1); actual Metal and Vulkan
Q=5,9,257 plus the long-Q cohort close it with Q-invariant retained bytes. The
whole-run backing rows remain GPU-addressable residents; this is not a runtime
Host/GPU service ring. The
Window cohort includes physical five-stage and maximum seven-stage
`Map -> Map -> Window -> Map -> Map` and
`Map -> Map -> Map -> Window -> Map -> Map -> Map` public programs for every
Sum/Min/Max and Clamp/Clip combination at Q=5,9,257. Every Map expression is
an independently retained canonical U32 authority, adjacent pairs exceed the
ordinary static expression-composition envelope, and the proof authenticates
all five or seven stages while retaining the same
one-submit/zero-Host-epoch/one-Final law and Q-invariant controller bytes.
Every Window case additionally requires
the GPU aggregate result to equal the proof-sealed `Q-1` canonical page
transitions and the exact checksum of all page/epoch projections; a Host-only
page counter cannot satisfy this oracle. The
two-input Graph-to-{Sum, CountNonzero, Min, Max} cases additionally bind two
distinct U64 backings in Program-port order, evaluate
`(a xor 0x55) + 3*b`, and prove `2Q` page-ins and twice the logical input bytes
with the same one-submit/zero-Host-epoch/one-Final law at Q=5,9,257 on both
actual backends. The maximum-width Graph case binds six external U64 inputs,
uses the remaining two fixed resource classes for the Map intermediate and
terminal output, and proves every supported reduction at Q=5 on both actual
backends: thirty exact input pages, one submit, zero Host epoch control, five
generated/completed pages, one Final, and one
Authority/two-Pipeline/one-backing publication. The
distinct maximum-width `GraphPointwise` case binds seven external U64 backings
to a two-stage graph at Q=5. It proves thirty-five exact page reads totaling 4,088
bytes, one queue submit and payload dispatch, zero epoch submits/Host service turns/callbacks,
five GPU-generated/completed page epochs, exact output, one Final, and the
same one-Authority/two-Pipeline/one-backing transaction on Metal and Vulkan.
The distinct deep `GraphPointwise` case binds six external U64 backings to
three stages at Q=5. It proves thirty exact page reads totaling 3,504
bytes, one queue submit and payload dispatch, zero epoch submits/Host service
turns/callbacks, five GPU-generated/completed page epochs, exact output, one
Final, and one atomic publication transaction containing one Authority accept,
three stage terminals, and one backing commit on both actual backends.
The five-input/four-stage `GraphPointwise` case proves twenty-five exact page
reads totaling 2,920 bytes, one queue submit and payload dispatch, zero epoch
submits/Host service turns/callbacks, five GPU-generated/completed page epochs,
exact output, one Final, and one transaction containing one Authority accept,
four stage terminals, and one backing commit on Metal and Vulkan. The three
continuation stages include XOR,
unsigned wrapping multiplication, and addition; their exact output proves that
the public route is not relying on an Add-only GraphPointwise whitelist.
The four-input/five-stage `GraphPointwise` case uses nine planner resources:
four external rows, four intermediate values, and one output.
At Q=5 it proves twenty exact page reads totaling 2,336 bytes, one queue
submit and payload dispatch, zero epoch submits/Host service turns/callbacks,
five GPU-generated/completed page epochs, exact output, one Final, and the
one transaction containing one Authority accept, five stage terminals, and
one backing commit on Metal and Vulkan.
The three-input/six-stage case proves fifteen exact page reads totaling 1,752
bytes and commits six stage terminals in that same transaction. The
two-input/seven-stage case proves ten exact page reads totaling 1,168 bytes and
commits seven stage terminals. Each runs Q=5 with one queue submit/payload
dispatch, zero epoch submits/Host service turns/callbacks, five GPU-generated
and completed page epochs, exact output, one Final, and one backing commit on
both actual backends.
The
admitted pure unsigned U32/U64 Inclusive and Exclusive Scan, exact canonical
total element-local one-through-seven-read/one-value-write U64 Map expression DAG
followed by U64 Scan, and
Sum/CountNonzero/Min/Max Reduce cases additionally prove device-owned state
across Q distinct pages. Reduce publishes one scalar; Scan owns its carry. The
composed Scan cases authenticate the canonical typed Map stage, its complete
ParsedIR-derived source, and exact parameter-byte contract while retaining one
payload dispatch and no intermediate VSM stage. Actual cases cover the
optimized `+7` form, `(x xor 0x55) * 3 + 7`, and two distinct resident inputs
combined as `a+b`. The two-input Q=5/9/257 Inclusive/Exclusive cases prove one
queue submit, zero Host epoch control, exact `2Q` page-ins and twice the
logical backing-read bytes, one aggregate Final, and one publication on both
Metal and Vulkan. The Scan and Sum
checked-overflow receipt names the exact first failed page, authenticates
Known/no-write, publishes
nothing, and reuses the same Pipeline state for a successful retry. The
backend retained controller bytes are equal across the three Q values. This is
success-path and Scan/Sum Known-failure evidence for the admitted whole-run
GPU-addressable-backing shapes; topology-specific Unknown, allocation, and
performance evidence remain separate.

The common maximum-width source oracle composes seven ordered U64 reads into
the same two-stage Map/Scan proof for Metal and Vulkan and checks exact
`7*8`-byte input plus `8`-byte output work per tile. Eight inputs are rejected
before execution. Actual Metal and Vulkan Q=5 Inclusive/Exclusive cases bind
all seven rows, read 35 input pages, submit once, perform zero Host epoch
control, and publish one aggregate Final; the independent one/two-input cohort
retains Q=5,9,257 depth coverage.

The stronger resident service-free Direct product route is separate from that
backing-serviced schedule. Actual Metal and Vulkan Q=5,9,257 contracts execute
the public `Pipeline::repeat<Q>::run()` route and prove one native queue submit,
one fused payload dispatch, zero Host service turns and epoch callbacks, one
aggregate Final/publication, exact output, fixed native storage, and identical
retained common Host/Device/Staging memory across Q. The immutable proof and
backend capability must both authenticate the fixed-common-storage fact;
capability-only relabeling is rejected.

The focused contracts own Plan tail projection, large-Q O(1) admission,
generation-zero attempts, global shifted chunks, stale receipts, failure and
Unknown terminals, global-Host warm cache masks, coherent direct-backing
sentinels, owner quarantine, and immediate Authority reuse after known
pre-native/between-chunk abort. Metal and Vulkan product contracts own actual
Q5 and Q9 execution, output, transfer-elision, and public receipt shapes. The
coherent Direct five-page/two-bank warm cohort retains all useful Host/Device
pages: after five cold reads, sixty warm runs perform zero Persistent reads and
zero H2D/D2H submissions on both admitted Metal and Vulkan paths. The rolling
parallel-read cohort additionally proves that a Host cache supply can reduce
two logical Device page-ins to one Persistent read; a coherent promotion may
be a direct Host write or one exact H2D receipt and never creates overlap
evidence. A pending HostRead/HostWrite view fault forces the rolling producer,
which must consume the injected fault rather than masking it behind coherent
readiness.

Still partial: general Graph, data-dependent or more-than-seven-input
Map-to-Scan composition,
unsupported Scan/Reduce operations and types,
Window/halo fill,
noncoherent transfer service, device-generated recurrence for the
backing-serviced persistent seam,
external transactional backing, and a continuous sliding recurrence for the
Host-driven chunk fallback. The legacy `OneSubmit` backing-serviced lowering
may retain transient O(Q) native descriptors; `BackendChunked` retains only its
fixed two-slot backend records and performs `ceil(Q/2)` submissions. Neither
is an O(1) device-generated-command proof. Performance results are measurement
evidence only and do not establish a general GPU win.
