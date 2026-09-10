# Native Whole-Run Scheduler

This page owns the target and lower bounds for moving recurrent Direct
execution from Host chunk chaining to one native whole-run schedule. The
immutable `execution::Plan`, one Authority lease, exact Host-service tickets,
and one backing publication remain unchanged. A backend lowering may differ
in native representation, but it may not invent a terminal, hide a queue
call, or retain uncharged storage.

Status: implemented for explicit/coherent or resident Direct schedule contracts
and for the dedicated bounded nonresident Window/Stream fallback on Metal and
Vulkan. An eligible ordinary all-staged unary nonresident Pointwise
Q>=2 first selects `StagedLoop` when exact mapped Host-visible/coherent input
and output views exist; it owns one native recurrence submit, Q GPU epochs,
zero transfer submits/bytes, zero Host epoch submit/service/callbacks, and one
Final/Authority/output publication/version. A pre-lease structural
`BackendUnsupported` cleanly declines to the legacy route only when no owner,
rearm, stage, lease, or native acceptance exists; mutation or any
non-capability failure is terminal. Persistent Pointwise R2 remains the
`BackendChunked` fallback with `ceil(Q/2)` submissions, while unsupported
Persistent spatial Window declines pre-lease to bounded Window/Stream.
Vulkan
lowers the whole recurrence to Q timeline submit records in
one actual `vkQueueSubmit`; its contracts execute Q=17. Metal cold-prepares Q
distinct MTL4 commands, performs Q queue wait/commit/signal operations, and
its contracts execute Q=5 and Q=9. Both feed one aggregate schedule terminal
into the same Stream/Authority owner. Dynamic true fixed-R PagedLoop remains
blocked by the absence of a portable same-submit GPU↔Host system-scope
rendezvous/forward-progress primitive. The common model covers Q=100,000 success
and a Q=101 Known failure whose accepted suffix drains through authenticated
completed/no-write terminals without overflowing the bounded failure journal.
The fixed-chunk Stream remains the pre-Authority fallback when native schedule
preparation is unavailable or cannot reserve its exact storage. Non-Direct
shapes retain their documented routes. The separate DeviceVsm scheduler owns
the implemented pure unsigned U32/U64 Inclusive/Exclusive Scan, exact total
canonical element-local one-through-seven-read/one-value-write U64 Map-DAG-to-Scan
composition, and
Sum/CountNonzero/Min/Max Reduce and is not represented by this Direct Schedule.

## Single authority

The product must not grow a second scheduler beside `execution::Plan`.
`execution::Schedule` is therefore a sealed lowering view of that Plan, not a
new page or cache formula. It contains only:

Compute schedule source ownership follows the same lifecycle boundary:
`pipeline/execution/schedule.cpp` owns plan lowering, immutable revalidation,
submit, signal, and abort, while `pipeline/execution/schedule/callback.cpp` owns
release/final callbacks and terminal evidence projection. The adjacent
`callback.hpp` is declaration-only and does not duplicate either implementation
authority.
The accelerator-side prepared adapter mirrors that lifecycle without creating
a scheduler: `accel/kernel/prepared/schedule.cpp` owns the one submit
transition, `schedule/validation.cpp` owns request/role validation,
`schedule/completion.cpp` owns release/final evidence and teardown,
`schedule/preparation.cpp` owns backend template preparation, and
`schedule/control.cpp` owns signal/abort. Their local header is
declarations-only, and every phase mutates the same
`PreparedResidencyScheduleControl` selected at submit.

```text
RunCredential = (plan identity, Authority token, run generation)
Q             = Plan.epoch_count
R             = four logical native roles: (bank, publication parity)
role(e)       = e mod R
bank(e)       = e mod 2
readyCell(e)  = e mod R
readyTurn(e)  = floor(e / R)
doneTurn(e)   = e
```

The four roles are logical even when a nontransactional Pipeline deduplicates
two prepared owners. This avoids making pointer aliasing a scheduling policy.
Page uses, dirty tails, locals, predecessor edges, and mutation regions remain
projected by the Plan. A backend receives immutable role templates and the
closed recurrence above; it never receives a materialized Q-entry policy
array.

One `ScheduleControl` owns the complete state transition:

```text
Empty -> Claimed -> Submitted -> Servicing -> Draining -> Closed
                                      \-> UnknownQuarantine
```

`Submitted` means all Q native batches were accepted. No later Host action may
enqueue a Dispatch. The Host service lane may only consume an authenticated
done turn, perform Output(e) and Input(e+2), and signal the already-submitted
ready turn. The caller performs the initial two Input services, one public
handoff, and one Final wait.

The fixed mutable journal is four role cells plus two Host-service cells. Each
cell carries its global epoch tag; reuse requires an exact prior terminal and
tag match. An aggregate Final carries Q, completed prefix, suppressed suffix,
queue-call facts, and certainty. It cannot carry or reconstruct a Q-sized
receipt array.

## Cost model and admission

Native lowering cost is part of cold schedule preparation, not a Boolean
capability query or an implementation detail hidden after Authority begins:

```text
retained_common(Q)  = O(R + banks + locals)
vulkan_transient(Q) = Q * (VkSubmitInfo + timeline values/stages)
vulkan_retained(Q)  = O(R)
metal_pending(Q)    = Q * (allocator + command buffer metadata)
```

Vulkan reports the logical Q-record transient payload bound and releases those
records after `vkQueueSubmit` returns. Allocator metadata, vector capacity
rounding, and allocation-owner bookkeeping are not included and are not
mislabelled as an exact process-heap footprint. Metal first freezes the remaining Device
Pipeline budget, materializes the cold candidate, partitions its exact tracked
retained charge, commits that partition, and refunds the unused reservation.
A `retained_bytes-1` contract proves a miss destroys the candidate and selects
the existing chunk Stream before any Authority token exists; refunding the
reservation then admits the same whole schedule.

The backend-neutral preparation result therefore names the selected lowering,
owns any cold candidate, reports retained/transient byte bounds and the
native queue-call count, and proves callbacks are asynchronous. Compute
switches on that result, never on Metal or Vulkan. A shared admission owner
keeps the committed reservation alive until the backend Final callback
returns.

Common admission reserves the sum of reported retained bytes and Vulkan's
Q-proportional transient Host lowering payload bound before Authority. A huge-Q
candidate that exceeds the frozen Pipeline budget therefore selects fallback
before an execution token. This is payload admission, not bounded-memory or
an exact heap-footprint claim: the
existing Schedule still allocates Q records at submit time and must not be
described as warm allocation-free. The fixed-slot replacement is owned by
[Sliding](./sliding.md).

Backing latency is also part of admission. Ordinary all-staged nonresident
unary Pointwise Q>=2 first probes `StagedLoop`; exact mapped Host-visible/
coherent views are required and the route enum is not recomputed by the
backend. A structural pre-lease decline reaches the legacy Persistent route,
where R2 inputs use `BackendChunked`; unsupported Persistent spatial Window
reaches dedicated bounded Window/Stream. Explicit/resident/required DeviceVsm
retains its independent native policy, and the rolling fallback remains
available when the selected capability is unavailable. This is a
backend-neutral capability decision, not a Metal special case. A shape that
admits two or more read lanes also remains on its authenticated BackendChunked
or rolling owner rather than being relabeled as `StagedLoop`.

## Terminal algebra

Success requires Q authenticated native terminals and Q exact Output service
terminals. A Known failure at epoch `f` has two disjoint regions:

```text
[0, f)  = exact completed/service prefix
f       = exact failed phase and may-write mask
(f, Q)  = accepted native suffix, completed through the no-write gate
```

The suffix is one algebraic interval in Final evidence, not Q duplicate
failures. Authority validates the recurrence, the first failure, the interval
cardinality, and the final timeline turn in O(1). `UnknownMayWrite` carries no
invented completion interval: the complete schedule, four prepared roles, all
physical banks, and backing recovery generation remain quarantined.

External backing publication occurs once, after Authority close. Per-epoch
Output writes target a recovery/private generation; an intermediate service
terminal never advances the public version.

## Dependency law

With two physical banks, epoch `e+2` can become ready only after the exact
native terminal and Host Output/Input service for the prior use of its bank:

```text
done(e) -> Output(e) -> Input(e+2) -> ready(e+2) -> Dispatch(e+2)
```

Submitting a command after `done(e)` recreates a Host terminal/submit edge.
A native whole-run schedule must therefore place every future wait, dispatch,
and done signal on the native queue before `ready(0)` opens. Host code may
service an authenticated Release and signal readiness; it may not enqueue the
next Dispatch.

## Vulkan lowering

Vulkan timeline submit records can express the complete recurrent chain in
one queue call. Epoch `e` waits ready cell `e mod 4`, executes a reusable
`SIMULTANEOUS_USE` command batch, and signals the globally ordered done value.
The four ready cells retain independent monotonic values, so signalling a
later reuse cannot open an unsignalled cell from another bank/class.

The Vulkan API requires one `VkSubmitInfo` and one timeline-info record per
epoch. Those records are transient Host lowering storage `O(Q)` and must be
allocated, bounded, and reported as such; they are not retained Authority,
Pipeline, page, or command owners. The reusable native commands and recurrent
Host-service journal remain `O(1)` in Q. Calling `vkQueueSubmit` once is a
native fact, not a relabelled public handoff.

An upfront schedule accepts every Q native batch. A later Host-service failure
therefore cannot reuse the chunk-Stream law that calls future chunks "unsent".
The production Authority schedule path implements the Known law: it drains
every already accepted suffix through authenticated no-write gates and closes
from one aggregate terminal. An unknown native terminal publishes only the
known prefix plus one exact Unknown receipt, quarantines the adapter, prepared
owners, physical banks, and Authority generation, and returns sticky
`DeviceLost`. Public Vulkan Q>4 evidence is therefore one handoff, Q native
batches, and one queue call.

The Vulkan implementation keeps that boundary physically direct-owned under
`node/src/accel/vulkan/kernel/pipeline/residency/schedule/`: `entry.cpp`
validates the request and projects capability, `submission.cpp` performs role
materialization and the single queue submit, `signal.cpp` authenticates ready
signals, `terminal.cpp` owns service/release/final projection, and
`cleanup.cpp` owns active-slot clearing and abort. Its private header carries
declarations only; the existing `VulkanResidencyScheduleRun` remains the sole
mutable schedule owner.

## Metal lower bound

MTL4 queue event waits and signals are queue operations around command-buffer
commits; they cannot be attached to individual entries of one
`commit:count:` array. The currently admitted SDK also provides no
device-generated dispatch primitive that can wait for a future Host-service
receipt and then invoke an existing Pipeline command stream.

An actual macOS 26 device probe encoded a non-empty MTL4 compute command. One
commit completed successfully. Reusing that same command buffer a second time
before its first terminal, both in one commit array and in two immediate
commit calls, was rejected by the driver. Consequently arbitrary-Q upfront
Metal scheduling requires either:

- Q distinct pending command-buffer storage and Q queue wait/commit/signal
  operations, all explicitly charged to the execution owner; or
- a future persistent/device-generated controller with a proven Host-ready
  memory and forward-progress model.

Two retained bank commands, constant native command storage, arbitrary Q, and
zero mid-run Host submit cannot all be claimed with the admitted MTL4 API. The
implemented lowering therefore deliberately selects the first honest option:
it cold-owns Q allocators/commands, charges their tracked retained bytes before
Authority, and queues Q native wait/commit/signal operations during one public
handoff. Listener callbacks only mark completion. One Schedule-owned serial
delivery lane emits the global contiguous Release prefix, so independent bank
listeners cannot reorder Authority receipts. Known admission failure drains
the accepted suffix with no-write gates and permits immediate same-owner
retry; terminal loss produces one Unknown Final and sticky Device quarantine.
The fixed-chunk Stream remains the budget/capability fallback. A future
persistent/device-generated controller is still required for O(1) native
command storage or asymptotic Metal queue-operation reduction.

## Non-Direct boundary

The pure unsigned U32/U64 Inclusive/Exclusive DeviceVsm Scan owns its carry
and overflow precheck on device. Its admitted exact canonical total
element-local U64 Map expression DAG consumes one through seven resident input rows
and is applied inside that same dispatch before the carry operation. Pure
unsigned U32/U64
Sum/CountNonzero/Min/Max Reduce owns its accumulator or identity tree and final
scalar on device. All close through one aggregate Final. Other Scan/Reduce
and Graph routes have route-specific recurrence and publication dependencies.
Moving them to this scheduler requires an on-device state machine and exact
failure/publication evidence; wrapping their existing Host loop in a callback
is not GPU-driven execution. Their typed topology, state, dependency, and
transfer-queue laws are owned by [Routes](./routes.md).
