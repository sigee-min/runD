# Scheduler Lanes

Scheduler lanes execute admitted leaf bodies and resume native coroutine
frames. They are an orchestration resource, not a second task representation.

## Ownership

The runtime configures a fixed lane width for its lifetime. A task record has a
stable home lane, but one fixed-capacity FIFO owns every ready transition.
Home lanes affect physical execution placement only; they cannot create a
second ready order. Ordinary front removal never shifts the remaining task
ids. The queue reserves `task_capacity` physical slots for lossless wakes of
already-admitted tasks and awaited child continuations, while
`ready_queue_capacity` remains the independent-spawn admission bound. Worker
count therefore changes neither queue storage nor ready order. Lanes do not
allocate task stacks or nested worker pools.

## Execution

Each lane claims one scheduler job, installs the task identity and commit
context, executes one leaf body or one coroutine resume quantum, publishes its
effect, and signals completion. A parked coroutine returns control normally
from `resume()`; a terminal coroutine publishes its completion and releases
its frame exactly once.

Leaf bodies run to completion. Calls to suspension primitives fail with
`task_leaf_primitive_forbidden`.

Ready leaf tasks are admitted to lane-owned segments in batches. One fixed
lane job carries the segment's logical task range, and ownership of its
pre-sized job storage moves into the worker frame without a per-task copy or
allocation. Coroutine tasks remain individual orchestration quanta because
their suspension and wake sources are semantic boundaries. The global FIFO
head for a semantic side exit is never bypassed merely because its physical
home lane is busy. A later ready task therefore cannot receive an earlier
commit ticket through lane timing. The root observes that head without
removing it, so physical wait duration does not create repeated queue-pop
telemetry. A leaf batch also stops when it encounters the first semantic side
exit and never harvests later leaves across that boundary. When the next
ordinary leaf belongs to a busy lane, the dispatcher removes that one
canonical head, assigns its next commit ticket, and links it into the lane's
retained direct queue before selecting another ready task. The task therefore
keeps its logical place without a second availability queue, completion map,
or per-task scheduling flag. This adds no sorting or allocation.

An externally blocked coroutine does not re-enter that ready FIFO when its
admitted operation completes. The wake owner issues its commit ticket before
linking the record into the home lane's external queue, and the lane worker
selects regular, external, direct, and mailbox work by that same ticket order.
This is a continuation of the canonical commit authority, not a second ready
ordering authority.

A root-submitted external coordinator also uses the ordinary `PopReady(0)`
selector. The dispatcher drains the complete canonical prefix through the
coordinator, reserves one contiguous ticket interval, and publishes every
prefix record under one all-lane transaction. It never selects by scope or by
home-lane index, because either would create a second order beside the normal
global-plus-round-robin selector. The prefix uses one retained
`task_capacity` scratch vector, so a wake-expanded ready depth greater than
`ready_queue_capacity` cannot allocate or terminate a `noexcept` dispatch.

Leaf calculation and scheduler commit are separate phases. For one admitted
ticket interval `[B, B + N)`, the planner stores the owning lane `h(i)` for
each canonical offset `i`. Every lane publishes only its greatest completed
ticket `p[lane]`. The shared commit frontier is therefore the greatest `F`
for which every earlier ticket is complete:

```text
F = max k such that p[h(i)] >= B + i for every 0 <= i < k - B
```

Every ordinary leaf calculates before waiting for its commit turn. A leaf that
reaches a host event, runtime-state query, or scheduler primitive acquires its
ticket at that boundary. The queried state owns this synchronization instead
of relying on a caller-side precondition. A pure leaf acquires its ticket after
calculation and before publishing
success, failure, evidence, terminal state, or join wakes. Once acquired, the
leaf holds the ticket through the rest of its quantum and advances the same
frontier. Thus user calculation can occupy all lanes while scheduler-owned
effects keep one logical order. A blocked home lane changes only calculation
time: the retained direct ticket still precedes every ticket issued later.
The scheduler does not claim determinism for raw user pointer races or
unsynchronized external side effects.

The root sequencer removes a leaf record from the ready authority before it
hands that record to exactly one lane. The lane owns only calculation and an
ephemeral `LaneSegmentEffect`; it does not publish terminal `TaskRecord`
state, failure, evidence, join wakes, or callable destruction. The root merges
those effects in canonical ticket order under one recursive evidence lock and
then materializes each terminal record exactly once. An all-success segment is
represented by its contiguous task/ticket summary. If a rare failure or trap
invalidates that summary, the root linearly merges the executed job prefix
with explicit effects and synthesizes omitted successes in `O(N+E)` time with
no allocation. Thus the common path keeps zero per-task effect packets while
the slow path cannot strand a calculated task in `Running`.
The summary participant hash is fused into that same canonical materialization
loop as `(task, ticket, kind, code, rank)`; physical lane index, lane count,
and per-lane range shape never enter it.

Primitive metadata lives only in the ephemeral effect until root commit. Leaf
bodies cannot suspend and therefore produce no local-yield state. Callable
destructors run only at canonical root publication, so destructor side effects
cannot expose physical lane completion order.

Lane availability probes are advisory. Every ordinary batch submission
rechecks the complete idle predicate while holding the target lane mutex; no
prechecked flag can bypass that lock boundary. A lane-owned segment first
claims each participating idle lane by setting its root reservation under the
same mutex. If any claim fails, earlier claims are released and the dispatcher
retains every popped task id. Publication then consumes those claims and cannot
overwrite a live lane slot.

The dispatcher is the sole owner of a popped batch until each id is either
accepted by a lane or restored once through the ready requeue owner. Planning,
ticket assignment, and lane claims never restore queue entries themselves.
All segment job and effect capacity needed after publication is retained or
reserved before the first lane is notified. Once any lane executes, result
collection and canonical commit perform no fallible allocation and cannot
return executed ids to the queue as an allocation fallback.

## Deterministic Commit

Parallel lane execution may finish in any physical order. Effects are merged
by canonical task/ticket order. Terminal state, join wake, channel wake, and
failure selection are committed by that logical order, never by wall-clock
completion order.

The frontier is a monotone atomic value and lane completion is release/acquire
published. No task-count threshold, worker-count branch, CPU-specific spin
constant, work stealing, or alternate serial implementation selects this
path. Primitive waiters use the same frontier on every supported host, while
pure leaves perform one lane-local completion publication and may advance a
contiguous ready prefix.

The public `side_exit_count` telemetry coordinate counts semantic exits from a
lane-owned segment. A primitive trap and each admitted task returned to the
global ready authority contribute through the same `SideExits` storage slot;
there is no recovery-path or fallback counter with overlapping meaning.

## Compute

Node Compute coordinators are coroutine tasks. Accelerator completion callbacks
wake the parked coordinator. CPU tile execution remains owned by Kernel's
worker backend; lanes must not create a nested CPU pool.
