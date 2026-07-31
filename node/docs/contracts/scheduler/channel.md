# Scheduler Channel Contract

Channels are scheduler-owned bounded communication records. Their public
surface is `channel<T>::make`, `send`, `recv`, and `close`.

## Ownership

A channel control block owns its buffer, sender queue, receiver queue,
rendezvous values, close state, and scheduler identity. Capacity admission is
bounded by `channel_capacity`, `channel_buffer_capacity`, and
`channel_wait_capacity`.

Creating a channel allocates its configured ring buffer but does not prepay
task-indexed state for both waiter directions. Send-wait state is created on
the first parked sender; receive-wait and rendezvous state is created on the
first parked receiver. Each store remains bounded by `task_capacity` and is
retained for warm reuse. Allocation failure is reported as
`channel_capacity_exceeded`; it never silently drops a waiter.

The public channel handle is a move-only RAII owner. Its destructor closes the
channel, discards buffered values with no remaining receive path through an
endpoint, and releases the scheduler record after all already-woken operations
finish. Coroutine operations share only the private control block; they do not
create another public owner. Explicit `close()` retains buffered values for
receivers and remains the graceful-drain operation.

## Coroutine Suspension

Channel operations that cannot complete immediately return an await token.
`await_suspend` registers the current coroutine task as a sender or receiver
and parks that task record. A matching operation writes the rendezvous/buffer
state, records the logical match, and enqueues the parked record exactly once.

There is no direct task-to-task context transfer, trace-region execution,
source-batch lowering, or alternate channel suspension engine.

The private channel facade returns one decision containing `Status` while
it is still deciding whether an operation parks or completes. Send and close
return `Status` directly; receive expands the decision into
`ReceiveResult<T>`. Committed
operations finish through one private channel completion authority. Its
committed and counted paths share the same state transition; suspension
remains private scheduler state.

## Ordering

Waiters are matched in channel queue order. Each worker writes only its owned
channel/result slots; scheduler evidence is committed in logical operation
order rather than worker completion time. Close wakes all remaining waiters in
stable task order with `channel_closed`. Close transfers its detached waiter
vector to the scheduler ordering boundary and sorts that storage in place; it
does not allocate and copy a second waiter list after leaving the channel lock.

## Leaf Boundary

A non-suspending `spawn(lambda)` leaf may not call a channel operation that
requires scheduler task semantics. Such an attempt fails the leaf with
`task_leaf_primitive_forbidden`. Suspending channel work uses `Task<T>` and
`co_await`.

## Verification

Coroutine lifecycle contracts cover blocked send and receive, rendezvous,
close failure propagation, capacity rejection, exactly-once wake, and task
completion.
