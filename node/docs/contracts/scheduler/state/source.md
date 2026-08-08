# Scheduler State Source Map

This page maps the private implementation of the single native task scheduler.
Public behavior is owned by the sibling scheduler contract pages.

## State

- `state.hpp` is the private scheduler surface router. It owns the `Scheduler`
  declaration and an opaque `SchedulerState*`; declaration-only host, network,
  Runtime, and task bridges do not import the complete scheduler storage graph.
- `state/forward.hpp` owns incomplete declarations used only by that surface.
  The surface does not import complete Host I/O, random-stream, or scope
  evidence aggregates.
- `state/model/` partitions complete scheduler values by ownership:
  `task`, `wake`, `timer`, `stop`, `join`, `work`, `batch`, `segment`, `lane`,
  and `context`. There is no model aggregate. Each implementation imports only
  the complete value it reads or writes.
- `state/storage/host/slot.hpp` owns the complete Host I/O slot and its wake
  value. `state/storage/host/io.hpp` retains only the Host I/O state shell;
  its compiled constructor and destructor make the incomplete slot array a
  valid lifetime boundary without adding an allocation or dispatch.
- `state/storage.hpp` is the sole complete `SchedulerState` layout owner. It
  aggregates identity, resources, ready queues, reactor, lanes, evidence, and
  batch state and is included only by implementation owners that dereference
  scheduler storage. It is not a public facade.
- `rund/host/io/fd.hpp` and `rund/host/random/seed.hpp` are the exact complete
  value owners needed by Scheduler storage. The full `rund/host/io.hpp` and
  `rund/host/random.hpp` aggregates remain operation surfaces and do not enter
  the Scheduler declaration or storage graph.
- `state/storage/evidence.hpp` owns the single inline live scheduler telemetry
  block. Counter mutation uses the append-only slot identities; public
  `task::Stats` is materialized only by `core/snapshot.cpp` while holding
  the evidence lock.
- `stats/schema/slots.def` owns the current physical identity and
  `stats/schema/public/*.def` owns public-name projection. Private replay wire
  order lives in `node/runtime/replay/task/stats.hpp` and semantic-hash order in
  `node/runtime/replay/task/schema/semantic.def`; both reference current slots
  by name rather than physical index. Each compact slot has one live producer
  meaning and one public getter.
  Source-only `stats/access.hpp` is the only authority that can materialize or
  read/write a snapshot by slot.
- `state/storage/batch.hpp` owns commit-ticket sequencing and one fixed pending
  root-join telemetry range; it does not mirror task lifetime or use allocator
  capacity as a trace boundary.
- `state/storage/ready.hpp` owns task records, id indexes, free slots, timers,
  and the configuration-reserved join-wait array.
- `state/storage/resources.hpp` owns scheduler limits, the reusable coroutine
  frame arena, the completion pool, channel capacity, and prepared memory.
- `state/task.hpp` declares task execution, retirement, handle matching, and
  deterministic commit helpers.
- `state/task/commit.hpp` owns the complete internal commit scope and
  `state/storage/check.hpp` owns the inline sequencer assertion. These hot
  helpers remain directly compiled at their true call sites; no virtual,
  erased, or heap-backed boundary was introduced to reduce build fan-out.
- `state/task/index.cpp`, `state/ready/queue.cpp`, and `state/terminal.cpp` own
  task lookup, ready admission, and terminal/failure selection.
- `state/record.hpp` and `core/record/` own deterministic logical evidence.

## Execution

- `core/spawn.cpp` and `core/spawn/` own leaf/native-coroutine admission,
  record reuse, identity, budget validation, enqueue, and rejection cleanup.
- `task/context/switch.cpp` routes one task quantum to either native coroutine
  execution or a non-suspending leaf.
- `task/context/switch/coroutine.cpp` resumes the scheduler-owned coroutine
  frame and validates its parked or terminal state.
- `task/context/switch/leaf.cpp` executes one leaf callable to completion.
- `task/retire.cpp` commits results, destroys frames exactly once, and recycles
  task records immediately. Deferred root-join state is telemetry only.
- `task/commit.cpp` owns the internal control-commit scope used by external
  spawn, post-terminal root retirement, and snapshot batch materialization.
- `task/external/` owns callback-driven external completion park/wake used by
  Node-native accelerator Compute.
- `lane/` owns persistent task workers, mailbox dispatch, lane segments, and
  deterministic effect commit.

## Suspension Sources

- `progress/` owns ready progress, join, scope, and final drain.
- `timer.cpp` and `timer/store.cpp` own sleep deadlines and timer wake order.
- `reactor/` owns single, timed, many, and ready-set I/O readiness waits.
- `channel/` owns bounded channel state, wait queues, wake, and close.
- `cancel/` owns stop-source identity, cancellation request order, and wait
  cleanup.

No private scheduler component owns a second task representation, worker pool,
continuation frame ABI, or mutable scheduler-stat snapshot.
