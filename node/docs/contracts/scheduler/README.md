# Scheduler Contract

This directory is the scheduler contract authority for deterministic
lightweight task execution under the public `rund::task` product domain and
the source-private `rund::node` scheduler implementation.

## Authority Path

1. [`/docs`](../../../../docs/README.md) for repository process, naming, and
   architecture routing.
2. [`/node/docs`](../../README.md) for the node runtime boundary.
3. [`/node/docs/contracts`](../README.md) for node contract routing.
4. This directory for scheduler behavior.
5. Installed support owners under `/node/include/rund/task/`, source-private
   runtime owners under `/node/include/node/runtime/runtime.hpp`, and the
   transitive facade support header `/node/include/rund/session.hpp`.
6. Implementation under `/node/src/runtime/task/`,
   `/node/src/runtime/task/scheduler/`, and `/node/src/runtime/runtime.cpp`.
7. Contract tests under `/node/tests/contract/`.

If this contract and implementation disagree, the task remains open until both
are updated or the conflict is recorded as a blocker.

## Pages

| Page | Owns |
| --- | --- |
| [Public](./public.md) | Scheduler scope, SDK projection, source-header owners, handle boundary, result shapes, and public/internal include separation. |
| [State Source](./state/source.md) | Live scheduler contract and private scheduler source-map entries. |
| [State Runtime](./state/runtime.md) | Stop tokens, configuration normalization, resource budgets, and Kernel provider interaction. |
| [State ABI](./state/abi.md) | Stable reason, operation, and observation ABI tables. |
| [Reactor](./reactor.md) | Scheduler-owned fd readiness reactor, timed reactor waits, backend normalization, readiness ordering, fd generation ownership, and network readiness integration. |
| [Coroutine Task](../coroutine.md) | Native `Task<T>` coroutine frames, await suspension, completion, cancellation, and leaf-task boundaries. |
| [Lane](./lane.md) | Task-worker lanes, home-lane ownership, lane segments, worker/executor execution model, deterministic effect merge, and task lifecycle reuse. |
| [Channel](./channel.md) | Channel lifetime, endpoint/control-block ownership, deterministic send/recv/close ordering, coroutine parking, and channel result compaction. |
| [Progress](./progress.md) | Root/scope execution, ready progress, yield/sleep progress, join/scope waits, recursive progress, deadlock wake behavior, and final drain propagation. |
| [Platform](./platform.md) | Portable scheduler core, readiness adapters, and platform network byte-substrate wrappers. |
| [Verification](./verification.md) | Owner `node-*` executables, ordered in-process contract groups, exact-case selection, task CTest checks, and package verification routing. |

## Scope Route

Scheduler scope is owned by [Public](./public.md#scope). This index routes
authority pages and does not restate scheduler behavior.
