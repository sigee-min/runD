# Product Acceptance

This page owns the release-facing UX gate for the installed SDK. It does not
restate subsystem mechanics. Public names are owned by
[API Stability](./api/stability.md), package reachability by
[SDK Surface](./surface.md), consumer linkage by
[SDK Consumption](./consumption.md), and execution semantics by the linked
Session, Replay, and Compute contracts.

## User Journeys

The product surface is judged from the user's task, not from the number of
available declarations. A release is accepted only when all seven journeys are
executable from the installed SDK with these UX invariants:

- one obvious entry for each task and no mode-specific graph duplication;
- one owner for identity, bytes, ordering, lifecycle state, and result truth;
- bounded memory and stable typed failures at every untrusted boundary;
- diagnostics that identify the failed operation without changing its result.

UX is evaluated as a vector rather than a subjective weighted score:

```text
F = (required entries, caller-owned authorities, unbounded storage,
     mode branches, nested result layers)
C = (heap allocations, copied bytes, scanned bytes, atomic updates, syscalls)
```

The journey table fixes the admissible components of `F`; the owning runtime
contract fixes the unavoidable lower bounds in `C`. A change is an improvement
only when it lowers at least one measured component without increasing another
or weakening semantics. This componentwise order prevents a shorter example
from hiding a new allocation, fallback, result translation, or duplicate
authority.

## Release Surface

The release candidate is accepted only when every row below is the sole live
product authority. A wrapper or diagnostic does not satisfy the required
product shape.

| Boundary | Required product shape | Evidence gate |
| --- | --- | --- |
| task vocabulary | `<rund/task.hpp>` is a composition-only focused entry: each declaration is owned once by its task header, cooperative execution uses `rund::task`, and implementation-only task machinery lives under `rund::detail::task`. `task::Task<T>` is only the move-only coroutine return and pre-admission RAII owner; it has no public raw handle type, raw-frame constructor, frame observer, release, or destroy operation. Shared runtime failure identity is the root `rund::ReasonCode`; Host, Runtime evidence, and Replay values remain in their focused product entries. Scheduler operation tags, transient operation records, and promise-support names are not product concepts. | Direct-header, SDK, black-box, task, network, and server consumers compile and execute with the product namespaces. Installed negative compilation rejects implementation namespaces and headers, non-product task operation paths, public operation tags or records, public promise-support names, and raw Task frame capabilities. Focused frame lifetime proves the sole `Task -> Scheduler -> None` transfer, exact destruction on completion and failed admission, and repeated internal cleanup as a no-op. |
| socket lifetime | `net::Socket` is the sole move-only owner. A light borrowed `SocketView` carries identity but cannot close it. Open and accept transfer ownership once; explicit close consumes the owner, and destruction performs the one generation retirement and native close attempt. Retirement may wait for active operation leases, so destructor latency is not bounded. | Copy construction is ill-formed; move, stale-view, explicit-close, destructor, peer transfer, failure cleanup, and zero-allocation warm server contracts pass. No example authors a cleanup wrapper or repeats close branches. |
| Session shutdown | Ordinary reusable-session code calls blocking `close()` once. It rejects late admission, waits for an active scope, cancels/drains resident Compute work, and returns Stopped. Explicit `drain()` exists only to stop admission while intentionally observing Draining. | Running-to-Stopped, Draining-to-Stopped, gated active scope/job, concurrent and stopped idempotence, late rejection, and exact one-Draining/one-Stopped trace order pass. |
| Replay binding | One `Binding` owns checkpoint schema and restore when state continuation is enabled. It creates each `Channel` by combining one input id/schema with one borrowed source. The source returns the canonical sequence, so every scope reads through `channel.read(context)` without rebuilding the source or repeating sequence wiring. Record, Replay, and Scenario share that channel and execution owner without a mode branch. | Installed Record, Scenario, checkpoint continuation, and History journeys compile and run; source counts, returned-sequence identity, schema rejection, callback borrowing, lvalue lifetime, source-return, direct-checkpoint-execution, and no-sequence-read compile contracts pass. |
| actionable telemetry | Telemetry derives at most five typed findings for allocation, copied bytes, canonical graph-read bytes, source-selected queue pressure at its exact bound, and critical path. Session events and Standalone `Job::profile().findings()` share one compiled Stats projection and one finding decision owner. Each finding carries exact or explicitly unavailable/saturated evidence, a meaningful same-unit reference only when one exists, typed cause, and concrete action without changing the run result. A configured Sink is created only by `bind(lvalue, level)`; callback, context, and level storage are private. `describe` streams stable cost, cause, and action text through a borrowed writer without allocating or retaining presentation storage. A Compute work maximum with at least one submitted command, one kernel sample, and `submit_wait_ns > kernel_ns` reports `submit-overhead -> batch-jobs`; equality or missing timing/submit evidence retains the graph-work action without an inferred threshold. Accuracy, cause, action, and reference kind have one stable public text projection; diagnostics expose no enum numbers or private mapping. | Basic/Detail non-time parity, exact `telemetry:detail`, Profile/Event projection equality, configured-Sink construction rejection outside `bind`, no-observer-copy/no-allocation, allocation-free streaming description, complete tie-mask cause/action mapping including submit-overhead equality and missing-sample rejection, bounded finding count, one linked text authority, and the installed printed diagnostic journey pass. |
| scalar network I/O | Ordinary stream and datagram SocketView operations are allocation-free, move-only, single-await values. Stream verbs return public `net::{Receive,Send}` operations and await to `net::{ReceiveResult,SendResult}`; datagram verbs use the same split below `net::datagram`. No private readiness composition enters the return type. Shape admission runs once before readiness; each admitted operation then performs one readiness decision and transfers its one move-only `ready::Ticket` to the existing scalar consumer without a nested task or repeated shape check. Invalid spans, datagram capacity, or peers park zero registrations. Moving or awaiting consumes the operation view; moved-from and repeated operations fail before native I/O. The immediate path parks zero reactor registrations; the suspended path parks exactly one. Explicit tickets remain the advanced authority for many/set/timed, vectored, accept, connect, and bounded-drain coordination. Every `Ticket&&` operation first claims and invalidates the capability; zero-budget, empty, and invalid-shape terminals cannot leave a reusable ticket. The claim validates code and interest without a registry lookup. Only an admitted operation acquires the direct entry/generation lease, with no descriptor-to-entry resolution. Scalar consumers attempt native I/O at most once. Explicit read, write, and accept drains make no more native attempts than their declared budget and stop on would-block, error, callback stop, or the bound. Read/write drain has no direct `SocketView` overload. Generation is validated at registration and lease acquisition so close/reuse cannot cross the boundary. | The scalar first-success contracts add no child task, emit one `IoReady` then one matching byte or datagram event per operation, and record exactly one matching native call. Invalid ordinary shape parks and calls native I/O zero times. Moved-from and repeated ordinary operations produce `IoFdInvalid` without another native call. Lifecycle races fail closed, generation reuse cannot cross a ticket, packet order and replay evidence match, preflight rejection consumes its ticket while performing no lease or native attempt, accepted scalar send/receive increase their native-call counters by exactly one each, rejected tickets increase them by zero, the installed surface names each ordinary operation and terminal type, accepts `Ticket&&` drains, and rejects direct `SocketView` drains; accept-drain contracts prove `attempts <= max_accepts`, server traces distinguish one listener registration per batch from peer-I/O readiness observations, and the constant-queue-occupancy 1,024-success plus 1,024-rejection warm loop allocates zero runD heap storage and emits only 1,024 bytes. |
| edit/build loop | Focused headers include no unrelated implementation surface, templates own only type-dependent work, and focused link closures contain only the selected semantic owner. Basic Compute parses neither standard-future construction nor Pipeline binding templates; the async entry completes the same deferred Flow terminal on demand, and the focused Pipeline entry owns only typed binding expansion plus the compiled owner. Session lifecycle declares resident admission through incomplete types; only the Compute Session entry owns Job and Pipeline admission through the one `compute::{Request,Submission,Poll,Completion}` sequence, while Session configuration reaches only the narrow Compile resource owner. The nested `Request::Awaiter` and one-owner `compute/session/*` support hierarchy add no root awaiter or second public entry. Program convenience execution lives in `rund/compute/program/run.hpp`; shared ABI consumers import only the exact leaf they use under `rund/compute/abi/`. | `tools/measure/build/run` records current header dependency bytes, reverse object reachability, and cold/warm exact-route work from the live build graph, including the Pipeline header's cold/warm syntax cost and unrelated-consumer fan-out. Installed current-surface, focused Pipeline, and isolated Session translation units prove the public include boundaries; compiler contracts prove the public ownership boundary and invalid Compute bodies. |
| Compute compilation service | Compile workers and bounded queue capacity belong to the selected Device or Session resource envelope. Full admission is rejected before packaged-task/future allocation, and failed factories cancel their exact reservation. | Independent devices with different capacities, allocation-free exact full rejection, FIFO reservation order, exception-safe capacity recovery, same-key coalescing, drain, destruction, cache identity, and concurrency contracts pass. |
| failure exits | Every installed product failure returns that result's own `exit_code()`. Exit `2` is reserved for an application assertion after all product outcomes have succeeded. Foreign Kernel/backend diagnostics cross one operation-typed projection, so an unavailable selected backend cannot degrade to `ReasonInvalid` and a raw command failure cannot impersonate `DeviceBusy`. | The Compute boundary-projection table contract, selected-backend installed journey, and installed `exit/failure.cpp`/`exit/assertion.cpp` consumers require exact typed reasons and process exits `1` and `2`; either path using the other's code fails package acceptance. |

| Journey | Minimum completion | Measurable friction gate | Executable proof |
| --- | --- | --- | --- |
| SDK integrator | Find the installed SDK, include only the used domains, and choose linkage from the C++ declaration boundary. | Exactly one imported target; one focused header per used domain, the focused Pipeline entry when it is the only Compute need, plus the opt-in Compute async, math, and Session entries; every direct entry comes from the checked-in surface registry; the declaration-free umbrella is optional. `PRIVATE` keeps runD in the implementation and `PUBLIC` exposes a runD declaration to downstream code. | Independent installed-header translation units, umbrella reachability, `package.consumer`, `rund_package_private_consumer`, and `rund_package_public_consumer` with its downstream application. |
| Simulation author | Open one `Session`, declare one authoritative input, and run Record and Replay through one callback. | One Session open and one `close()` for `N` runs; one bound input and one callback; no caller-authored record/replay branch or per-mode graph. The reusable-session journey calls `close()` once even after an operation failure, returns that originating operation failure first, and returns a close failure only after the Replay operation succeeded. Source calls per canonical input occurrence are Record `1` and Replay `0`; every completion has one typed `replay::Code`, truth conversion, `error()`, and `exit_code()`. | The installed [`replay.cpp`](../tests/consumer/example/replay.cpp) first success plus the Record/Replay source-count assertions in `package.consumer`. |
| Thought-experiment author | Bind state schema and one lvalue restore codec once, apply explicit choices, and run a Scenario through the same callback. | `Binding{schema, restore}` is the sole state authority; `binding.checkpoint(record, bytes)` captures state and `binding.resume(checkpoint)` yields one `Resume`. Restore receives only immutable bytes; `Resume::record`, `run`, and `scenario` do not repeat schema or restore. Choices are formed only by `Channel::choice(sequence, bytes)`. Invalid or duplicate choices fail with typed codes before restore or simulation; Scenario source calls are `0`. | The installed [`checkpoint.cpp`](../tests/consumer/example/checkpoint.cpp) continuation and [`scenario.cpp`](../tests/consumer/example/scenario.cpp) choice journey plus focused Scenario rejection and the consolidated current-surface type contract. |
| Bounded server author | Open one listener, declare one accept bound, and pass one peer handler to `server::serve`. | One listener socket, one `server::Task`, and one typed `server::PeerResult` per handler; no caller-authored accept loop, handler-copy path, task-result nesting, out-of-band peer result, or mirrored counters. For `N` completed peers the server performs `N` handler invocations, exactly `N` terminal counter updates, and no runD heap allocation after configured scheduler and socket storage are warm. A ready peer handler remains owned by either the scheduler queue or one lane across temporary lane backpressure; it cannot disappear into `task_deadlock`. Sequential failure preserves its exact typed reason; parallel failure deterministically reports the lowest accepted-index non-success independent of completion order. | The compile-checked [`server.cpp`](../tests/consumer/example/server.cpp) is the short first-use surface. The installed [`contract/server.cpp`](../tests/consumer/contract/server.cpp) executes the loopback journey; allocation, ordering, move-only handler, exact-reason reduction, exact-counter, capacity, ready ownership, and scope-local progress assertions remain in the focused runtime contracts. |
| Reproduction operator | Append bounded segments to one `History`, locate a retained checkpoint, inspect an optional bounded raw ingress range, and request Diff or Window evidence without invoking the live source. | Retention is bounded by bytes and segment count; retained sequence lookup is `O(1)` over the contiguous suffix; `captures()` and every expected/actual Window context are borrowed spans; canonical input mismatch identifies id, schema, sequence, byte length, and hash; Replay and Scenario source calls are `0`; storage, capacity, and mismatch failures are typed. | The installed [`history.cpp`](../tests/consumer/example/history.cpp) append/evict/find/replay journey plus the black-box capture, oversized Segment, immutable survivor, Diff, and Window contracts. |
| Compute author | Declare one typed `Flow`, compile one `Program`, and reuse it with one explicit `Target` and bounded resources. | One graph language and one fingerprint owner; no backend-specific graph, hidden fallback, arbitrary failure text, or unbounded asynchronous admission. The ordered Program graph has its own public `graph::{NodeCapacity,ValueCapacity,BuffersPerNode,OutputCapacity}` envelope, independent of the per-Map expression/IR envelope. Synchronous and asynchronous compilation return the same typed `compute::Reason`; cache hits reuse the same canonical identity, and a full async queue rejects immediately with its capacity reason. | The installed [`compute.cpp`](../tests/consumer/example/compute.cpp) first map and [`program.cpp`](../tests/consumer/example/program.cpp) changed-input reuse journey, installed Compute service contracts, the maximum graph admission/rejection contracts, and a 1,536-node `Fixed<I,F>` witness for 32- and 64-bit storage with identical graph fingerprint and exact raw result on CPU, Metal, and Vulkan. |
| Device-program author | Compose arbitrary finite bounded data-plane work with the existing typed `Flow`: stable append/order, count-aware indexed load, bounded worklist iteration, deterministic conflict resolution, persistent published/pending state, and a flattened `Bounded<T,Count>` Program boundary. | Every capacity and iteration bound is frozen before compile; device-produced count is consumed by the next Program without host observation; oversized count/index failure is atomic; source-ordinal Fixed fold bits match CPU, Metal, and Vulkan; one `state(published, pending)` plus `commit()` publishes only complete ticks and `snapshot()` captures the published generation and immutable content hash. Two warm ticks leave the retained-memory snapshot exact-stable and report zero compilation, Buffer/descriptor allocation, host upload, host download event, and host downloaded bytes; producer-to-consumer resident bytes remain accounted as roundtrip evidence and are not host traffic. The CPU installed route also observes zero process allocation. The compile-time authoring callback count cannot change during either tick. No public Kernel/shader/native callback vocabulary exists. | The installed [`device/program.cpp`](../tests/consumer/example/device/program.cpp) is the ordered journey runner; its `device/program/` leaves separately own the shared model, failure recovery, profile validation, step attribution, and tick execution. Together they execute the 1D simulation normal form spatial update → stable broadphase order/candidate emit → narrowphase filter → island key/deterministic resolve → bounded solver → integration → state hash/compact events through two committed generations. They require every available CPU/Metal/Vulkan target to report nonzero device-generated and indirect execution evidence while exact-matching the semantic state bytes, snapshot/state hash, event values/counts, conflicts, and iterations through `runD::sdk`; backend-specific physical execution totals remain transparent rather than normalized. Kernel adversarial Fixed/reference tests, focused CPU/Metal/Vulkan bounded primitive contracts, and the Pipeline restore contract own numeric, failure, and restart parity. |
| Pipeline author | Declare dependent Program steps once, bind resident Buffers in declaration order, prepare once, and reuse one move-only Pipeline through standalone or Session execution. | One fluent `then(program, read(...), write(...))` step path; recurrence makes output intent explicit with terminal-only `write_final(...)` or lossless caller-owned `write_each(...)`; bounded windows accept `write_final(...)`; actual CPU intervention is a separate `host_feedback(pipeline, count, callback)` prefix-commit boundary. No dependency handles, graph unrolling, DAG, reorder, implicit transfer, or second Session result vocabulary. Omitting host feedback preserves the one-submit resident GPU path; selecting it exposes one run/completion per host iteration. | The installed [`pipeline.cpp`](../tests/consumer/example/pipeline.cpp), `tools/test/run compute.pipeline`, `tools/test/run compute.window`, `tools/test/run runtime.compute-pipeline`, accelerator Session coverage, installed release consumer, and Pipeline build/compute measurements. |
| Performance operator | Inspect `Job::profile().findings()` directly for Standalone Compute, or bind one lvalue Session observer; request Detail only while diagnosing a Session critical path. | Profile and Event findings return at most five values from the same compiled raw-counter projection without another counter owner. `telemetry::bind` is the sole configured-Sink construction path, retains no callable copy, and allocates no storage. `telemetry::describe` streams the remediation text without an owned buffer. Basic and Detail preserve typed codes, ordering, hashes, counters, and every non-time finding. A measured submission-dominated work phase names `batch-jobs`; graph-dominated work names `reduce-graph-bound`. The only detail selector is [`telemetry:detail`](../../node/docs/contracts/telemetry.md#user-contract). | The installed [`telemetry.cpp`](../tests/consumer/example/telemetry.cpp) proves the first observer/findings journey. `tools/test/run compute.telemetry` and `tools/test/run telemetry:detail` own projection and parity verification; overhead follows the [telemetry measurement method](../../docs/reference/performance/method.md#telemetry-overhead-method). |

Convenience may remove bookkeeping only when it preserves these owners and
bounds. A source-level test is useful development evidence, but it cannot close
an installed-user journey by itself.

This page owns durable acceptance criteria, not mutable release status. The
frozen-manifest evidence packet records PASS or RED for a candidate so a stale
snapshot cannot become a second product contract.

## Cost Gate

Let `L` be Session preparation cost and `W_i` the useful cost of scope `i`.
Rebuilding a Session for `N` scopes costs

```text
N * L + sum(W_i)
```

One reusable Session costs

```text
L + sum(W_i).
```

The reusable path must therefore remove exactly `(N - 1) * L` units of
preparation work. This is a structural bound, not a speedup claim. Release
evidence follows the single
[telemetry measurement method](../../docs/reference/performance/method.md#telemetry-overhead-method).

Basic observation must not require high-resolution timing on the hot path.
`telemetry:detail` may add timing work, but it must retain the same semantic
result, ordering, hashes, and stable counter values. Its overhead is reported
through the linked method rather than hidden.

## Release Proof

Acceptance requires every evidence cell above to pass on one source manifest
and installed artifact. A rejected operation must expose the asserted stable
typed code, its derived error text, and a nonzero exit code. Backend selection
executes the explicitly selected backend or returns its typed failure.

A documentation example is not proof by itself. Every official `cpp compile`
fence must compile against the installed SDK, every `run` fence must execute,
the canonical example must be byte-identical to its package source, and every
local documentation path and heading anchor must resolve. Missing or unrun
evidence keeps the release gate open.

## Semantic Owners

- [Session](../../node/docs/contracts/runtime.md) owns lifecycle, repeated
  scopes, completion, and Session telemetry.
- [Replay](../../node/docs/contracts/replay.md) owns the one canonical input
  boundary, strict substitution, checkpoints, and retention.
- [Telemetry](../../node/docs/contracts/telemetry.md) owns level selection,
  event shape, emission order, and default/detail parity.
- [Compute](../../docs/reference/compute.md) owns typed graphs, explicit
  backend selection, resident reuse, and Compute telemetry.
- [SDK Consumption](./consumption.md) owns the `PRIVATE`/`PUBLIC` CMake
  boundary and black-box user path.
