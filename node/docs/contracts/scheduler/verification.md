# Scheduler Verification

This page owns native task scheduler verification routing. Package consumption
is owned by [`/package`](../../../../package/README.md).

## Authority

- `/node/tests/contract/cases.def` and `/node/tests/contract/cases/`
- `/node/tests/contract/runtime/task/coroutine/`
- `/node/tests/contract/runtime/task/net/`
- `/node/tests/contract/runtime/task/replay/`
- `/node/tests/contract/runtime/product/`
- `/node/tests/contract/runtime/stress.cpp`

## Commands

Use `tools/test/run <case>` for a single semantic case and `tools/check/run` for all
registered groups. Performance measurements are explicit investigations, not
permanent contract-test routers. Public/internal boundaries are checked by the
installed-package consumer.

Each grouped CTest starts its owner executable once and passes the complete
ordered case list. The shared runner reports the active case, stops at the first
nonzero result, and returns that result to CTest. Exact-case execution uses the
same runner with one name. Contract cases must release owned state before
returning; process restart is not a cleanup authority.

The `.def` registry is the single authority for case names and function
symbols. The per-target thin table derives declarations and lookup rows from
that registry; the target-neutral runner consumes only the canonical
`Case`/span ABI and never the case macro. There is no manual declaration mirror
or static registration. Grouped verification retains
all owner sources. An exact Node configuration retains only the source that
defines the selected registry symbol plus support sources that define no
registered case inside the selected suite partition. It registers only that
Node case and target, so sibling case translation units do not enter the exact
build graph. Every native task, network, and replay case owns a closed
suite-to-source partition. Core Runtime, task, host, network, reactor, and
replay partitions form the single `node-runtime` build target rather than six
duplicate links over `RUNTIME_BASE`. Their six grouped CTest rows remain
separate process owners with the same ordered case lists and failure isolation.

Explicit source lists admit Node contract translation units. A
validation-only configure-dependent glob proves that every live contract
`.cpp`/`.mm` file, including the runner and dispatch table, appears in those
lists and no listed file is absent. Case-owner source reads are configure dependencies too,
so source additions, removals, and owner moves regenerate before an exact
target is built.

Core Runtime, CPU Compute hosting, and accelerator Compute hosting have
separate source sets and executables. Each focused build contains only its
semantic translation units, while the CTest process count and semantic routes
are fixed. The root suite registry assigns the three owner fragments directly
to their Runtime groups; there is no parallel Runtime case router.

Core Runtime and every task, host, network, reactor, and replay row select the
`runtime` link profile in that same registry. Their exact and broad targets
build the Runtime base closure: host, replay, scheduler, selected platform,
Core, and Kernel. Public Compute and Accel execution objects are absent.
Runtime Compute rows select `product`, so installed-product integration still
closes over every Node component.

The Node route separates CPU-only Compute cases from cases that open Metal or
Vulkan. Only the latter share the accelerator resource lock; CPU work
can progress while another owner holds the device route. One all-backend case
is the authority for each semantic matrix, with backend-specific cases reserved
for distinct backend-only contracts.

`runtime.task.basic` primes one leaf slot, then counts process-global
allocations from warm `spawn` through `join`; the required count is zero.
It also primes a four-lane 64-task segment and requires zero process-global
allocations for the next complete spawn, parallel execution, deterministic
commit, and join batch.
`runtime.task.ready-queue` drives a 4,096-leaf four-lane segment and compares
one-worker and four-worker failure priority when an earlier leaf enters a
scheduler primitive and a later leaf throws. This proves that the lane commit
frontier permits pure parallel execution without changing canonical failure
selection. `runtime.task.env` admits host-event-producing leaves across eight
lanes and proves every event is retained through the ordered primitive
boundary. `runtime.task.lane` separately proves the primitive side-exit path
remains allocation-free and terminates without a lane/commit deadlock.
`runtime.task.hash` is the compile-time and runtime golden oracle for operation,
observation, and host-event domain scalars, field order, signed-width
conversion, and 64-bit wraparound. Its operation fixture consumes scalar
arguments directly, so reintroducing an aggregate operation record is not a
test dependency or alternate production authority.
`runtime.task.ready-queue` compares one and four workers for a batch containing
different failure reasons and requires the same lowest logical task failure;
it also verifies mixed leaf/coroutine batches leave coroutine suspension on the
individual quantum path.
`runtime.task.coroutine-frame` and `runtime.task.result` separately prove arena
and completion-cell reuse so a regression can be localized to its owner. The
result owner also proves zero warm allocations for value publication and
direct rejection, canonical FIFO waiter wake order after an `O(1)` middle-node
cancellation, immutable first-terminal reason selection, producer/observer
lifetime, type checking, retained-page growth, and stale-generation rejection.
The implementation proof is structural as well: every waiter has explicit
previous/next links, so cancellation touches at most two neighbors, while
terminal publication alone traverses all `W` detached waiters in two bounded
linear passes: one order/state pass and one callback pass. Generation
increment is saturating by retirement rather than modular, so the equality
predicate used by handles cannot alias after `2^32` admissions of one cell.
The
coroutine lifecycle suite owns that same allocation probe because its channel
construction rows measure warm allocation counts; an exact lifecycle link is
therefore closed without borrowing a sibling suite.
Nested typed-result reuse, leaf-primitive rejection, and discarded-operation
non-admission are part of the same `runtime.task.coroutine` lifecycle owner.
They do not retain separate registry cases or translation units: each uses a
fresh `rund::run` scope, so merging them preserves isolation at the product
boundary while removing two dispatcher entries and two compile units.

`runtime.task.reactor-registry` is the focused single-authority and warm
storage proof. It checks nonzero wait-id admission, canonical wait-id order,
per-fd slot linkage, exact aggregate interest, deterministic successful-prefix
removal, canonical first-occurrence pre-removal interest rows, and stable
arena/order/free/fd storage addresses and capacities across add/remove churn.
The surrounding
reactor group proves persistent registration, deferred re-arm, fd generation,
batch apply/drain, backend parity, budget/backlog, cancellation, cleanup,
lifecycle, ordering, capacity, and scratch reuse through the same registry
owner.

Timed readiness verification asserts the exact `IoTimedOut` reason after a
parked single wait. ReadyMany boundary verification separately asserts the
same exact reason after both immediate zero-timeout and positive-duration
parked completion. These rows prove that public timeout projection and both
resume paths use the canonical reactor result rather than mirrored timeout
state.

Accel verification has one internal graph-factory smoke owner, one CPU SIMD
owner, and one device owner. This keeps CPU/device scheduling independent while
avoiding per-category executable links and process startup. The surface row
selects `header`, meaning no Node execution SCC; its target links only the exact
Kernel Compute view that owns compiled graph-signature construction. CPU and
device rows select `accel`. Backend selection execution belongs to those
stronger CPU/device rows instead of being repeated by the factory smoke. Their
broad and exact targets therefore link only the production closure required by
their assertions; CPU/device distinction remains in the group, assertions, and
accelerator resource lock.

## Unavailable Platform Contract

`runtime.platform-adapter` is the focused product-link and runtime owner for a
host without a native reactor/IO/network adapter. It checks the complete
selected native function table, stateless reactor preparation, precise
`IoUnsupported` and `ReactorBackendUnavailable` mapping, ordinary portable
scheduler execution, and the fixed-width public file mode. The one public
verification command is:

```sh
tools/check/platform/unavailable
```

The force option changes only selected platform ownership; it does not define
a success-masquerading test stub. A non-Unix build host selects the same source
owner through `NOT UNIX` without the force option. Cross-configuration compile
agents additionally build `node-object-platform`, `node-object-host`,
`node-object-runtime-base`, `node-object-scheduler-hostio`, and
`node-object-scheduler-reactor` with their real non-Unix toolchain.

Vulkan discovery is an independent configuration dimension. The command owns a
fresh `.cache/platform-unavailable` tree, configures OFF and executes the exact
runtime owner, toggles that tree ON and compiles `node-object-accel-vulkan` plus
`node-object-platform`, then toggles it back OFF, rebuilds both owners, and
executes the exact runtime owner again. ON validates source and object ownership
without opening a GPU. Both OFF phases reject Vulkan SDK, glslang, and SPIR-V
compile definitions, so cached ON discovery cannot masquerade as a clean OFF
graph. The audit derives the expected source sets from the two canonical CMake
source fragments rather than maintaining a second list. It also requires one
exact `runtime/platform/adapter.cpp` test object carrying
`RUND_NODE_PLATFORM_UNAVAILABLE=1`, then proves that object was built; the
guarded unavailable semantic body therefore cannot silently compile out. Every
subprocess is bounded, and before/after source manifests plus the phase audit are recorded in
`.cache/evidence/platform-unavailable/`. Shader process tools are selected only
on hosts with the implemented process adapter; other hosts reject Vulkan
selection as tool unavailable rather than compiling POSIX process headers.
