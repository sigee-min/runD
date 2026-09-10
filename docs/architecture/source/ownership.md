# Source Ownership

Source layout follows semantic ownership, not measured line or include counts.
A file should own one behavior, state transition, or integration boundary. A
split is useful only when the new files have independent responsibilities;
routers, anchor functions, and forwarding layers do not establish ownership.

Public interfaces and private implementations remain separate. Private sibling
declarations may use a local header, but local headers are not package surface.
Dependency direction is enforced by compilable interfaces, link boundaries,
package consumers, and runtime contracts.

Large files are reviewed by meaning. A change should split an owner when it
mixes unrelated lifecycle, scheduling, validation, execution, or evidence
decisions.

## Path Meaning

Code leaves name one direct concept. When two independent concepts need to
identify an owner, their order is expressed by directories rather than by
concatenating the words or appending a width or phase number. Established
domain tokens such as `checkpoint`, `overflow`, and `u64` remain valid. New
names receive semantic review in the owning subsystem; a byte-identity tool
does not infer natural-language boundaries.

## Header Dependency Ownership

Internal headers include the narrow header that owns every value in their
interface. A type aggregate is not an implementation dependency: a consumer of
`ComputePlan`, for example, includes its type owner rather than the planning,
lowering, reference, and validation aggregate. Complete by-value members,
arrays, and template element types use their real definitions; forward
declarations may not stand in for an owning include.

A scheduler-wide state header owns stored runtime state, not every public
operation value named by a member function. Public request, result, budget, and
configuration values that appear only in function declarations are forward
declared; the operation owner includes their definitions. In particular, the
network many-readiness surface is not an edge of the scheduler state storage
graph. A readiness API edit therefore dirties only the reactor and host-I/O
closure, not unrelated lanes, channels, or task execution owners.

This rule makes invalidation semantic. For the include graph `G = (V, E)`, a
changed header `h` dirties the translation units in its reverse-reachable set
`R(h)`. An aggregate edge makes that set the union of every imported owner's
set. Leaf-owner edges limit it to
`R(h) = union(R(o) for o in actual owners used by h)`. Rebuild reduction claims
must compare those exact dirty sets, and timing claims require the same source
manifest and build route; the graph law alone is not performance evidence.

## Failure Boundary Ownership

Public failures use the owning subsystem's typed vocabulary; a C++ exception
class is never a second public error authority. One private boundary may
project exception types into that vocabulary only after the full lifetime and
rollback owner is in scope. Result construction and semantic reason selection
stay at that boundary, while aggregate mutation rollback stays with one RAII
transaction owned by the mutated state.

Exception classification may be shared within one semantic subsystem, but it
does not own cleanup and may not erase differences between allocation,
descriptor, transport, cancellation, or invariant failures. An inner catch is
admitted only when it must restore state before propagation, crosses a
`noexcept` callback, or selects a meaning that differs from the outer boundary.
Otherwise exceptions unwind to the one owning boundary. Catch-all conversion
requires an explicit unexpected-exception contract; it is not a substitute for
enumerating the exception classes that the boundary is authorized to project.

## Measurement source ownership

The current-source Virtual crossover diagnostic has one authority per
responsibility:

- `tools/measure/compute/virtual/crossover/schema.hpp` owns the frozen
  value-only grid, ratios, capacities, and cell ordering shared by production
  and aggregation.
- `tools/measure/compute/virtual/crossover/prepare.cpp` owns paired workload
  preparation, backing setup, and deterministic input seeding.
- `tools/measure/compute/virtual/crossover/evidence.cpp` owns output/hash,
  cold-terminal, Profile, and warm-sample evidence validation.
- `tools/measure/compute/virtual/crossover/run.cpp` owns ABBA sampling and
  ordered measurement orchestration; `report.cpp` owns measurement-row
  formatting and the frozen CSV field order.
- `tools/measure/compute/virtual/crossover/aggregate.cpp` owns CSV column
  emission, packet parsing, schema validation, consensus, bracket, and slice
  reporting.
- `tools/measure/compute/virtual/crossover.hpp` owns only the public
  measurement entry points consumed by the CLI.

The producer and aggregator communicate through the serialized CSV contract;
neither includes the other's implementation or duplicates its parsing or
workload logic.

The natural route-matrix producer has the same single-owner rule:
`virtual/route_matrix/run/prepare.cpp` owns case geometry, admission, and
backing initialization; `run/sampling.cpp` owns timed execution and the fixed
ABBA sample cohort; `run/evidence.cpp` owns terminal/profile/output/hash and
route evidence validation; and `run.cpp` owns device/case ordering and the
substantive coordinator. The observer is split without a second accumulator:
`observer/evidence.cpp` owns lifecycle/hash/capability and counter
accumulation, `observer/hooks.cpp` owns the two delegating DeviceOps hooks, and
`observer/lifecycle.cpp` owns installation/restoration/reset/sealing;
`observer/local.hpp` is a declaration/type seam only. The route `oracle.cpp`
and `report/` owners retain their predicate and serialized packet authorities;
`internal.hpp` remains a declaration/value seam only.

The current-source preparation-memory diagnostic has one authority per
responsibility:

- `tools/measure/compute/pipeline/prepare/model.hpp` owns the frozen workload
  constants and observation schema.
- `tools/measure/compute/pipeline/prepare/contract.cpp` owns plan, memory,
  backend-reservation, telemetry, and failure-location contract validation.
- `tools/measure/compute/pipeline/prepare/program.hpp` owns synthetic program
  construction and no observation or CSV policy.
- `tools/measure/compute/pipeline/prepare/report.cpp` owns CSV columns,
  serialization, and memory-category naming.
- `tools/measure/compute/pipeline/prepare/run.cpp` owns device opening,
  program/buffer setup, plan/prepare orchestration, and observation sequencing.

The public `pipeline.hpp` declarations remain the CLI measurement boundary;
the preparation-memory producer, contract, and reporter communicate through
the one observation schema without implementation includes or forwarding
wrappers.

The shared Compute measurement suite has one owner per cross-scenario concern:
`suite/core.hpp` exposes only common CLI declarations and timing constants;
`suite/output.cpp` owns CSV escaping, environment rows, and workload columns;
`suite/reference.cpp` owns the process-wide hash ledger and reference checks;
`suite/capture.hpp` and `suite/bench.hpp` retain only the genuinely generic
resident-job templates; `suite/bulk.hpp` owns the compile-time bulk cost
model; and `suite/warm.hpp` owns the warm-counter arithmetic contract. The
executable CMake source lists include the compiled output, parser, and
reference owners explicitly, so no translation unit creates a second inline
reference or output authority.
