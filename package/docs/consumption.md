# SDK Consumption

An engine, simulation, service, or tool consumes runD as a black-box SDK.
The runD source tree remains private; callers use the installed package
artifact through CMake package discovery.

Choose link visibility from the application's C++ boundary:

```cmake
find_package(runD 1.0.5 EXACT CONFIG REQUIRED)

# runD appears only in this target's implementation.
target_link_libraries(engine PRIVATE runD::sdk)

# engine's public headers expose runD values or templates.
target_link_libraries(engine_sdk PUBLIC runD::sdk)
```

`runD::sdk` is the only link and include authority in both cases. `PUBLIC`
does not expose a second runD target: it propagates the same exact artifact,
C++20 requirement, deterministic floating-point options, transitive headers,
and link closure to consumers of `engine_sdk`.

Forbidden consumption paths:

- `add_subdirectory(runD)`
- vendored runD source directories
- runD as a submodule
- source-tree include paths

## C++ Boundary

A library that wants to keep runD out of its own ABI links `PRIVATE` and keeps
runD includes in implementation files. A library may instead expose the
documented SDK values or templates from a registered direct header; it
then links `PUBLIC` so CMake propagates the exact SDK requirements to every
consumer. The visibility must match the C++ declaration boundary. Hiding a
runD type in a public declaration while linking `PRIVATE` is invalid because
downstream translation units cannot inherit the required include, compile,
and link contract.

Only names documented by the installed surface may cross that boundary.
Source-private Kernel, Accel, and direct Node owner headers remain forbidden;
link visibility does not promote them. The exact direct-header and transitive
support rules are owned by [SDK Surface](./surface.md).

The installed consumer proves both shapes: an implementation-only static
library linked `PRIVATE`, and an engine library whose public header returns a
runD result linked `PUBLIC`. The final application in the second proof links
only the engine library, so any missing transitive include, compile feature,
numeric option, or archive fails at the actual downstream boundary.

## User Path

The product path is one installed target and one focused header per domain. A
consumer must not assemble subsystem libraries, include source-owner headers,
or reproduce runD lifecycle, replay, backend, or telemetry policy in an
adapter. Each runD domain begins with its matching focused header. Compute
consumers add `<rund/compute/async.hpp>` only for `compile_async()`, add
`<rund/compute/math.hpp>` only for composite functions or matrix, transform,
factor, solve, and spectrum stages, and add
`<rund/compute/pipeline.hpp>` only for prepared dependent-Program execution;
the basic `<rund/compute.hpp>` entry deliberately excludes Pipeline.
Add `<rund/compute/session.hpp>` only when submitting a resident Job or
Pipeline through a Session; that entry supplies `Request`, `Submission`,
`Poll`, and `Completion`
under `rund::compute`, while its `compute/session/*` support leaves remain
transitive rather than separate consumer entries;
`<rund/rund.hpp>` is the declaration-free convenience composition when one
translation unit intentionally consumes all seven. Transitive support headers
are not separate application entry points.

Do not include the umbrella beside the focused entries it already composes,
and do not use it as the default for a single-domain file. That combination
adds no declaration or linkage capability, expands the translation unit's
dependency closure to every runD domain, and makes unrelated header changes
eligible to rebuild it. The installed consumer therefore proves each direct
header independently instead of compiling a second all-headers fixture.

Application code owns domain meaning and binds it once at the narrowest
authoritative boundary. runD owns the mechanism after that binding:

| User intent | Application declares | runD guarantees |
| --- | --- | --- |
| Execute repeatedly | session configuration and callback | one reusable lifecycle and bounded scope admission |
| Reproduce a run | canonical input source/schema and state codec | record order, hashes, strict replay, checkpoint lineage, and bounded retention |
| Test a choice | replacement input bytes | the same downstream callback and zero live-source fallback |
| Diagnose ingress | bounded raw byte and record windows | one borrowed `captures()` range plus paired Window spans that never replace canonical input |
| Observe cost | one telemetry sink | stable counts, typed codes, and derived errors; optional timing below `telemetry:detail` |
| Select compute | one explicit `Target` | that exact target or a typed failure, never fallback |

Every convenience follows the same rule: it may remove caller bookkeeping,
but it may not create a second source of identity, length, order, result truth,
or lifecycle state. In particular, replay input length is inferred from the
produced or retained bytes; it is evidence, not a caller-authored lookup key.
The installed consumer follows the same physical rule: large end-to-end
journeys have a small ordered runner and semantic leaves below a matching
folder. A leaf owns one user behavior, while one adjacent model owns shared
fixture state. The runner fixes journey order, and a leaf edit rebuilds only
that translation unit plus the final link.

The current proof owners are:

- `tests/consumer/blackbox.cpp` orders Replay, numeric evidence, Cluster, and
  network leaves under `blackbox/`;
- `tests/consumer/example/device/program.cpp` orders failure, profile,
  attribution, and tick-execution leaves under `device/program/`;
- `tests/consumer/compute/flow/primitives.cpp` orders the installed Flow
  primitive families under `flow/primitives/`, while `surface` owns the
  compile-time type checks.

The detailed behavior remains owned by the
[Runtime](../../node/docs/contracts/runtime.md),
[Replay](../../node/docs/contracts/replay.md), and
[Compute](../../docs/reference/compute.md) plus
[Pipeline](../../node/docs/contracts/compute/pipeline.md) contracts rather than being
restated here.

## Domain Ownership

runD remains meaning-neutral. The embedding application owns sessions, client
identities, packet schemas, protocol policy, ticks, rollback, matchmaking,
simulation state, and domain interpretation.
