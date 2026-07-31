# Accel Internal API

`/accel/include/accel` contains backend-neutral values and graph factories used
by the compiled Node compute bridge. It is not a direct SDK include surface.
The consumer entry is `<rund/compute.hpp>`.

Each internal consumer includes the leaf that owns the value or operation it
uses. Accel has no all-types or all-policy aggregate header: value owners are
split across `api.hpp`, `check.hpp`, `device.hpp`, `buffer.hpp`, `runtime.hpp`,
`context/*`, `graph/*`, and `kernel/*`; advisory policy consumers select the
required leaves under `run/policy/*`.

Compound owners use the path hierarchy directly. Buffer shape, graph buffer
reference, run binding, and segmented-reduce factory authority live at
`context/buffer/descriptor.hpp`, `graph/buffer/ref.hpp`,
`kernel/run/binding.hpp`, and `graph/factory/primitive/segmented/reduce/*`.

## Authority

- Kernel owns Compute IR, primitive descriptors, validation, identities,
  planning, lowering, and deterministic reference laws.
- Accel owns backend-neutral graph-node assembly and run/evidence value shapes.
- Node owns device observation, backend selection, contexts, resident buffers,
  transfers, compilation, submission, and backend diagnostics.
- `rund::compute` owns the typed public device, buffer, program, graph, run,
  status, and statistics vocabulary.

Internal graph factories attach Kernel descriptor hashes and edge signatures;
they do not create a second descriptor authority. Backend-specific limits,
shader source, pass geometry, pipeline caches, scratch storage, timestamps, and
SDK types remain private Node facts.

Graph factories accept Kernel-owned primitive descriptors directly. They do
not re-export Kernel operation, shape, status, layout, normalization, graph
kind, or complete descriptor types at the `rund` root. Those values keep their
one `rund::kernel` owner; internal callers name that owner directly. A factory
header owns graph-node assembly only.

Each operation's `.../<domain>/node.hpp` leaf attaches that domain's Kernel
hash, plan-derived signature, descriptor, and element extent. There is no
aggregate node implementation, hash adapter, or transitive all-plans include.
Callers that need one operation include one leaf; the umbrella is only an
explicit all-factories convenience surface.

The checked internal expression DSL lives at `rund::compute_dsl`. It is a
translation authority for Kernel tests and the compiled public bridge, not a
consumer namespace. New SDK operations are added to the short typed
`rund::compute` frontend and lowered in compiled sources.
