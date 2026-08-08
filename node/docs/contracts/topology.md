# Topology Evidence Contract

## Scope

This page owns node topology evidence before it is projected into kernel inputs.

Public authority:

- `/node/include/rund/session/config.hpp`
- `/node/include/rund/session/resources.hpp`
- `/node/include/rund/session.hpp`

Implementation authority:

- `/node/src/runtime/backend.cpp`
- `/node/src/runtime/resource/discovery.cpp`
- `/node/src/runtime/resource/discovery.hpp`
- `/node/src/runtime/runtime/config.cpp`

Verification authority:

- `/node/tests/contract/runtime/topology.cpp`
- `/node/tests/contract/runtime/product/kernel.cpp`

## Contract

Node topology evidence has explicit truth levels:

- `Unknown`
- `Hint`
- `Verified`

Required worker-capacity truth admits exactly when the truth level is
`Verified`, the capacity vector has the requested worker width, and every
capacity is positive. In symbols, for requested width `W` and capacities
`c`, admission requires `truth = Verified`, `|c| = W`, and
`forall i in [0,W): c[i] > 0`. Required affinity admits exactly when its truth
level is `Verified`. These laws are private to the admission owner; the public
headers expose evidence values, not parallel predicate functions. Unknown and
hint-only topology must remain unavailable or hint-only, or cause admission to
fail closed when truth is required.

`resource::Resolve` is the one private describe-and-admit transaction used by
Session configuration; the projection and admission helpers are TU-local
details. It returns the admitted `ResourceEnvelope` directly. The embedded
`Resources::code` is the sole outcome authority; there is no outer result code
that can diverge from the observed resource value. The selected backend's
requested width is also the sole worker-width authority. The admission request
adds only required evidence levels and does not mirror that width. There is no
standalone discovery API or second backend owner. Runtime describes and admits
the exact backend already owned by its configured Scheduler; it must not
allocate a placeholder backend, describe it, and later replace it with the real
backend.

The public Session selects only a worker width. A Kernel worker backend is not
a Session value, and the built-in Session worker pool is the sole product
worker owner. The private selector overload exists only to inject topology
evidence into this contract's Node-owned tests; it is not installed, forwarded,
mirrored, or accepted by the SDK.

Resource envelopes own worker-capacity evidence. Admission must not preserve a
borrowed caller view whose lifetime can end before the admitted run envelope is
consumed.

NUMA is node-local resource topology. It is not kernel semantic meaning.

## Update Rules

- Truth-level changes or new topology evidence fields must update this page and
  the Runtime topology contract. Session admission, Kernel integration, and
  topology contracts must also change when selected-resource evidence or
  invalid-envelope generation changes.
