# Accel Window Contract

Window is the semantic adapter for one-dimensional affine Sum, Min, and Max
queries. Kernel owns `WindowDesc`, `PlanWindow`, identity, numeric policy, and
the deterministic reference. Node owns resident admission and execution over
the common source-private Range substrate.

## Authority

Implementation authority:

- `/kernel/include/kernel/program/compute/window/`
- `/node/src/accel/window/shape.{hpp,cpp}`
- `/node/src/accel/graph/admission/window.cpp`
- `/node/src/accel/kernel/bindings/range.{hpp,cpp}`
- `/node/src/accel/cpu/window/run.cpp`
- `/node/src/accel/metal/window/`
- `/node/src/accel/vulkan/window/`
- `/node/include/rund/compute/flow/stage/{window,pool}.hpp`
- `/node/src/compute/cpu/run/primitive/algebra.cpp`
- `/node/src/compute/cpu/scratch.{hpp,cpp}`

Verification authority:

- `/kernel/tests/contract/program/compute/window/`
- `/node/tests/contract/accel/kernel/cpu/window.cpp`
- `/node/tests/contract/accel/kernel/window.cpp`
- `/node/tests/contract/compute/collective/modes/core.cpp`
- `/node/tests/contract/compute/collective/modes/bounded.cpp`
- `/node/tests/contract/compute/memory/scratch.cpp`

## Semantic and graph law

For input count `N`, authored output count `Q`, window size `K`, stride `S`,
and left padding `P`, output `j` combines the `K` logical positions beginning
at `jS-P`. Clamp repeats the nearest endpoint; Clip excludes positions outside
`[0,N)`, equivalently combining the operation identity. Validation requires
`K,S>0`, `P<K`, matching element/domain width, checked input/output byte
counts, and an intersecting final window. `K>N` is valid. A graph node carries
exactly two bindings in `(read input, write output)` order; their shapes are
`N` and `Q`, their byte spans must not overlap, and the graph domain and fixed
format carried by the graph root do not override the descriptor. The adapter
derives the descriptor from the current value, and the descriptor's complete
numeric policy is authenticated by its semantic hash. This permits a Window
to consume a stored format produced by an earlier Map without confusing that
format with the graph root's numeric header.

Pool is the Exact-stage strided adapter. It validates its public width, stride,
tail, and edge law, derives one affine Window descriptor, and then uses the
same Range plan and execution owners.

Kernel accepts the canonical zero-work Sum descriptor with `N=Q=0` so the
semantic ABI has an empty value. Low-level resident Accel graph admission
requires nonzero buffers and rejects that descriptor. The product Compute
surface elides an empty Window before low-level admission.

Integer Sum wraps at the selected lane width. Fixed Sum applies the declared
overflow policy after every addition. `Wrap` projects to modulo-width Range
algebra, including deterministic approximation policy, and may acquire
PrefixDifference. `Saturate` projects to the non-invertible saturating law and
may use only a semantic-order-preserving legal path. Min and Max use signed,
unsigned, or fixed ordering from the frozen domain.

## Physical execution

Window admission projects the validated semantic plan into one affine
`RangeShape` and freezes exactly one `RangePlan` from the selected backend
capability. It does not recompute `Q` or select a second candidate. Stencil and
Window share the exclusive physical `RangeBinds`; their wrappers retain only
semantic validation, resident lookup, and public error/result translation.

The standalone Accel CPU backend intentionally uses `RangeCaps::cpu_reference`
and consumes a frozen Direct plan as the deterministic meaning oracle. The
product Compute CPU route uses the planned CPU Range execution and the existing
prepared CPU scratch owner for linear PrefixDifference or BlockPrefixSuffix
work. These are separate execution surfaces, not fallback from one to the
other.

Metal and Vulkan wrappers prepare and run the generic multi-pass Range
executor. Pipeline/source identity, dispatch topology, shared allocation, and
typed temporary requirements come from the frozen plan. Global temporaries are
placed by the existing Pipeline scratch authority; Window owns no arena or
parallel scratch vector. After prepared Pipeline memory is retained, warm
resident execution performs no allocation.

A Bounded product Window lowers to one Window node with an ordinary resident
logical-count binding. Its authored input and output counts are the common
capacity `M`; the generic Range control stage turns the current scalar `n` into
the exact active stage parameters and dispatch evidence. Vulkan and standalone
Metal consume the resulting indirect topology. Metal Pipeline-private capture
uses the frozen capacity grids with those runtime parameters because its ICB
contract has no indirect-dispatch command. The output preserves the same
logical count lineage. No per-radius Gather graph, host count readback, or
Window-private scratch owner participates in this route.

The contracts compare CPU, Metal, and Vulkan output with the kernel reference,
exercise a multi-stage affine Window, verify exact dispatch evidence, and keep
backend-unavailable builds behind their normal capability gate.
