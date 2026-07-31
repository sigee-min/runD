# Numeric

Scope: cross-layer numeric policy and the public `evidence::Contract` model for
user-authored deterministic work. This page owns numeric authority routing,
invalid contract combinations, preset UX, serialization/hash identity, and the
bridge from admitted numeric proof to kernel-owned reduction inputs. It does
not own arithmetic implementation law.

runD is not `float`-first, `u32`-value-first, or fixed-point-only. Numeric
meaning is explicit policy chosen by the API or contract that admits work.

## Authority Path

1. [`/docs/README.md`](../README.md) owns repository-wide documentation rules.
2. This page owns cross-layer numeric policy and public `evidence::Contract`
   shape.
3. [`/math32/docs/README.md`](../../math32/docs/README.md) and
   [`/math64/docs/README.md`](../../math64/docs/README.md) own public
   deterministic vector-first integer and fixed-point arithmetic law.
4. [`/kernel/docs/contracts/reduction.md`](../../kernel/docs/contracts/reduction.md)
   owns reduction-local `FoldValueDomain` and strict FP reduction behavior.
5. [`/kernel/docs/contracts/skeleton.md`](../../kernel/docs/contracts/skeleton.md)
   owns the caller-supplied hot-loop callback boundary.
6. Public C++ values and identity live in
   `/node/include/rund/evidence/numeric/contract.hpp`; built-in constructors
   live in `/node/include/rund/evidence/numeric/preset.hpp`; the single
   implementation lives in `/node/src/evidence/numeric.cpp`.
   `/node/tests/contract/evidence/numeric.cpp` verifies both support owners and
   the canonical codec. The numeric directory carries their shared ownership;
   no root-level evidence forwarding path exists.

## Policy

Numeric authority has four separate axes:

| Axis | Meaning | Owner |
| --- | --- | --- |
| Scalar domain | Caller-visible value family: signed integer, unsigned integer, fixed-point, or floating-point. | Public API authority that admits the value. |
| Arithmetic law | Overflow, rounding, saturation, division, approximation, and accumulator behavior. | `/math32/docs` and `/math64/docs` for public deterministic integer/fixed-point SIMD lane law; owning contract otherwise. |
| Execution authority | Whether the value affects authoritative state, derived state, presentation-only state, or diagnostics. | The admitting API or domain contract. |
| Reduction classification | Fold primitive value class needed to validate ordered reduction. | Kernel reduction contract only. |

These axes must not collapse into one enum or shorthand type. A `u32` packet
count is kernel metadata, not proof that user values are `u32`.
`FoldValueDomain` is reduction-local, not the repository numeric model.

## Contract

Public math formula law is owned by `/math32/docs` and `/math64/docs`. This
page only routes numeric policy and contract identity across layers.

The public numeric contract uses these axes:

| Type | Meaning |
| --- | --- |
| `evidence::Domain` | Caller-visible scalar family: `I32`, `U32`, `I64`, `U64`, `Fixed`, `F32`, or `F64`. |
| `evidence::Arithmetic` | Operation meaning: integer wrap, integer saturation, fixed-point, non-authoritative floating-point, strict floating-point, reduction slot, or hash digest law. |
| `evidence::Authority` | Whether the result may enter authoritative, derived, presentation-only, or diagnostic state. |
| `evidence::Determinism` | Replay/hash requirement: required, best-effort, or not required. |
| `evidence::Contract` | Composition of scalar domain, arithmetic law, authority tier, determinism policy, and—only for `Fixed`—I/F, rounding, overflow, and approximation policy. |
| `evidence::Id` | Stable identity derived directly from the canonical contract fields and carried by run identity/proof surfaces. |

Do not use repo-wide `ValueDomain` as a public type name. `FoldValueDomain` is
reserved for kernel reduction-local classification.

## Invalid Combinations

- `evidence::Authority::Authoritative` with anything except
  `evidence::Determinism::Required`.
- `evidence::Arithmetic::FloatingPoint` or `evidence::Arithmetic::StrictFloatingPoint`
  outside `evidence::Domain::F32` or `evidence::Domain::F64`.
- `evidence::Authority::Authoritative` with `evidence::Arithmetic::FloatingPoint`;
  authoritative floating-point must use `evidence::Arithmetic::StrictFloatingPoint`
  and a fixed floating-point law.
- `evidence::Arithmetic::FixedPoint` outside `evidence::Domain::Fixed`, or a fixed
  contract whose positive I/F fields do not sum to 32 or 64.
- integer arithmetic laws outside integer scalar domains.
- reduction slot and hash digest laws outside `evidence::Domain::U64`.
- caller-supplied operation APIs as part of this contract surface.

`BestEffort` determinism is non-authoritative only. It may be used for derived,
presentation-only, or diagnostic data when that data cannot feed back into
authoritative deterministic state.

## Presets

Public UX must offer preset constructors for common contracts so callers do not
assemble raw axes for ordinary cases:

- `evidence::i32()`
- `evidence::i64()`
- `evidence::fixed<IntegerBits, FractionBits>()`
- `evidence::strict_f32()`
- `evidence::strict_f64()`
- `evidence::diagnostic_f32()`
- `evidence::diagnostic_f64()`
- `evidence::presentation_f32()`
- `evidence::presentation_f64()`

The `strict_f32` and `strict_f64` names are reserved for authoritative
floating-point contracts.

## Identity

`evidence::identify(contract)` validates and hashes the nine canonical
contract fields directly in `O(1)` time. There is no registry, inverse search,
serialized mirror, or public kernel-lowering record. Admission keeps the
resolved `evidence::Contract` beside its derived id and rejects the proof when the
contract is invalid or the id differs.

`evidence::Numeric` is the single replay-evidence value for an admitted numeric
contract. The value owns the nine-byte `evidence::Contract` and one-byte public
`evidence::Numeric::Code`; its truth conversion, `ok()`, `error()`, and
`exit_code()` derive from that code, while `id()`, `strict_float()`, and
`hash()` are derived observers. No bool, reason pointer, or exit status is
stored as a second outcome authority. The canonical text carries
schema, all nine contract fields, and the hash. Decode is one bounded pass over
the supplied view, allocates no inverse-search table, and validates the
contract and hash before returning success.

Unknown schema tags or fields, duplicate or missing fields, out-of-range enum
values, invalid fixed widths, invalid axis combinations, and hash mismatches
fail closed. Arithmetic and reduction lowering remain owned by the operation
or kernel contract that actually executes them; numeric evidence does not
invent a second execution-policy map.

## Compute Maps

`ComputeMap` authoritative execution admits `Fixed<I, F>` when `I + F` is 32
or 64. Every storage or primitive boundary carries that complete format
explicitly; lane width never supplies a default I/F split, and an unknown
scalar domain, operation, or numeric mode is rejected rather than translated
to a live Fixed or integer mode.
Detailed Compute map semantics route to
[`/kernel/docs/contracts/compute.md`](../../kernel/docs/contracts/compute.md), node
adapter/runtime ownership routes to
[`/node/docs/contracts/accel.md`](../../node/docs/contracts/accel.md), and the
public value UX routes to
[`/docs/reference/compute.md`](../reference/compute.md). Ratio construction
uses widened integer arithmetic,
explicit quotient rounding, and saturation; it never routes through a host
floating-point conversion. CPU availability requires frozen executable
`CpuCaps` and direct SIMD runner evidence. Accelerator availability requires
real backend discovery, pipeline creation, dispatch, readback, and
hash-checked fixed execution. Performance evidence remains separately required
before any speedup claim. Compute floating point is outside the public Compute
surface.

`ComputeIR` is a separate owning API authority for checked Compute map operations. It
does not change skeleton callback semantics and must not be treated as
arbitrary C++ callback lowering.

Compute Matrix descriptors carry their arithmetic law independently from
element width. `I32`/`I64` use signed-width wrapping multiply/add, `U32`/`U64`
use unsigned-width wrapping multiply/add, and fixed descriptors carry their
declared I/F and quantization policy. The law participates in primitive identity
and is executed identically by CPU, Metal, and Vulkan.

## Numeric Invariants

- Do not use `float` as shorthand for all non-integer numeric work.
- Do not treat `u32` kernel metadata as the universal user value domain.
- Do not treat `FoldValueDomain` as a universal scalar-domain model.
- Math32/Math64 own their Q1.31/Q1.63 raw-lane formula laws. Compute Fixed
  construction and arithmetic own the declared I/F scaling and quantization
  policy; lane width never supplies that binary point. Evidence tooling is
  never an execution mode or numeric-domain authority.
- Do not add a second public caller-supplied operation body, callback surface,
  or extension point that bypasses the skeleton authority without a separate
  owning API authority and verification surface.

## Update Rules

- Scalar domain, arithmetic law, authority tier, determinism policy, or numeric
  policy changes must update this page, public headers, and contract tests in
  the same change.
- Cluster run identity changes involving numeric contracts must update
  `/cluster/docs/contracts/run.md`.
- Numeric evidence changes must update this page and
  `/node/tests/contract/evidence/numeric.cpp`.
- Arithmetic operation law changes must update `/math32/docs` or
  `/math64/docs`, not this page.
- Reduction classification or strict FP reduction changes must update
  `/kernel/docs/contracts/reduction.md`, not this page.
