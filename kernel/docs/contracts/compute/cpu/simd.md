# Compute CPU SIMD Contract

## CPU SIMD Caps

`CpuCaps` is a frozen kernel input record. The kernel does not observe host CPU
features, call OS APIs, query compiler target state, inspect workload size, or
change semantic order from this record. Node may observe host features and copy
one chosen strategy into `CpuCaps`; after that handoff the record is ordinary
declared evidence.

The contract knows these deterministic CPU strategy names:

- `Scalar`
- `Sse2`
- `Avx2`
- `Avx512`
- `Neon`

`Cpu` is the public compute backend. SIMD strategy is not a backend and is not
semantic authority. It records only the lane shape selected before execution.
Every admitted CPU strategy must preserve the same fixed integer law as the
scalar DSL contract before node may execute it. If node cannot prove an
observed strategy executable, node must freeze or narrow to an executable
strategy before publishing an ok CPU backend.

A valid `CpuCaps` record has `ok = true`, `reason = "ok"`, backend `Cpu`, a
known SIMD strategy, nonzero lane bytes, and nonzero fixed_lane32
and fixed_lane64 lane counts. Invalid or unavailable host evidence must freeze to a
fail-closed caps record instead of silently selecting a runner.
`CpuCapsValid` is the one structural predicate. For strategy lane width `W`, it
requires `lane_bytes = W`, `fixed_lane32_lanes = W / 4`, and
`fixed_lane64_lanes = W / 8` (with scalar strategy explicitly mapping both lane
counts to one), in addition to `ok` and the CPU backend tag.

## CPU Artifact Admission

`LoweringArtifact` is the one artifact type for CPU, Metal, and Vulkan.
`LowerComputeIR(ir, ComputeApi::Cpu)` emits kind `CpuPlan`; the same checked IR
must produce the same key, source text, metadata, and canonical IR payload.
Source text starts with `rund.compute.cpu.plan` and is the sole emitted payload.
It records semantic binding and node layout, not a host lane strategy.

`CpuCaps` is execution evidence, not a second artifact identity. SIMD strategy,
lane width, and lane counts affect prepared CPU code and its device-capability
cache identity, but cannot split the canonical program artifact. One CPU
artifact can therefore prepare against any independently admitted compatible
host without a strategy-specific artifact/key/emitter/validator authority.

CPU lowering admits the same fixed_lane32/fixed_lane64 map op set as Metal and
Vulkan: params, reads, constants, writes, wrap arithmetic, unary fixed
ops, comparisons, predicates, bitwise ops, checked constant shifts,
saturating/fixed-scale arithmetic, `div_fixed`, `recip`, `sqrt`, and `rsqrt`.
Dynamic shifts, floating-point authority, atomics, unordered reductions, and
shared/scatter writes remain outside the admitted operation set.

The direct CPU runner first validates `CpuCaps`, then calls the common
`AdmitArtifact(ir, ComputeApi::Cpu, artifact)` owner. This boundary
requires exact canonical-payload equality, authenticates key and kind, parses
once, re-emits once, and compares all metadata and the sole source payload
before preparing. The same parsed admission is moved into compact CPU
preparation; there is no conversion artifact and no second parser. Its cold
boundary is parse `1` / emission `1`. CPU consumes the declaration-only
artifact-admission surface; it does not instantiate an inline validator body.

Internal Node Program preparation validates the same caps, calls the generic
`AdmitComputeInput(ir, ComputeApi::Cpu)` owner once, and builds the compact
prepared instruction owner directly. Its boundary is parse `1` / emission `0`.
Canonical IR, parsed IR, and artifact text do not survive as CPU Program owners,
and each already prepared execution is parse `0` / emission `0`.
