# Accel Verification

## Evidence Boundary

Accel verification proves backend-neutral value shape, graph factory routing,
package surface reachability, and node-owned execution parity. It does
not prove hardware speedup, driver behavior, cache behavior, PMU state, or
portable timing.

`AccelEvidence` may carry node-produced diagnostic resource counters such as
pipeline compile/cache, descriptor allocation/reuse, and buffer
allocation/reuse. These counters are evidence fields only; graph identity and
output identity stay backend-neutral.

Node contract coverage proves representative affine, interpolation, vector,
statistics, correlation, and graph primitive composites through the staged
prepared map or graph path when a backend is available. The evidence is
fixed_lane32/fixed_lane64 bit parity or descriptor identity, not a performance claim or a
layout authority.

## Commands

- Fast graph-factory loop: `tools/test/run accel.surface`.
- Direct SIMD loop: `tools/test/run accel.cpu-simd.vector`.
- CPU backend loop: `tools/test/run accel.cpu-simd.backend`.
- CPU graph/kernel loop: `tools/test/run accel.cpu-kernel`.
- Numeric topology/parity loop: `tools/test/run accel.kernel-numeric`.
- Full backend graph/kernel loop: `tools/test/run accel.kernel-core`.
- Accel contracts: `tools/test/run --match '^accel[.]'`.
- Complete local closure, including package consumers: `tools/check/run` followed
  by `tools/release/run`.

The first three CPU execution routes acquire no accelerator resource. Direct
SIMD closes at `CPU_SIMD`; CPU backend and CPU graph/kernel close at
`CPU_ACCEL`. Their build graphs must contain zero Metal or Vulkan source
compiles, link flags, undefined archive symbols, and runtime library
dependencies. `accel.kernel-core` retains
`ACCEL_EXECUTION` and the accelerator resource because it owns full
real-backend selection and parity.

`accel.surface` admits no Node execution SCC. It links the exact
`kernel-closure-compute` view because inline Accel graph factories attach graph
signatures whose sole compiled implementation owner is Kernel. The factory
loop has no header implementation mirror and excludes CPU, Metal, Vulkan,
context, and runtime Node objects.

The numeric loop compares 32- and 64-bit outputs with the canonical CPU
reference on admitted native backends. Its Vulkan topology contract also
rejects a one-lane Factor, Solve, Spectrum, or Transform shader and requires
the stage barriers used by the deterministic lane-strided implementation.
This is semantic and structural evidence. Performance claims require the
sealed `tools/measure/compute/run` packet; test elapsed time is not benchmark
evidence.
