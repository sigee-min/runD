# Accel Docs

`/accel` is the internal backend-neutral execution contract boundary. It
consumes `/kernel` and is consumed by `/node` for concrete CPU SIMD, Metal,
and Vulkan execution. Applications reach it only through
`<rund/compute.hpp>` and compiled Node bridge sources.

## Authority

1. [`/docs/architecture/topology.md`](../../docs/architecture/topology.md)
2. This page for the Accel boundary.
3. [Internal API](./api.md) for graph factory and Node handoff authority.
4. [Verification](./verification.md) for Accel-owned evidence routing.
5. Internal support headers under `/accel/include/accel`.
6. Node backend selection and resource execution under
   [`/node/docs/contracts/accel.md`](../../node/docs/contracts/accel.md).
7. Package surface policy under [`/package`](../../package/README.md).

## Boundary

Accel owns backend-neutral names, graph/kernel/run/evidence value shapes, and
the deterministic advisory policy used to decide whether a proved kernel
compute plan may use an accelerator path. It may depend on kernel compute
contracts because `AccelKernel` is a lowered compute contract over checked IR.

Accel does not observe host devices, choose a backend, open contexts, allocate
resident buffers, own command queues, compile Metal or Vulkan artifacts, measure
timing, or submit work. Those concrete runtime responsibilities stay under
node.

Kernel is the sole owner of compute-plan shape, byte, dispatch, and checked
arithmetic validity. Accelerator admission consumes
`ComputePlanShapeValid(...)` directly; Accel does not mirror the equations or
translate the plan through an adapter.

The advisory run policy has one storage-path state: `staged == true` admits
the staged accelerator path and `false` requires resident execution. There is
no inverse mirror flag or invalid four-state combination.

## Docs Map

| Page | Owns |
| --- | --- |
| [Internal API](./api.md) | Internal Accel values, graph factories, and naming boundaries. |
| [Verification](./verification.md) | Shape, package, and node Accel verification routing. |
