# GPU Performance

One million elements is large data. It is not automatically enough work to
amortize a GPU submission.

## The Cost Model

```text
CPU = element count * CPU cost per element

GPU = submit + synchronize + transfer + element count * GPU cost per element
```

A GPU wins only after the avoided per-element CPU work exceeds the fixed submission
and synchronization cost. Arithmetic intensity matters as much as element
count: a light map can move many bytes while performing very little arithmetic.

Resident execution removes warm allocation, upload, and download. It does not
remove dispatch, synchronization, or device-memory traffic. This is why a
light one-million-element resident map can still favor the CPU on one machine,
while a fused Pipeline or one Batch of many jobs can strongly favor the GPU.

The exact admitted M4 Pro observations, equations, scope limits, and
MoltenVK interpretation are maintained in the
[GPU workload sizing reference](https://github.com/sigee-min/runD/blob/main/docs/reference/performance/gpu.md).

## Use the Right Surface

| Need | Use |
| --- | --- |
| One convenient host result | `collect()` |
| Reuse one compiled graph with new input | `Program::run()` |
| Keep state and intermediates on the device | `Program::resident()` |
| Execute dependent stages without host round trips | `Pipeline` |
| Submit many independent jobs together | `Batch` |

The high-return path is:

1. Compile once.
2. Keep reusable data resident.
3. Fuse dependent element-wise stages.
4. Batch independent jobs behind one submission.
5. Read back only at the application boundary.

runD does not select another algorithm from the workload size. The canonical
graph, ordering, Fixed policy, and output bits remain the authority; only the
physical execution plan may change.

## Read Measurements Correctly

- A baseline row is valid only for its sealed host, driver, source identity,
  workload, and verified output.
- Metal on Apple is native Metal evidence.
- Vulkan on the admitted Apple profile is Vulkan lowering through MoltenVK,
  not native Vulkan throughput.
- A regression limit is not a portable speed claim.

See the [measurement method](https://github.com/sigee-min/runD/blob/main/docs/reference/performance/method.md)
and [Compute API](./Compute.md) before
publishing a comparison.
