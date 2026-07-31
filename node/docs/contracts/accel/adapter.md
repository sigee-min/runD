# Node Accel Adapter Contract

## Contract

`AccelPolicy` is node policy. `Api::Auto` and `Api::Fake` must not cross into
kernel-facing `rund::kernel::ComputeApi`; admitted real backend capabilities
map only to CPU, Metal, or Vulkan.

`Api::Cpu` is a public policy target and a real backend admission path when
node can freeze executable CPU caps. CPU admission observes host CPU features,
narrows them to a direct deterministic runner shape, freezes kernel `CpuCaps`,
publishes generic `ComputeCaps` with `ComputeApi::Cpu`, and returns a usable
`ComputeBackendDispatch`. The CPU staged backend executes checked fixed_lane32/fixed_lane64
fixed map artifacts only; it is not compiler auto-vectorization, arbitrary
source execution, float authority, or workload-size-specific routing. The same
pick can open an `AccelContext` whose CPU native AccelKernel executor supports
`Map`, `Scan`, `SegmentedScan`, `Sort`, `Compact`, `Gather`, `Partition`,
`Reduce`, limited `Scatter`, and `Stencil` over authenticated host-resident
buffers with the same public UX as Metal and Vulkan.

`PickAccel` must:

- reject an empty policy with `compute_policy_empty`
- reject invalid `preferred_count > 3` with `compute_policy_invalid`
- accept explicit CPU selection when an executable deterministic CPU SIMD
  strategy can be frozen; otherwise reject with the precise frozen CPU-caps
  reason
- preserve stable policy order and return the first available adapter
- discover an adapter only; it must not choose whether an admitted workload
  runs on CPU or an accelerator path
- on Apple, select Metal when a system Metal device and command queue are
  available
- reject required unavailable Metal with `accel_metal_unavailable`,
  `accel_metal_device_unavailable`, `accel_metal_queue_unavailable`, or
  `accel_metal_sdk_unavailable`
- reject required unavailable Vulkan with a precise Vulkan reason such as
  `accel_vulkan_loader_unavailable`, `accel_vulkan_instance_unavailable`,
  `accel_vulkan_portability_unavailable`, `accel_vulkan_device_unavailable`,
  `accel_vulkan_queue_unavailable`, `accel_vulkan_shader_tool_unavailable`, or
  `accel_vulkan_unavailable`
- never select `Fake` unless `allow_fake == true`
- return frozen `rund::kernel::ComputeCaps`, including one nonzero power-of-two
  `storage_alignment`, a usable `ComputeBackendDispatch`, and an owner lifetime
  handle when an adapter is selected
- reject an otherwise successful adapter with
  `compute_adapter_capability_invalid` when that alignment contract is absent
  or malformed
- report SDK-free backend information such as device name, driver name, and
  driver info when the selected adapter exposes it

The default policy tries Metal and Vulkan; it does not include CPU or
Fake in its active preferred count. Callers that want CPU execution request
`Api::Cpu` explicitly or put it in their preferred policy order. `PickAccel`
freezes adapter caps and backend ownership; callers remain responsible for
explicit CPU/Accel run selection, and execution never falls back after the
selection is frozen.

Public code selects a backend through `rund::compute::open`. The compiled
bridge validates that closed public enum,
maps it once to exactly one real `Api`, builds a one-entry `AccelPolicy`, calls
`PickAccel` once, and requires the returned API to match before publishing the
Device. An invalid public enum rejects with `compute_backend_unsupported`
before adapter discovery. There is no internal backend enum, backend inventory,
name table, or second `require`/`prefer` selection surface. Adapter tests
exercise `AccelPolicy` directly; the installed Compute consumer owns explicit
CPU, Metal, and Vulkan selection coverage through the public enum.
Backend-specific resource reuse, command submission, timestamps, and CPU SIMD
scratch are private node facts; they must not leak into graph construction or
change semantic hashes.

Accel operations report one `rund::AccelCheck`. It is the sole owner of
operation success, reason, failed-batch count, first failed batch, and first
status; Node neither mirrors nor converts those fields into another result
type.

The CPU adapter is a private node adapter over kernel-owned deterministic
Compute contracts. On pick, its private `DetectCpu()` owner narrows observed
features to currently executable direct SIMD strategy evidence, freezes
`CpuCaps`, and exposes SDK-free backend info such as `cpu`, feature source, and
selected strategy name. The current executable CPU backend admits 16-byte
`Sse2` or `Neon` lane shapes and rejects unavailable or not-yet-executable CPU
strategy evidence fail-closed. Observed AVX2/AVX-512 features are advisory until
direct lane runners for those strategies are implemented and verified.

CPU staged execution admits executable 32-bit and 64-bit `Fixed<I,F>`
`CpuPlan` artifacts for the same kernel-owned supported fixed
map subset as Metal and Vulkan, including wrap arithmetic, unary fixed ops,
comparisons, predicates, storage bitwise ops, checked constant shifts,
saturating and fixed-scale arithmetic, and nonlinear `div_fixed`, `recip`, `sqrt`,
and `rsqrt`. Before execution, node validates the frozen plan and windows,
requires staged output storage, rejects resident CPU input/output and sequence
tile remap in this path, and authenticates key/domain/fixed format/canonical
input/source/metadata through the kernel common artifact admission owner.
Node then gives that owner's same parsed admission to compact CPU preparation
with frozen `CpuCaps`. It neither copies canonical bytes into an intermediate
owner nor reparses during preparation. The prepared binding layout and operation dispatch
then run tile chunks through direct SIMD vector loads and masked tail stores.
Kernel `ComputeTileExecutor` keeps worker partitioning and failure-order
authority for the CPU path.

The Apple Metal adapter is a real private node adapter when a Metal device and
command queue are available. It freezes `ComputeCaps` with
`ComputeApi::Metal`, compiles kernel-owned `MetalSource` lowering artifacts,
and caches compute pipelines by `ArtifactKey`. Pipeline admission proves
support for the canonical 256-thread transform group; device identity never
selects a different schedule. During selection, its adapter
state is the single identity source: it copies the nonempty UTF-8
`MTLDevice.name`, stores
`Metal` as the driver name, and leaves driver details empty because this path
has no separate API-proven driver string. The selected `AccelDevice` receives
an owning copy of that record. A missing/invalid UTF-8 name or C++ allocation
failure rejects selection with a precise Metal reason instead of publishing an
empty successful identity. It packs each dispatch window's input
spans into contiguous Metal buffers in `bindings.sequence_tile_at(...)` order,
bulk-copying identity sequence windows only when staged input and output
strides are contiguous after computing that staged-bulk proof once per
dispatch window,
validates the plan against the adapter's frozen caps before allocation or
dispatch. Its semantic window bound is the generated shader's `u32` grid
range; the kernel planner independently limits each window by the frozen 1 MiB
staging budget and actual bytes per tile. Node binds parameters, read buffers,
and one output buffer in generated
source argument order, dispatches exactly the window tile count, waits for
command completion, and scatters output bytes back into kernel staged-output
storage by semantic tile index for staged execution. Resident map execution
keeps the same semantic plan windows and resident buffer offsets, but
contiguous identity resident bindings may collapse to one full-range physical
dispatch after backend admission re-proves `PlanCompute` dispatch-count shape plus
generated 32-bit dispatch/addressing bounds for every resident input/output
ref. It remains a staged-output producer; Node graph execution owns result
publication after backend completion. The current Metal adapter admits executable 32-bit and
64-bit `Fixed<I,F>` `MetalSource` artifacts for the kernel-owned supported fixed map
subset, including wrap add/sub/mul/neg, saturating abs, unsigned-magnitude
bits, sign, signed min/max/clamp, `0/1` comparisons and predicates, and
select-on-nonzero, plus storage bitwise `bit_and`, `bit_or`, `bit_xor`,
`bit_not`, and checked constant shifts `shl_const<N>`,
`shr_logical_const<N>`, and `shr_arithmetic_const<N>`, plus fixed-scale
arithmetic `add_sat`, `add_sat_unsigned`, `sub_sat`,
`neg_positive_fixed`, `mul_fixed`, `mul_fixed_scaled`,
`mul_unsigned_fixed`, and `mul_add_fixed`, plus nonlinear fixed operators
`div_fixed`, `recip`, `sqrt`, and `rsqrt`. The constant shift operations use
the kernel-owned unsigned storage law: logical right shift shifts unsigned
storage, arithmetic right shift uses explicit sign-extension masks, and
constant overshifts fail before shader compilation. Fixed multiply-family
source uses kernel-emitted widened helpers and the declared `(I,F)` binary
point; it does not assume a Q1 format. Division and square root use widened
unsigned long division or explicit limb long-division/integer-sqrt helpers
over widened radicands or numerators. These helpers do not accept shader 128-bit integer
types, backend float conversions, backend intrinsic sqrt/reciprocal/rsqrt
authority, fast math, atomics, scatter writes, reductions, or arbitrary shader
text as arithmetic authority. Dynamic shift operators are not admitted.
Unsupported artifact kinds still fail closed through the backend path. Before
pipeline compilation, Metal consumes the kernel common artifact admission
token. That owner borrows canonical input, parses and emits once, and compares
the complete key, metadata, and sole source payload. Metal has no backend-local
re-lowering, partial key authority, or source marker scan.

The Vulkan adapter is a real private node adapter when Vulkan loader/header
linkage, instance creation, macOS portability enumeration when needed,
physical-device discovery, `shaderInt64` support, logical device and compute
queue creation, and `glslangValidator` discovery have all succeeded.
It also requires at least 256 first-dimension workgroup invocations and freezes
the advertised `maxComputeWorkGroupCount[0]` exactly. The canonical 64-bit
complex transform block needs 4 KiB of compute shared memory, below the Vulkan
Core minimum, without a device-specific schedule. The map tile bound is
`min(maxComputeWorkGroupCount[0] * 256, UINT32_MAX)`; the kernel planner then
limits the chosen window by actual staging bytes per tile. Node does not clamp
or invent a smaller fixed tile window, and collective encoders continue to
chunk their own shapes against the same frozen device fact.
`spirv-val` is recorded and used when available, but absence of `spirv-val`
does not make an otherwise valid Vulkan adapter unavailable. It reports
SDK-free `AccelBackendInfo` with the Vulkan device name and, when Vulkan driver
properties are available, the driver name and driver info; MoltenVK picks must
surface `MoltenVK` driver information through that support field. That pick
freezes real `ComputeApi::Vulkan` caps and owns private Vulkan objects.

`ComputeApi::Vulkan` names the API and lowering contract, not an assertion that
the physical device driver is native Vulkan. When backend information reports
`MoltenVK`, Node is executing the Vulkan object, SPIR-V, descriptor, command,
barrier, and synchronization path through Vulkan-to-Metal translation on the
Apple GPU. Those runs prove that exact translated path and may be compared only
as same-path regression evidence. Their timings are not native Vulkan
throughput evidence. The direct Metal adapter remains the native Metal timing
path on that host; comparing its time with MoltenVK compares two complete
software stacks and does not isolate Vulkan API cost.

The selected accelerator's `ComputeCaps::device_bytes` is the backend's
aggregate working-set budget, not installed physical memory and not a
per-Buffer range. Metal reads `recommendedMaxWorkingSetSize`, falling back to
`maxBufferLength` only when the working-set value is unavailable. Vulkan reads
the device-local `VK_EXT_memory_budget` heap budget when supported and falls
back to device-local physical heap size only when that extension supplies no
budget. On MoltenVK these values describe the same underlying Metal device and
must not be added or interpreted as independent hardware capacity. Native
out-of-device-memory results project to `compute_device_capacity`; a host-side
or object-count allocation limit projects to `compute_buffer_capacity`.

`ComputeCaps::storage_alignment` is the minimum legal byte offset for a
resident storage subrange. CPU publishes its 64-byte host arena alignment,
Metal publishes the four-byte scalar alignment accepted by its Buffer-offset
binding, and Vulkan publishes
`max(4, minStorageBufferOffsetAlignment)`. Pipeline View planning consumes this
frozen value before allocation. It does not infer alignment from the current
machine, driver name, or a hard-coded Vulkan constant.

Vulkan staged execution admits executable 32-bit and 64-bit `Fixed<I,F>` `VulkanSource`
artifacts for the same kernel-owned supported fixed map subset, including the
storage bitwise operations, checked constant shifts, fixed-scale arithmetic,
and nonlinear fixed operators named above. Before dispatch,
node validates the plan against frozen caps, validates runtime bindings and
dispatch windows, and consumes the same kernel common artifact admission token
used by CPU and Metal. It then compiles the authenticated source to nonempty
SPIR-V with `glslangValidator`, treats the compiler child-process wait as
interruptible host IO, hashes the SPIR-V bytes, and runs `spirv-val` when
available. An exact compiler-path, validator-path, and complete-source match
may reuse immutable SPIR-V that this process already compiled and validated;
the bounded internal cache never accepts a hash match alone and does not reuse
unvalidated output under a validator-enabled key. Vulkan has no backend-local
re-lowering or second admission during pipeline compilation. The shader source
prefilter, cache-key framing, and SPIR-V byte hashing consume the
same allocation-free FNV state transition owner. Each call site supplies its
frozen seed and framing bytes, so this consolidation changes neither stored
hashes nor cache identity and adds no second traversal. It removes backend-local
copies of `h' = (h xor byte) * prime mod 2^64`.

It then creates a shader module, descriptor set layout, pipeline layout,
compute pipeline, descriptor set,
command pool, command buffer, storage buffers, and fence; binds descriptors in
kernel lowering order (`0` params, `1..N` reads, `N+1` output), pushes the
window tile count once per map dispatch, and emits `ceil(tile_count / 256)`
fixed-width workgroups. The generated shader returns excess lanes before any
buffer access. Node then waits for the fence, reads back the
host-visible output buffer; and scatters output bytes by
`bindings.sequence_tile_at(...)` into existing staged storage. Node graph
execution owns result publication and dependency order.

Vulkan execution must fail closed with a precise reason when source validation,
SPIR-V compilation/validation, shader module creation, descriptor creation,
pipeline creation, command allocation/recording, memory-backed buffer
allocation, submit, fence wait, or readback fails. Runtime reasons include
`compute_artifact_mismatch`, `compute_artifact_non_executable`,
`accel_vulkan_shader_compile_failed`,
`accel_vulkan_shader_process_failed`, `accel_vulkan_spirv_read_failed`,
`accel_vulkan_spirv_invalid`, `accel_vulkan_shader_module_unavailable`,
`accel_vulkan_descriptor_unavailable`, `accel_vulkan_pipeline_unavailable`,
`accel_vulkan_command_unavailable`, `accel_vulkan_buffer_unavailable`,
`accel_vulkan_memory_unavailable`, `accel_vulkan_submit_failed`,
`accel_vulkan_command_capacity`, `accel_vulkan_command_sequence`,
`accel_vulkan_fence_failed`, and `accel_vulkan_readback_failed`.

`accel_vulkan_unavailable` is the final Vulkan discovery catch-all. It must be
used only after loader/tooling, instance, portability, physical device, and
compute queue evidence have been classified and discovery reaches an otherwise
unclassified or impossible state. Specific failures must keep their specific
reason.

On non-Apple builds, Metal remains a C++ fail-closed unavailable surface.
Vulkan uses private Vulkan/MoltenVK discovery and execution when the loader,
ICD, compute queue,
shader compiler, shader module, descriptors, command buffer, memory-backed
dispatch, fence wait, and readback all validate. A fail-closed Vulkan reason is
an exact blocker record, not a permanent availability contract. Node must not
claim Vulkan execution from an identity-only artifact, fake adapter, missing
shader module, or skipped dispatch.

## Header Boundary

Public node and kernel headers must not include Metal, Objective-C, Vulkan,
platform windowing, or SDK loader headers as host headers. Adapter SDK
dependencies stay in private node implementation files; kernel lowering may
emit backend source text as data without making that SDK a public host include.

## Verification

```bash
tools/check/run
```
