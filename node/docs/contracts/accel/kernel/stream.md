# AccelKernel Stream Contract

## Scope

`RunAccelKernel(context, kernel, run)` executes an authenticated resident graph
on CPU, Metal, or Vulkan. Kernel owns graph semantics and frozen plans. Node
owns resident binding admission, backend preparation, ordered execution, and
evidence publication.

## One Execution Authority

Compilation creates one immutable kernel token. Run admission authenticates the
context and every buffer, then builds one ordered `BoundRun` containing borrowed
step plans, resident views, and already-planned dispatch windows. The token
owner pins all borrowed storage. Common code performs no backend-specific
primitive switch and retains no second graph.

CPU executes the ordered steps directly. Metal and Vulkan prepare one native
command-stream owner, encode supported steps in graph order, submit once, and
finish status in the same order. Unsupported steps reject before encoding;
execution never retries on another backend.

Prepared Metal and Vulkan streams share one concrete submission transition and
one ordered finish fold. A non-null prepared-owner pointer is the complete
busy/completion identity; claim, take, and cancellation publish or clear it
under one mutex. Duplicate completion is therefore a no-op by construction,
while failure precedence remains increasing prepared-step order. Backend code
owns native command encoding and status observation only.

The validated graph node order is the sole step-order authority. Runtime
preparation performs one linear validation over the final steps and their
binding occurrences, then exposes the identity order `0..S-1`; it does not
construct a second dependency scheduler, rescan pairwise resource conflicts,
or reorder a step from backend timing. For `R` total final-step binding
occurrences, this admission work is `Theta(R)` time and `O(1)` additional
storage. Logical alias and dependency meaning remains frozen in the kernel
graph that produced those steps.

Map fusion planning also owns canonical IR capacity. A region ends before
adding the next Map would exceed 64 bindings or 1024 IR nodes, using the exact
retained costs and the boundary law `(B + b - 2, V + v - 2)`. Node consumes
those boundary bits directly and lowers each resulting maximal region once;
there is no optimistic whole-region attempt, size fallback, or second split
authority.

That region limit is independent of the 16,384-node Program graph envelope.
The complete ordered graph remains one authenticated kernel token and one
native command stream. Region boundaries never create a host callback,
readback, Program split, submission split, or backend-specific size strategy.

Every final step retains the exact half-open source interval `[begin, end)` it
replaces. The intervals are a gap-free ordered partition of the original graph:

```text
begin[0] = 0
end[j] = begin[j + 1]
end[S - 1] = original_node_count
```

The unique projection `project(i) = j` where
`begin[j] <= i < end[j]` is the sole original-to-final step authority.
First-write initialization is authored once as `BufferInit::Zero` on the
canonical Write binding and enters graph identity. Token mint validates the
partition, matches every surviving binding to its exact original owner, and
seals `(binding, first execution step, alias last execution step)`. Last use is
selected by maximum original source ordinal, independent of final-binding
traversal order. A legally fused-away internal intermediate retains no physical
binding and therefore no reset plan. An external or otherwise materialized
write must survive and own exactly one plan.

For `S` final steps, `B` canonical graph binding occurrences, and `R`
surviving reset routes, this cold proof is `Theta(S + B)`. The immutable token
retains two 32-bit source interval endpoints per final step and one 16-byte
reset plan per reset: exactly `8 * S + 16 * R` logical bytes. `AccelRun`
carries no reset pointer, count, source step, or last-use mirror, so an
invocation cannot forge or drift a frontier. Reset-free preparation returns in
`Theta(1)` with no reset allocation. Reset binding projection is `Theta(R)`;
`reset/overlap.cpp` exclusively owns the independent physical overlap proof in
`O(R log R)`. One interval store and three reusable group workspaces serve the
whole proof; a workspace allocates again only when a later group exceeds its
retained capacity. Preparation
never rebuilds fusion decisions, disables fusion, or searches final steps.

Up to four final steps and eight run bindings stay inline. Crossing either
bound is transactional: overflow storage is published only after allocation
succeeds, so cleanup never observes a partially constructed range.

Reset contract ownership is split under
`node/tests/contract/accel/kernel/reset/`: `model.cpp` owns range width,
layout, and invalid-range proofs; `execution.cpp` owns Vulkan reset command
shape; `projection.cpp` owns transfer lookup and source-partition projection;
`overlap.cpp` owns lifetime compatibility; `sealing.cpp` owns bound reset-plan
identity and rejection; and `support.cpp` owns only the shared range-validity
predicates. `dispatcher.cpp` remains the sole `ResetModelContract` authority
and preserves the original short-circuit order.

The authority phase proof is likewise split by contract boundary:
`authority/phase.cpp` owns the nested-coordinate codec, recurrence identity,
and prepared-control checks, while `authority/phase/source.cpp` owns generated
Metal/Vulkan source consumption and hash/ordering evidence. Both leaves consume
the same backend phase definitions; neither recreates a phase table or source
recipe.

## Ordering and Numeric Meaning

Map windows preserve exact sequence identity. Scan and reduction consume the
kernel's fixed plan and publish the canonical overflow reason. Segmented
operations preserve ascending segment order and validate head flags. Stable
sort and partition preserve input order within equal keys or predicate groups.
Gather, histogram, scatter, stencil, and numeric algebra use their admitted
resident shapes and status buffers. Backend completion order, thread arrival,
and cache state are not semantic authority.

Final dispatch counts come from the frozen final kernel steps. For each fused
Map region, compilation stores only the checked difference between the
original per-Map dispatch sum and the fused dispatch count. Runtime reconstructs
the public original count as `final + removed`; it does not retain or recompute
a second whole-graph dispatch plan. Physical dispatches and command submissions
are runtime evidence and may differ without changing graph or output identity.

Metal and Vulkan numeric algebra share one fixed Transform schedule. A
256-lane local dispatch cooperatively loads up to 256 bit-reversed values and
executes the first `min(8, log2(N))` radix-2 stages in shared memory. Remaining
stages execute in adjacent pairs, retaining their intermediate value in a
register; an unpaired final stage keeps the same operation order. The exact
dispatch count is `1 + ceil(max(log2(N) - 8, 0) / 2)`. Factor, Solve, and
Spectrum dispatch one 32-lane threadgroup per batch. A barrier separates every
data-dependent pivot, column, substitution row, Jacobi rotation, and
orthogonalization step; within one step, independent rows, columns, and cells
are distributed across the lanes.

Metal numeric preparation publishes group count, lane width, grouped mode,
dispatch count, and prepared ownership through one `PublishPrepared`
transition after operation-specific validation, binding, pipeline, parameter,
and status work completes. Transform, Matrix, Factor, Solve, and Spectrum do
not retain separate mutable publication epilogues. The transition moves the
existing owner and writes four scalar fields; it allocates and copies nothing.

Metal execution has one command lifecycle authority.
`CommandRun` owns the strong command-buffer and encoder references;
`OpenCommand` performs queue projection and both native creations, while
`CloseCommand` is the sole encoder-ending transition and `FinishCommand`
submits only a successful encoding. Histogram, compact, gather, partition,
reduce, scan, scatter, segmented scan/reduce, sort, stencil, numeric algebra,
and staged or resident runtime windows consume that same transition.
Both transitions are inline Objective-C++ and add no allocation, copy,
dispatch, or native call.
Prepared Kernel and Pipeline streams consume the same `CommandRun` and
`OpenCommand` authority with the fixed `ResourceRefs::Borrowed` policy because
their prepared owner already pins every referenced resource through
completion. Ordinary standalone commands use `ResourceRefs::Retained`. The
policy is a call-site constant in the inline transition; it does not create a
workload-dependent path or a second lifecycle state.
The `command/submit` owner then provides the only two terminal transitions:
`WaitCommand` for synchronous completion and `QueueCommand` for callback
completion. Backend-prefixed forwarding names, staged command owners, and
resident-window command owners are not admitted.
Failed staged input binding and failed resident-window encoding still close
the one encoder exactly once and never submit a partial command.

Vulkan command storage has one native lifecycle owner.
`command/model.hpp` defines the sole
`VkCommandPool + VkCommandBuffer + optional VkFence` state.
`command/resources.cpp` creates and destroys that state transactionally
through its narrow compiled interface. The closed
`OneShotPrimary`, `ImmutableSecondary`, and `ReusablePrimary` kinds freeze
level, pool flags, begin flags, inheritance, fence presence, and initial fence
state. Batch keeps immutable secondary execution, Pipeline keeps one reusable
primary, and the adapter ring keeps independent one-shot primary slots.
Reusable primaries freeze `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT`:
temporal window/stream submission may place the same retained bank command in
multiple pending submit records. Timeline dependencies still order execution;
the recording flag admits repeated pending instances, not unsynchronized
resource access. The backend runtime command-plan test checks this exact flag,
alongside unchanged one-shot and immutable-secondary flags.
Timestamp query pools remain diagnostic slot or Pipeline-profile resources;
they are not command-lifetime state. Immutable Kernel and Pipeline recording
share the compiled cold begin/end transition. The ring retains its
reset-and-one-time hot transition beside sequence publication, so the common
lifetime owner adds no call, branch, allocation, or lock to submission.

The topology does not reassociate fixed arithmetic. Every dot product and
stored accumulation still folds in ascending `k` order, because saturation and
rounding make `(a + b) + c` unequal to `a + (b + c)` in general. Parallel work
is limited to outputs whose writes are disjoint and whose inputs were frozen by
the preceding barrier. For transform size `N`, per-lane butterfly work is at
most `ceil((N/2)/256)` per stage. For numeric-algebra dimensions bounded by 16,
each dependency step exposes up to 256 disjoint cells to 32 lanes. The 32- and
64-bit source instances consume one common algorithm body; storage-width
helpers change representation only. `accel.kernel-numeric` must prove the
fixed source/dispatch topology and bit-for-bit CPU parity for every available
backend rather than accepting batch parallelism or a threadgroup-size setting
alone as evidence.

The Metal numeric Solve recipe is physically split by shader role under
`node/src/accel/metal/numeric/source/program/solve/`: `factor.hpp` owns the
LU, Cholesky, and QR factor-input substitutions; `direct.hpp` owns direct
matrix-input LU and Cholesky; and `kernel.hpp` owns the one ordered dispatch
entry and status publication. The adjacent `solve.hpp` is an include-only
ordered facade that appends those three unchanged fragments. It carries no
second recipe or solve-state authority, so the program's Factor → Solve →
Spectrum order and source bytes remain fixed.

GPU map windows are byte-budgeted rather than fixed at a small tile count. The
kernel chooses the largest window admitted by device, caller, phase, and
staging bounds. Metal dispatches the exact thread count. Vulkan binds the same
window once, publishes its `u32` tile count, and dispatches fixed 256-lane
groups; excess lanes return before reading or writing. For `N` tiles this
reduces the Vulkan workgroup count from `N` to `ceil(N / 256)` while preserving
one independent invocation for every original `gid`. No padding buffer, copy,
fallback pipeline, or alternate result order exists.

Metal Map binding specialization also freezes one four-byte word-access class
per binding. For authored offset `o` and stride `s`, the class is enabled
exactly when

```text
o mod 4 = 0 and s mod 4 = 0.
```

Every access selects an integer logical index `k`: direct access uses
`k = b + gid` for exact window begin `b`, uniform access uses `k = 0`, and
indexed access uses its checked runtime index. The byte address `o + k*s`
remains divisible by four under that predicate. Direct windows bind native
offset `o + b*s` and add `gid*s` in the shader; the other access modes preserve
the same general address law. Metal Map passes alignment `1` to source
specialization, so the shader base remains zero. This satisfies Metal's
[`device` buffer-offset rule](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbuffer%28_%3Aoffset%3Aindex%3A%29),
which requires alignment to the argument data type's size. The specialized
kernel parameter is therefore `device uint*`,
and overload selection performs one word load/store for a 32-bit lane or two
ordered low/high word accesses for a 64-bit lane, replacing respectively four
or eight scalar `uchar` accesses in the byte helper. It never requires an
eight-byte-aligned pointer. If either congruence fails, the parameter stays
`device uchar*` and the canonical bytewise helper preserves the arbitrary
public View exactly. These are exact source-operation counts, not claims about
a device's final memory-transaction count.

The exact specialized source, including every pointee token, remains the
native pipeline-cache identity. One allocation-free `(offset, stride)`
projection owns the per-binding `Bytewise`/`Word32` class. Public Program
template equivalence, private `BackendRun` equivalence, the Metal registry
matcher, recurrence terminal/history equivalence, and the Pipeline
specialization fingerprint all consume that projection. Recurrence capacity
and template-count deduplication therefore reserve distinct immutable owners
for aligned and unaligned routes before publication; changing only the class
cannot select or under-reserve a typed pipeline prepared for another route.

Prepared invocation resets consume the single overflow-free `Range` value and
projection proof owned by
[Compute Memory Ownership](../../compute/memory.md). Metal and Vulkan retain
only their native handle, descriptor, and command form around that proved
range; command encoding never reconstructs the range or repeats its bounds
proof. `Range::valid()` is the sole reset-proof disposition: rejection returns
the canonical empty range, and each consumer projects the fixed
`accel_kernel_reset_invalid` reason at its own API boundary. No parallel check,
reason, or conditional range carrier survives proof publication.

## Prepared Lifetime

Synchronous run and asynchronous submit consume the same prepared owner and
the same execution claim. The prepared owner retains its resources through
callback return while the device adapter supplies one slot from its bounded
command envelope. Backend resources are destroyed before their borrowed common
binding and plan storage. Reusing the same prepared graph concurrently fails
with `compute_job_busy`; independent prepared graphs may occupy distinct device
slots; a foreign adapter fails before claim acquisition.

Resident execution stages no host payload and performs no implicit readback.
`DownloadAccelBuffer` is the explicit result boundary.

## Verification

`accel.kernel-core` covers token admission, resident binding identity, prepared
lifetime, reset-frontier rejection, collectives, and failure ordering. Its
reset planner matrix enumerates all gap-free fusion partitions and every reset
source for one through seven original steps: 769 valid projections. It also
rejects a read-first alias, an out-of-interval source, a cyclic representative,
a duplicate marker, and a fused-away external binding.
`compute.flow` covers the fused source-interval projection with two partial
`u64` scatter resets and exact cold/warm CPU, Metal, and Vulkan output.
`accel.kernel-numeric` covers Transform, Matrix, Factor, Solve, and Spectrum.
Its Spectrum runner keeps the ordered dense, values-only, and full-vector
execution evidence in `tests/contract/accel/kernel/spectrum.cpp`; the
invalid-shape admission and native-backend availability checks are owned by
`spectrum/validation.cpp` and retain the same numeric case and call order.
Backend-specific fixed and runtime evidence is owned by
`accel.backend-fixed` and `accel.backend-runtime`.

The Solve contract keeps `kernel/solve.cpp` as the native-backend runner.
`kernel/solve/model.hpp` is its sole buffer, fixed-format, download, and
comparison fixture; `raw.cpp` owns direct matrix-input Solve and `reuse.cpp`
owns Factor-to-Solve reuse. The three translation units retain the existing
CPU, Metal, Vulkan, 32-bit, 64-bit, factor, status, dense, and exact-output
order without registering another case.

The Factor contract is physically split under
`tests/contract/accel/kernel/factor/`:
`rejection.cpp` owns the cross-primitive oversize-shape rejection,
The backend-neutral run model is physically split under `kernel/backend/run/`:
`bound.hpp` owns sealed reset and step bindings, `model.hpp` owns template
demand and the execution route, `window.hpp` owns recurrence/window/publication
state, `batch.hpp` owns native batch entries, and `binding.hpp` owns rebinding
and bound-run construction declarations. `backend/run.hpp` is an ordered
implementation-free compatibility umbrella and owns no parallel model. Its
implementation follows the same boundary: `backend/run.cpp` owns only ordered
bound-run assembly, while `backend/run/bindings.cpp`, `map.cpp`, and `state.cpp`
respectively own primitive/control binding, Map dispatch-window projection,
and `BoundRun` move/bind state. Their `internal.hpp` contains only private
declarations.
`execution.cpp` owns the ordered small-matrix and native-backend runner, and
`dense.cpp` owns the batched typed reference/output comparison. These leaves
retain the existing CPU, Metal, Vulkan, 32-bit, 64-bit, operation, status, and
dense-case order; `internal.hpp` carries only the dense-case declarations.
