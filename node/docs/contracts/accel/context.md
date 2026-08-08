# Node AccelContext Contract

## AccelContext Support Surface

`AccelContext`, `AccelBuffer`, `AccelBufferDesc`, `AccelCheck`,
`AccelContextEvidence`, `OpenAccel(pick)`, `OpenAccelBuffer(context, buffer,
desc)`, `CreateAccelBuffer(context, desc)`, `UploadAccelBuffer(...)`, and
`DownloadAccelBuffer(...)` are repository-internal Node support types. The context and
buffer surface owns admission, ownership, and explicit buffer-transfer support
only; it does not run kernels, allocate a second backend registry, or create a
new execution layer.

`OpenAccel` accepts an already selected resident-capable `AccelDevice` only when
private node authentication proves the pick owns a CPU, Metal, or Vulkan
adapter. Admission reuses the existing private CPU, Metal, and Vulkan owner
predicates; it does not trust public aggregate fields, raw pointer equality,
arbitrary function pointers, or lookalike `shared_ptr` values with a different
control block. The public frozen caps in `AccelDevice` must exactly match the
canonical private adapter caps for API, device bytes, staging bytes, max window
tiles, subgroup width, ok bit, and reason before context
admission succeeds. `Api::Fake` remains pick-admissible diagnostic plumbing, but
it has no resident registry and is not context-admissible. A successful context keeps the
authenticated adapter owner in `context.pick.owner`, mints a separate
context-local support token in `context.owner`, assigns a nonzero `context.id`, and
records node runtime stats as advisory evidence. The token is authenticated
through a private deleter type sealed into its `shared_ptr` control block, not
by trusting the public `shared_ptr<void>` field alone. Admission recovers the
typed token with `std::get_deleter` in `O(1)` and returns that retained
`shared_ptr<ContextToken>` directly. A null token is rejection and a non-null
token is admission; there is no parallel `AccelCheck` or wrapper state that
can disagree with the capability lifetime. Admission checks the public context
id, adapter owner, caps, API, evidence, and exact stored pointer. There is no
process-global context table, mutex, expired-entry compaction, owner-order
search, or second token lifetime authority. A forged control block lacks the
private deleter and an alias stored pointer fails the exact-pointer check. The
context-local token and id bind `AccelBuffer` use to the context that admitted
or created it; backend authentication still comes only from the existing
adapter owner and resident registry.

Context opening fails closed with:

- `accel_context_pick_invalid` for missing, failed, ownerless, or unusable picks.

`OpenAccelBuffer` creates a typed resident view over an existing node `Buffer`.
It does not allocate backend memory and does not authenticate by raw pointer
identity. Admission derives a typed resident view, then reuses the selected
CPU, Metal, or Vulkan resident registry lookup with the buffer's private handle
sidecar. Synthetic public `Buffer` aggregates, forged ids, forged resident
handles, and alias control blocks do not authenticate even when their public
byte extent or owner pointer values look plausible. Registry lookup returns
the canonical private resident extent; public buffer id, byte extent, element
width, stride, count, and usage-derived kernel usage must
match that canonical registry fact before typed admission succeeds. The
returned typed resident descriptor is derived from the canonical registry
extent plus the requested typed shape, not from mutable public buffer fields.
Successful admission seals the typed view in a second private-deleter
capability. The capability retains the context token, original backend handle,
canonical resident descriptor, and typed shape. Public `AccelBuffer::handle`
and its nested support buffer handle name that capability; the backend handle
never becomes the typed-view identity. Warm typed-buffer admission is `O(1)`
capability recovery followed by the existing backend resident lookup. It
checks the complete public aggregate against the sealed facts, then confirms
the retained backend handle and extent against the backend registry. There is
no context-buffer registry, registration pass, weak-record compaction, global
buffer mutex, or owner-order scan.

Buffer admission fails closed with:

- `accel_context_buffer_invalid` for missing context admission, missing buffer
  admission, zero shape, unknown usage, usage mismatch, missing owner/handle,
  or owner-token mismatch.
- `accel_context_buffer_overflow` when `scalar_width_bytes * count` overflows or
  exceeds the backing buffer byte extent.

One typed descriptor check owns usage, nonzero shape, and checked byte extent.
Create, open, support, and graph compilation consume its `AccelCheck` directly;
they do not reparse an `"ok"` reason string or repeat the multiplication law.

The native resident layer likewise has one internal `ResidentDesc`, one common
`ResidentEntry` state prefix, one usage/capability predicate, one compiled
descriptor/registry validator, one descriptor-or-registry-to-`ResidentBufferRef`
projection, and one typed reject factory. Metal and Vulkan retain different
storage owners and API handles, but do not mirror the common identity,
shape, capability, and owner fields or their validation order. Backend lookup
passes that common state plus its missing-id reason to the same validator.
The compiled boundary keeps kernel binding validation and owner matching out
of backend consumer headers: changing validation semantics rebuilds one common
object rather than every resident consumer. Execution still performs the same
constant number of comparisons and one weak-owner lock; it adds no allocation,
virtual dispatch, buffer copy, or registry traversal.
CPU stores the common entry directly beside its byte storage and indexes one
weak type-erased owner by canonical resident ID. It does not retain a second
`ResidentBufferRef`, repeat checked stride/range arithmetic, or scan the
registry. Metal and Vulkan carry the single locked owner from validation into
their result instead of locking it again.

`CreateAccelBuffer(context, desc)` allocates the backing byte buffer through the
existing backend creation path and seals the resulting canonical backend facts
directly with the already admitted context. It does not call the public
`OpenAccelBuffer` path and therefore does not re-admit the context or reopen the
newly created resident handle. Another context opened from the same adapter
cannot use that `AccelBuffer`: the private typed-buffer capability retains the
original context token and backend handle, and all public context, buffer, and
shape fields must match. Transplanting a valid public owner token from another
context into `AccelContext`, `AccelBuffer`, or `AccelBuffer::buffer` does not
authenticate.
Contract verification for forged context buffers keeps `context/reject.cpp` as
a router only. `context/reject/` owns owner transplant, range overflow, forged
buffer field, and typed descriptor rejection checks separately.

`UploadAccelBuffer` and `DownloadAccelBuffer` are context-routed wrappers over
the existing buffer upload/download path. Before routing, they re-check the
context capability, typed-buffer capability, typed shape, resident descriptor,
private backend handle, and resident registry canonical extent. Batch upload
and download admit the context once, then authenticate each typed-buffer
capability and range; a batch of `K` buffers performs one context admission,
not `K`. They route
data transfer through
`UploadBuffer(context.pick, ...)` and `DownloadBuffer(context.pick, ...)` using a
canonical adapter-owned buffer view, so host/device byte counters and resident
registry ownership remain the existing buffer subsystem's evidence. Transfer
ranges are limited to the typed context-buffer byte extent. Range failures return
the existing transfer reasons `accel_buffer_upload_overflow` and
`accel_buffer_download_overflow`; ownership, token, handle, registry, shape, and
unsupported usage failures return the context reasons above.
After successful admission, a backend transfer failure keeps its first causal
reason. In particular, `compute_device_lost` from a snapshot download is never
rewritten as a context-buffer or generic transfer failure. The Compute boundary
projects known typed reasons directly and maps only unknown backend-private
reasons to its transfer boundary.

`AccelGraphBufferRef`, `AccelGraphNode`, `AccelGraph`, `AccelKernelCheck`,
`AccelKernel`, and `CompileAccelKernel(context, graph)` are repository-internal graph/kernel
compile support types. `AccelGraphNode` accepts checked `ComputeIR` facts produced
by the kernel Compute DSL for `Map` nodes, a `ScanDesc` plus matching primitive
hash for `Scan`, a compile-derived `SortDesc` plus matching primitive hash
for valid four-buffer `Sort`, a compile-derived `CompactDesc` plus matching
primitive hash for valid two-buffer `Compact`, `AccelGraphNode::gather` plus a
matching primitive hash for valid three-buffer `Gather`, `AccelGraphNode::partition`
plus a matching primitive hash for valid three-buffer `Partition`,
`AccelGraphNode::reduce` plus a matching primitive hash for valid two-buffer
`Reduce`, `AccelGraphNode::scatter` plus a matching primitive hash for valid
three-buffer `Scatter`, `AccelGraphNode::stencil` plus a matching primitive
hash for valid three-buffer `Stencil`, plus authenticated `AccelBuffer`
references. Graph kind and primitive descriptors use their sole
`rund::kernel` owners directly; Accel does not add root aliases. Node does not
accept arbitrary shader text, unvalidated callback source, backend source,
runtime storage captures, or backend-provided primitive identity as graph
input. Compile is admission only: it converts authenticated context graph
facts to kernel `Graph` descriptors, admits and parses each Map exactly once
for the context backend API, derives its `ExecutionMetadata` from that retained
parse, and defers backend-source emission until fusion fixes the final step
set. An unfused final Map emits once; every maximal legal straight-line Map
region of any admitted length `K >= 2` emits one final fused artifact. A
visibility, dependency, unsupported-operation, capacity, or non-Map boundary
terminates only that region; later legal regions remain independently
fusible. Kernel plans capacity from the retained Map's exact binding and IR
node counts before Node lowers a region, so Node never retries a smaller region
after a lowering failure. Compile stores
collective descriptor facts on private non-map steps, validates buffer context
ownership and resident byte authenticity through the same context-buffer
admission authority used by transfer routing, and validates `AccelGraphNode`
buffer role/name order for Map nodes before owner checks or logical-id
assignment. Name-disambiguation admission consumes the checked Map metadata
`read_count` and `write_count`; it must not rescan metadata roles only to
recount binding ambiguity.
`AccelGraphBufferRef::binding_name` disambiguates IR bindings; `visibility`
defaults to `External`, while `Internal` promises the ref may be omitted as a
fused intermediate and is not an ownership or authentication shortcut.
`init` is a graph fact: `Preserve` performs no first-write initialization and
`Zero` seals one exact clear immediately before that Write binding. Read+Zero,
an unknown init, and multiple Zero writers for one logical value fail compile.
Init participates in Kernel graph identity and is never supplied again by an
execution request.
Multiple-read or multiple-write map nodes must set `binding_name` to the
corresponding `ExecutionMetadata::binding_names[index]`;
otherwise names are optional, but supplied names must still match the IR
binding. Collective descriptor nodes do not consume binding names here. A
count, role-order, binding-name mismatch, mixed map/primitive identity field,
collective node carrying an IR, missing primitive hash, or zero element count
fails compile admission with `accel_kernel_graph_invalid` before execution.
Compile first reserves every explicit nonzero logical ID across the bounded
graph, then maps authenticated resident ids in first-encounter traversal order
to the lowest still-unreserved positive IDs. A later explicit ID therefore
cannot collide with an earlier implicit physical binding. Compile-local ordered
maps own resident-to-logical and logical-to-first-binding lookup, giving
`O(B log B)` admission for `B` binding occurrences while the occurrence vectors
alone retain graph order; lookup container iteration is never graph identity.
Compile then calls kernel
`ValidateGraph(...)` for semantic graph validation and deterministic graph id
derivation. Backend resident ids are authenticity/equivalence evidence only;
they are not kernel graph identity input. No backend command queue, pipeline,
descriptor set,
shader module, dispatch, upload, download, merge, or reduction happens during
`CompileAccelKernel`.

Flow Map lowering has one name projection owner for both sides of this check.
It emits `input`/`output` for a singular binding and `inputN`/`outputN` for a
plural binding. Indexed Map may carry up to 16 semantic source reads and 16
deduplicated U32 index reads, so the same owner covers all 32 physical input
ordinals. The checked IR runtime binding list and the Accel graph reference
list call that owner directly; neither reconstructs names with local arrays or
decimal formatting. This keeps artifact metadata and graph admission
identical when indexed fusion crosses the ordinary 16-input boundary.

A successful `AccelKernel` records `check`, `graph_id_hi`, `graph_id_lo`,
`node_count`, backend `api`, fixed `scalar`, frozen caps,
`context_id`, a nonzero kernel id, and one private source-only owner token. A
private deleter type seals that token's `shared_ptr` control block;
`AdmitKernelTokenWithContext(...)` recovers the typed token from the
control-block capability in `O(1)` and checks its kernel id plus the public
`AccelKernel` aggregate against the owning context token, graph id, backend
API, scalar, node count, and frozen caps. It returns that retained token
directly: null is rejection and nonnull is a completed authenticity proof,
with no parallel check or admission-token wrapper. A source-private projector
then freezes the semantic admission snapshot from the validated public kernel;
the token remains an admission support fact and the lifetime owner for private
execution vectors, not a replacement semantic authority. There is no
process-global capability table, mutex, weak entry, or owner-order scan. A
forged control block lacks the private deleter; an alias stored pointer still
fails the exact stored-pointer check. Future execution must still validate
kernel, context, and buffer ownership before dispatch. Compile and kernel
admission fail-close with:

- `accel_kernel_graph_invalid` for invalid context admission, invalid graph
  shape or numeric metadata, mismatched scalar, or kernel graph validation
  failure.
- `accel_kernel_buffer_owner_mismatch` when a graph references a foreign,
  forged, stale, or otherwise unauthenticated `AccelBuffer`.
- `accel_kernel_artifact_invalid` when checked IR cannot lower to an executable
  artifact for the frozen context backend API.

Primitive compile storage has one active-operation authority. If `D_i` and
`P_i` are descriptor and plan widths for primitive `i`, a node retains

```text
OperationBytes = max_i(sizeof(D_i) + sizeof(P_i)) + variant tag/alignment
```

instead of the sum over every inactive primitive. The admitted operation is
moved from the compile node to the final execution step; it is not copied into
a parallel descriptor table. A warm run retains only its common `ComputePlan`,
domain, artifact pointers, and dispatch windows. Primitive pass count is
projected once into `ComputePlan::dispatch_count`, so run planning has no second
kind tag, plan table, or dispatch-count switch. Owner extent and preparation
copies are physical concerns; graph order, descriptor bits, primitive hashes,
backend lowering, and numeric results are invariant.

The context support surface keeps backend SDK handles private to node
implementation files. It may expose existing `AccelDevice`, `Buffer`,
`BufferDesc`, `RuntimeStats`, resident refs, ids, graph ids, and owner tokens
as support facts, but those facts remain admission evidence rather than
semantic kernel authority. AccelKernel execution APIs must continue to use kernel
graph, fusion, lowering, binding, and dispatch contracts as semantic
authority.

Application-level solver or physics integration remains outside AccelKernel
runtime ownership.

The fake adapter is deterministic diagnostic plumbing only. It reports node
API `Fake`, freezes kernel-facing caps as `ComputeApi::Metal`, and validates Compute
dispatch windows and bindings through the kernel `ComputeBackendDispatch` function-pointer
handoff. Its runtime binding validation derives input-buffer count, per-tile
input/output byte totals, parameter byte totals, and metadata byte totals from
the `ComputePlan`, not from the binding set's self-declared values. It does not
claim real Compute execution, shader compilation, driver behavior, occupancy,
throughput, or hardware availability.

`<rund/compute.hpp>` is the only compute SDK entry. Its opaque resources and
typed operations are translated in compiled Node sources to the focused
internal owners under `node/accel/`. The checked internal DSL and raw backend
inventory remain support-only and have no aggregate include surface. The
prepared facade consumes only frozen Kernel caps,
dispatch, ownership, and a stable check reason; Kernel headers do not include
Node headers for that handoff.
