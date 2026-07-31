# Compute Collective Primitive Contracts

## Graph Edge Signature Authority

Kernel owns graph edge and result type authority through `GraphValueType`,
`GraphSignature`, `GraphSignatureFor(...)`, and `BuildMapGraphSignature(...)`.
A graph signature is derived from kernel-owned plans, checked map IR, or
execution metadata only; it is not derived from Accel factory defaults, Node
backend admission, queue state, resident storage, or driver capability.
Descriptor hashes still own primitive identity, while signatures own the
ordered read/write edge contract for a node.
`kernel/program/compute/graph/` is the sole source-tree owner for `BufferRole`,
`NodeKind`, `GraphValueType`, `GraphSignature`, signature construction, node
identity, and validation. General execution metadata does not include or expose
that graph schema. The complete graph tree is internal and absent from the
installed SDK; it is not a public router or a second schema authority.

`graph/signature.hpp` is declarations-only. The six-owner Kernel Compute
closure compiles signature construction once in `graph/signature.cpp`, which
depends on primitive models rather than executable planners. A primitive plan
or scheduling edit therefore cannot propagate through the graph schema into
unrelated primitive consumers. Each Accel factory leaf includes only its own
descriptor identity and planner; there is no aggregate primitive-node or
descriptor-hash implementation hub.

Every public graph primitive must expose the value kind, buffer role, scalar
byte width, element count, matrix rows/columns, and batch count for each edge.
`GraphValueKind::Status` records per-batch numeric status output; `Aux` records
public reusable factor side data. Internal workspace is not a public edge: for
example raw-matrix `SolveInput::Matrix` may plan LU workspace internally, but
only factor-reuse `SolveInput::Factor` exposes an `Aux` read edge. A count of
zero is reserved for dynamic map-local element counts, and an element byte
width of zero is reserved for graph-scalar byte widths resolved by Node binding
admission, such as Fourier transform split buffers.

Accel graph factories attach the kernel signature to public nodes. Node
admission must recompute the same kernel signature and reject mismatched public
factory nodes before backend dispatch. A missing signature is invalid for
production and test nodes alike; no raw-node admission path bypasses this edge
contract.
Map graph factories use `BuildMapGraphSignature(...)` so public signature
construction shares the same IR/hash/lowerable validation path as execution
metadata without building `ComputeMap`, parameter storage, or graph binding
name payloads.

## Scan Primitive

`ScanDesc`, `ScanPlan`, `ScanHash`,
`ScanResult`, `HashScan(...)`, `PlanScan(...)`,
`ReferenceExclusiveScanU32(...)`, `ReferenceExclusiveScanU64(...)`,
`ReferenceInclusiveScanU32(...)`, and `ReferenceInclusiveScanU64(...)` are the
scan collective support surface. Kernel owns deterministic planning, identity,
and CPU-reference semantics for exclusive and inclusive unsigned sum scans. It
does not call backend APIs, does not claim speedup, and does not make hardware
cache, occupancy, PMU, or timing claims.

Scan planning is pure and deterministic. It consumes only a caller-provided
`ScanDesc` and must never call node, OS, backend, Compute, filesystem, clock,
Metal, Vulkan, Foundation, driver, allocator, hardware discovery, runtime
cache, or timing APIs. The contract admits `ScanOp::ExclusiveSum` and
`ScanOp::InclusiveSum`, `ScanElement::U32`, `ScanElement::U64`,
nonzero capacity, nonzero block size, and either descriptor-owned count or a
single U32/U64 resident logical-count input. The descriptor element count is
always the admitted capacity; a buffer count is validated against that
capacity at execution and never read by the host merely to plan a GPU pass. It
computes `element_bytes`
from the element enum,
`block_count = ceil(element_count / block_size)`, `pass_count = 1` for one
block and `2` otherwise, and checked `temp_bytes = element_count *
element_bytes`.

A successful `ScanPlan` records op, element, element count, element byte
width, block size, block count, pass count, temp bytes, count source,
`ok = true`, and reason `ok`. The plan is shape and semantic evidence consumed
by node backend execution; it is not an OS, driver, timing, or adapter token.
`HashScan(...)` derives the primitive descriptor hash from op, element,
capacity, block size, and count source only. A resident-count
scan therefore cannot alias the identity of a descriptor-count scan.

The CPU reference helpers traverse elements in ascending index order and emit
exclusive or inclusive prefix sums plus the final total. U32 helpers fail when
the exact total cannot be represented by the `u32` output domain. U64 helpers
fail on `u64` addition overflow. Callers may only consume output buffers when
the returned result is ok; rejected reference results are failure evidence, not
a semantic output.

Scan primitive tests follow the owner-local contract shape: `scan.cpp` is only
the runner, while `scan/` owns planner rejection, deterministic identity, plan
shape, and CPU reference semantics.

Scan rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan value | `compute_scan_invalid` |
| Unknown scan operation | `compute_scan_op_unsupported` |
| Unknown logical-count source | `compute_scan_count_source_unsupported` |
| Unknown element width | `compute_scan_element_unsupported` |
| Zero element count | `compute_scan_count_zero` |
| Zero block size | `compute_scan_block_invalid` |
| Temp byte arithmetic overflow | `compute_scan_temp_overflow` |
| Missing CPU reference input, output, or total pointer | `compute_scan_buffer_invalid` |
| CPU reference exact sum cannot fit the declared element domain | `compute_scan_sum_overflow` |

## SegmentedScan Primitive

See [Compute Segmented Scan](./segmented/scan.md) for kernel-owned descriptor,
hash, planner, reference, rejection, and non-backend scope.

## SegmentedReduce Primitive

See [Compute Segmented Reduce](./segmented/reduce.md) for kernel-owned
descriptor, hash, planner, reference, rejection, and non-backend scope.

## Fixed Numeric Contract

Matrix, Transform, Factor, Solve, and Spectrum descriptors carry one complete
fixed format: sign-inclusive integer bits, fraction bits, rounding, overflow,
and approximation. Integer and fraction bits must both be nonzero and their sum
must equal the 32- or 64-bit element storage width. Every descriptor hash mixes
all five fields. Every successful plan copies the format, and
`*PlanMatchesDesc(...)` rejects a plan whose format differs from its descriptor.
Pointer identity, call order, backend state, and temporary ids are not identity
inputs.

Fixed primitive execution is integer-only. Multiplication first forms the
widened product, applies the descriptor rounding law while restoring the
declared binary point, and then applies `Saturate` or modulo-`2^W` `Wrap` at
the stored width `W`. Division uses a widened scaled numerator. Square root
uses deterministic integer square root over the scaled radicand. Each stored
primitive step applies the same declared rounding and overflow law on CPU,
Metal and Vulkan. This primitive storage-stage law is distinct from the
Map expression law: Map expressions retain derived precision until the
caller-authored `quantize<T>()` node.

Matrix accepts `Exact` or `Deterministic` approximation because its operations
are exact integer algorithms. Transform, Factor, Solve, and Spectrum require
`Deterministic`; supplying `Exact` is rejected with the operation's
`compute_*_numeric_policy_unsupported` reason. No backend may silently replace
or downgrade an approximation policy.

## Transform Primitive

`TransformDesc`, `TransformPlan`, `TransformHash`, `TransformResult`,
`HashTransform(...)`, `PlanTransform(...)`, and
`ReferenceFourierSplit{I32,I64}(...)` are the kernel-owned descriptor,
identity, planning, and CPU-reference surface for Fourier transforms. The
public semantic is a transform graph primitive, not an FFT-named API; FFT is an
implementation strategy. Kernel does not select a backend and does not hide a
fallback.

The contract admits `TransformOp::Fourier`, forward or inverse direction,
`TransformLayout::Split` or `TransformLayout::Interleaved`,
`TransformNorm::None`, `InverseLength`, or `Unitary`, and a nonzero
power-of-two `element_count`. Non-power-of-two transform descriptors fail
closed as `compute_transform_count_not_power_of_two`.

The split CPU reference traverses a deterministic radix-2 Cooley-Tukey plan
over real and imaginary scalar buffers. It consumes only descriptor facts and
caller buffers; descriptor hash construction uses op, direction,
layout, normalization, element count, and the complete fixed format. Backend runtime state, resident
storage authenticity, queues, timing, driver facts, and fallback policy are not
hash inputs.

One canonical radix schedule owns CPU and native accelerator ordering. It has
no element-count threshold, autotuned variant, or alternate small-workload
algorithm. A 256-lane local dispatch cooperatively loads up to 256
bit-reversed values and executes the first `min(8, log2(N))` stages. The
largest 64-bit complex block occupies 4 KiB of shared memory. Remaining radix-2 stages are
consumed in adjacent pairs while the intermediate value remains in registers;
an unpaired final stage retains the same fixed-point operation order. Thus the
transform dispatch count is

```text
1 + ceil(max(log2(N) - 8, 0) / 2).
```

The planned twiddle table has `N / 2` entries per component, stored as a cosine
half followed by a sine half. For `N > 1`, its resident workspace is
`N * element_bytes`; `N = 1` needs no bytes. The table is generated once from
the fixed numeric policy and retained with the resident execution owner, so a
warm run performs no trigonometric generation, allocation, upload, or
download. `None`, `InverseLength`, and `Unitary` normalization use planned
divisors `1`, `N`, and `floor(sqrt(N))`; the integer square root has a fixed
32-step bit bound for u64 input and is never evaluated per GPU lane.

Transform rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_transform_invalid` |
| Unknown transform operation | `compute_transform_op_unsupported` |
| Unknown direction | `compute_transform_direction_unsupported` |
| Unknown layout | `compute_transform_layout_unsupported` |
| Unknown normalization | `compute_transform_norm_unsupported` |
| Invalid fixed format or approximation policy | `compute_transform_numeric_policy_unsupported` |
| Zero or non-power-of-two element count | `compute_transform_count_not_power_of_two` |
| Missing CPU reference input or output pointer | `compute_transform_buffer_invalid` |

## Matrix Primitive

`MatrixShape`, `MatrixDesc`, `MatrixPlan`, `MatrixHash`, `MatrixResult`,
`HashMatrix(...)`, `PlanMatrix(...)`, and
`ReferenceMatrix{Mul,Transpose}{I32,I64}(...)` are the kernel-owned descriptor,
identity, planning, and CPU-reference surface for generic matrix algebra graph
primitives. Small fixed-size formula helpers remain in the map-local DSL
`mat(...)` family; buffer/layout matrix work belongs to graph primitives.

The contract admits `MatrixOp::Mul`, `MatrixOp::Transpose`, and
`MatrixOp::BatchMul`, row-major or column-major layout, element widths of four
or eight bytes, `SignedWrap`, `UnsignedWrap`, or `Fixed` arithmetic, and
nonzero dimensions. Integer multiplication and accumulation wrap at the
declared element width. Fixed multiplication uses the declared `(I, F)` binary
point, rounding, and overflow policy at every stored accumulation step. `Mul`
consumes left, right, and output buffers for one batch; `BatchMul` applies the
same shape across `batch_count`; `Transpose` consumes one input and one output.
`Solve`,
determinant, eigen, and SVD are intentionally outside this primitive because
they require separate factorization and numeric policy authority.

The CPU reference helpers traverse output elements in deterministic row/batch
order. The arithmetic law is part of the descriptor hash; integer bit patterns
are never evaluated with the fixed-point law. Descriptor hash construction
uses op, layout, arithmetic law, rows, cols, inner, batch count,
element width, and the complete fixed format.

Matrix rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_matrix_invalid` |
| Unknown matrix operation | `compute_matrix_op_unsupported` |
| Unknown layout | `compute_matrix_layout_unsupported` |
| Unknown arithmetic law | `compute_matrix_arithmetic_unsupported` |
| Unsupported element width | `compute_matrix_element_unsupported` |
| Invalid fixed format or numeric policy | `compute_matrix_numeric_policy_unsupported` |
| Zero dimensions or missing multiply inner dimension | `compute_matrix_shape_zero` |
| Element-count arithmetic overflow | `compute_matrix_shape_overflow` |
| Missing CPU reference input or output pointer | `compute_matrix_buffer_invalid` |

## Factor Primitive

`FactorShape`, `FactorDesc`, `FactorPlan`, `FactorHash`, `FactorResult`,
`HashFactor(...)`, `PlanFactor(...)`, and `ReferenceFactor{I32,I64}(...)` are
the kernel-owned descriptor, identity, planning, and CPU-reference surface for
direct factorization graph primitives. `Matrix` remains dense movement/algebra
only; LU, QR, and Cholesky are owned here.

The contract admits `FactorOp::LU`, `FactorOp::QR`, and
`FactorOp::Cholesky`, row-major or column-major layout, `Packed` or
`Separate` output, element widths of four or eight bytes, and nonzero
dimensions up to 16 rows and 16 columns. LU and Cholesky require square input.
Cholesky and QR use
`PivotOp::None`; LU may use `PivotOp::Partial`. QR `Separate` output stores
`Q|R` so solve reuse can consume the reusable factor payload.

Data-dependent numeric failures do not reject dispatch. The reference writes a
per-batch status word and returns `ok = true` with `failed_batches`,
`first_failed_batch`, and `first_status` populated for singular, non-SPD,
pivot-underflow, or invalid-scaling batches.

CPU-reference scratch initialization follows overwrite-before-read ownership.
Cholesky clears its factor output once and reads caller input separately; it
does not copy input into storage that the clear immediately overwrites.
Cholesky therefore performs zero dead source reads and destination writes for
the `B * n * n * S`-byte factor extent. QR writes every `Q` lane and temporary
projection-vector lane before reading it, so pre-clear traffic for those
`2 * rows * cols * S` bytes per batch is zero. QR reads only the upper
triangle of `R`, and the public materialization boundary writes the observable
strict-lower triangle directly as zero, so pre-clear traffic for the
`cols * cols * S`-byte `R` workspace is also zero. These are structural traffic
models, not measured wall-clock speedups.

Factor rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_factor_invalid` |
| Unknown factor operation | `compute_factor_op_unsupported` |
| Unknown layout | `compute_factor_layout_unsupported` |
| Unknown output shape | `compute_factor_output_unsupported` |
| Unknown pivot policy | `compute_factor_pivot_unsupported` |
| Unsupported element width | `compute_factor_element_unsupported` |
| Invalid fixed format or approximation policy | `compute_factor_numeric_policy_unsupported` |
| Zero dimensions | `compute_factor_shape_zero` |
| Rows or columns exceed the 16-by-16 tile-local numeric algebra dimension | `compute_factor_shape_dimension` |
| Non-square LU or Cholesky input | `compute_factor_shape_square` |
| Element-count arithmetic overflow | `compute_factor_shape_overflow` |
| Missing CPU reference input, factor, aux, or status pointer | `compute_factor_buffer_invalid` |

## Solve Primitive

`SolveShape`, `SolveDesc`, `SolvePlan`, `SolveHash`, `SolveResult`,
`HashSolve(...)`, `PlanSolve(...)`, and `ReferenceSolve{I32,I64}(...)` are
the kernel-owned descriptor, identity, planning, and CPU-reference surface for
linear-system solve graph primitives. `SolveInput::Matrix` factors raw matrix
input inside the primitive. `SolveInput::Factor` consumes reusable factor
payloads produced by the factor owner.

The contract admits `SolveOp::Linear`, raw matrix or reusable factor input,
`FactorOp::LU`, `FactorOp::QR`, or `FactorOp::Cholesky`, row-major or
column-major layout, element widths of four or eight bytes, nonzero rows,
nonzero RHS columns, nonzero batch count, and up to 16 rows by 16 RHS columns.
QR solve uses a separate `Q|R` factor payload; LU uses the pivot aux stream;
Cholesky and QR require
`PivotOp::None`.

Data-dependent numeric failures are runtime status evidence, not admission
rejection. Singular, non-SPD, pivot-underflow, and invalid-scaling batches are
reported through the status output and result summary while dispatch itself
remains complete.

QR solve computes every `Y = Q^T * B` scratch lane before back substitution
reads it. The CPU reference therefore does not pre-clear `Y`, and dead-store
traffic for its `rows * rhs_cols * S` bytes per batch is zero. The row,
right-hand-side, and reduction order is unchanged, so fixed-point rounding and
saturation remain bit-identical. This is a structural traffic model, not a
measured wall-clock speedup.

For matrix-input QR solve, the CPU reference consumes the reusable row-major
`Q` and upper-triangular `R` workspaces directly. It does not materialize and
then reread a layout-converted `Q|R` payload. For `B` square `n`-by-`n`
batches, the solve workspace excludes a `2 * B * n * n * S`-byte factor
payload and its `4 * B * n * n * S` bytes of read/write materialization
traffic. RHS and output indexing follow the caller's row-major or column-major
layout, while Q/R scratch indexing is internal row-major. Batch order,
ascending reductions, failure status, and first-failure selection are fixed.

Solve rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_solve_invalid` |
| Unknown solve operation | `compute_solve_op_unsupported` |
| Unknown input mode | `compute_solve_input_unsupported` |
| Unknown factor operation | `compute_solve_factor_unsupported` |
| Unknown layout | `compute_solve_layout_unsupported` |
| Unknown pivot policy | `compute_solve_pivot_unsupported` |
| Unsupported element width | `compute_solve_element_unsupported` |
| Invalid fixed format or approximation policy | `compute_solve_numeric_policy_unsupported` |
| Zero dimensions | `compute_solve_shape_zero` |
| Rows or RHS columns exceed the 16-by-16 tile-local numeric algebra dimension | `compute_solve_shape_dimension` |
| Element-count arithmetic overflow | `compute_solve_shape_overflow` |
| Missing CPU reference matrix/factor, aux, RHS, output, or status pointer | `compute_solve_buffer_invalid` |

## Spectrum Primitive

`SpectrumShape`, `SpectrumDesc`, `SpectrumPlan`, `SpectrumHash`,
`SpectrumResult`, `HashSpectrum(...)`, `PlanSpectrum(...)`, and
`ReferenceSpectrum{I32,I64}(...)` are the kernel-owned descriptor, identity,
planning, and CPU-reference surface for spectral graph primitives. SVD and
Eigen are primitive operations, not backends; backend choice remains node
authority.

The contract admits `SpectrumOp::SVD` and `SpectrumOp::Eigen`,
`SymmetricReal` or `GeneralReal` domains, `None`, `ValuesOnly`, `Thin`, or
`Full` vector policies, row-major or column-major layout, element widths of
four or eight bytes, nonzero dimensions, nonzero batch count, and nonzero max
iteration count, up to 16 rows and 16 columns. Eigen requires
square `SymmetricReal` input. SVD uses `A^T A` symmetric reference planning for
deterministic singular value status/value evidence.

The CPU reference fills Eigen scratch and SVD `A^T A` scratch directly. No
pre-clear is valid because every live lane is overwritten; this keeps scratch
initialization traffic at zero for `n * n * S` bytes per batch with dimension
`n` and lane width `S`. SVD computes the upper triangle of the symmetric
`A^T A` matrix and
mirrors each off-diagonal result. Fixed multiplication is commutative for the
paired operands, and every dot product retains the same ascending-row
accumulation order. For an `r`-by-`n` input, dot-product multiply-add work is
exactly `n * (n + 1) * r / 2`; all `n * n` output stores remain. These
are structural operation and traffic models, not measured wall-clock speedups.

SVD vector construction first classifies at most `c` singular columns. When
all columns are positive and above epsilon, every column writes all `r` lanes
from `A*V/sigma`, so the complete `r*c` pre-clear is skipped and removes
`r*c*S` store bytes per batch. If any column needs deterministic basis
completion, the CPU reference retains one contiguous clear and the original
diagonal basis write; this avoids replacing a cache-friendly clear with
strided full-column initialization. Column, row, and orthogonalization order
are unchanged.

Non-convergence and invalid scaling are runtime status evidence. They populate
the status output and result summary; backend admission, resident binding
validation, and execution scheduling remain outside kernel authority.

Spectrum rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_spectrum_invalid` |
| Unknown spectrum operation | `compute_spectrum_op_unsupported` |
| Unknown spectrum domain | `compute_spectrum_domain_unsupported` |
| Unknown vector policy | `compute_spectrum_vectors_unsupported` |
| Unknown layout | `compute_spectrum_layout_unsupported` |
| Unsupported element width | `compute_spectrum_element_unsupported` |
| Invalid fixed format or approximation policy | `compute_spectrum_numeric_policy_unsupported` |
| Zero dimensions or iteration count | `compute_spectrum_shape_zero` |
| Rows or columns exceed the 16-by-16 tile-local numeric algebra dimension | `compute_spectrum_shape_dimension` |
| Unsupported domain/shape pair | `compute_spectrum_shape_unsupported` |
| Element-count arithmetic overflow | `compute_spectrum_shape_overflow` |
| Missing CPU reference input, values, status, operation-required vector output, or SVD-only order/vector workspace pointer | `compute_spectrum_buffer_invalid` |

## Stencil Primitive

`StencilDesc`, `StencilPlan`, `StencilHash`,
`StencilResult`, `HashStencil(...)`,
`PlanStencil(...)`, and `ReferenceStencil{Sum,Min,Max}{U32,U64}(...)` are the kernel-only planning, identity, and
CPU-reference contract for deterministic one-dimensional clamp-boundary
window sum/extrema work. It does not execute backend work, does not mutate
resident Compute buffers, does not call backend APIs, and does not make
hardware cache, occupancy, PMU, timing, or speedup claims.

Stencil planning is pure and deterministic. It consumes only a
caller-provided `StencilDesc` and must never call node, OS, backend,
Compute, filesystem, clock, Metal, Vulkan, Foundation, driver, allocator,
hardware discovery, runtime cache, or timing APIs. The contract admits only
`StencilOp::Sum`, `StencilOp::Min`, `StencilOp::Max`, `StencilBoundary::Clamp`,
`StencilElement::U32`, `StencilElement::U64`, nonzero element
count, and radius values in `[1, element_count]`. This is a contiguous
1D shape contract: each output index reads the center element plus the clamped
left/right neighbors at distances `1..radius` from the same input stream, then
writes exactly its own output slot.

A successful `StencilPlan` records op, element enum, boundary enum,
element count, element byte width, radius, input bytes, output bytes, zero
temp bytes, pass count `1`, `ok = true`, and reason `ok`.
Input/output byte counts are checked in u64 before admission.
`HashStencil(...)` derives the primitive descriptor hash from
op, element enum, boundary enum, element count, and radius. Runtime
data values, backend selection, resident storage authenticity, pipeline caches,
command queues, measured timings, and adapter state are not hash inputs.

The CPU reference helpers traverse output indices in ascending order. `Sum`
starts from the center value and adds every clamped left/right value at
distances `1..radius`; `Min` and `Max` apply the same window and keep the
selected unsigned extremum. U32 and U64 arithmetic uses the native unsigned
wrap law of the declared element domain for `Sum`; extrema compare the declared
unsigned values. There is no floating-point, saturation, reduction, atomics, or
cross-output dependency. Callers may only consume output buffers when the
returned result is ok; rejected reference results are failure evidence, not a
semantic output.

Stencil primitive tests follow the owner-local contract shape: `stencil.cpp`
is only the runner, while `stencil/` owns planner rejection, deterministic
identity, plan shape, and CPU reference semantics.

Stencil rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_stencil_invalid` |
| Unknown stencil operation | `compute_stencil_op_unsupported` |
| Unknown boundary policy | `compute_stencil_boundary_unsupported` |
| Unknown element width | `compute_stencil_element_unsupported` |
| Zero element count | `compute_stencil_count_zero` |
| Zero or out-of-range radius | `compute_stencil_radius_invalid` |
| Input/output byte arithmetic overflow | `compute_stencil_bytes_overflow` |
| Missing CPU reference input or output pointer | `compute_stencil_buffer_invalid` |

## Sort Primitive

`SortDesc`, `SortPlan`, `SortHash`,
`SortResult`, `HashSort(...)`, `PlanSort(...)`,
`ReferenceStableSortU32(...)`, and `ReferenceStableSortU64(...)` are the
kernel-only planning, identity, and CPU-reference contract for deterministic
stable ascending key/value radix sort. It does not execute backend
work, does not mutate resident Compute buffers, does not call backend APIs, and
does not make hardware cache, occupancy, PMU, timing, or speedup claims.

Sort planning is pure and deterministic. It consumes only a caller-provided
`SortDesc` and must never call node, OS, backend, Compute, filesystem, clock,
Metal, Vulkan, Foundation, driver, allocator, hardware discovery, runtime
cache, or timing APIs. The contract admits only stable ascending sort,
`radix_bits == 8`, `SortKey::U32` or `SortKey::U64`,
`SortValue::U32` or `SortValue::IdentityU32`, and nonzero element
count. `SortValue::IdentityU32` means the sorted value stream is the
original u32 input index and no input value buffer is part of the descriptor
meaning. `key_bits == 0` means the full key width. A nonzero
`key_bits` is a declared key-domain contract, not a runtime data scan; it
admits only `16`, `32`, `48`, or `64` bits that fit the selected key enum.
It computes `key_bytes` from the key enum, `value_bytes = 4`,
`bucket_count = 256`, and `radix_pass_count = key_bits / radix_bits`.

A successful `SortPlan` records key/value enums, element count, key/value
byte widths, radix width, effective key bits, pass count, bucket count, temp
key bytes, temp value bytes, temp count bytes, temp rank bytes, total temp
bytes, `stable = true`, `ok = true`, and reason `ok`. Temp storage is
checked as one temporary key array, one temporary u32 value array, one u32 rank
per element, and u32 bucket counters for each bucket in each radix pass:
`temp_key_bytes = element_count * key_bytes`,
`temp_value_bytes = element_count * 4`,
`temp_rank_bytes = element_count * 4`, and
`temp_count_bytes = bucket_count * radix_pass_count * 4`. Every multiply and
sum is checked in u64 before `temp_bytes` is admitted.
`IdentityU32` still carries a temporary u32 value array across radix passes;
only the source value stream is generated from the original input index.
Backend-private implementations may allocate a narrower proven rank
representation, or no pass-local rank storage when the backend recomputes the
same stable block-local rank at scatter time, only as node-owned execution
evidence. This does not change the kernel sort plan, descriptor hash, radix
target index, or stable-order contract.

`HashSort(...)` derives the primitive descriptor hash from key
enum, value enum, element count, radix width, declared key bits, and stable flag
only. The same descriptor facts produce the same hash regardless of backend runtime state,
resident storage authenticity, pipeline caches, command queues, measured
timings, or adapter selection.

The CPU reference helpers traverse input indices in ascending order and insert
each `(key, value, original_index)` into ascending key order, preserving input
order for equal keys. `ReferenceStableSortU32(...)` sorts u32 keys with u32
values; `ReferenceStableSortU64(...)` sorts u64 keys with u32 values. Callers
may only consume output buffers when the returned result is ok; rejected
reference results are failure evidence, not a semantic output.

Sort rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_sort_invalid` |
| Unstable sort request | `compute_sort_stability_required` |
| Unknown key width | `compute_sort_key_unsupported` |
| Unknown value width | `compute_sort_value_unsupported` |
| Zero element count | `compute_sort_count_zero` |
| Radix width other than 8 bits | `compute_sort_radix_invalid` |
| Unsupported declared key bit domain | `compute_sort_key_bits_invalid` |
| Temp byte arithmetic overflow | `compute_sort_temp_overflow` |
| Missing CPU reference input keys, input values, output keys, output values, or output original-index buffer | `compute_sort_buffer_invalid` |

## Compact Primitive

`CompactDesc`, `CompactPlan`, `CompactHash`,
`CompactResult`, `HashCompact(...)`,
`PlanCompact(...)`, and `ReferenceCompactIdsU32(...)` are the kernel-only
planning, identity, and CPU-reference contract for deterministic stream
compaction of u32 input indices selected by u32 flags. It does not execute
backend work, does not mutate resident Compute buffers, does not call
backend APIs, and does not make hardware cache, occupancy, PMU, timing, or
speedup claims.

Compact planning is pure and deterministic. It consumes only a
caller-provided `CompactDesc` and must never call node, OS, backend, Compute,
filesystem, clock, Metal, Vulkan, Foundation, driver, allocator, hardware
discovery, runtime cache, or timing APIs. The contract admits only nonzero
`element_count`, nonzero `output_capacity`, `flag_bytes == 4`, and
`output_bytes == 4`. It represents u32 flags and u32 output ids. Widths
outside that pair fail closed as `compute_compact_invalid`.

A successful `CompactPlan` records element count, output capacity, flag byte
width, output byte width, scan temp bytes, status bytes, total temp bytes, pass
count, `ok = true`, and reason `ok`. Temp storage is
checked in u64 as `scan_temp_bytes = element_count * 4`,
`status_bytes = 4` only when `output_capacity < element_count`, otherwise
`status_bytes = 0`, and `temp_bytes = scan_temp_bytes + status_bytes`; every
multiply and sum is checked before admission. Full-capacity compact cannot
overflow output capacity because `selected_count <= element_count`. The shape
has `pass_count = 2`, representing flag scan plus scatter.

`HashCompact(...)` derives the primitive descriptor hash from element count,
output capacity, flag byte width, and output byte width
only. The same descriptor facts produce the same hash regardless of backend
runtime state, resident storage authenticity, pipeline caches, command queues,
measured timings, or adapter selection.

The CPU reference helper traverses flags in ascending input-index order and
writes the input index as a u32 output id for every nonzero flag. The semantic
output is therefore the ascending list of selected input indices. The descriptor
alone cannot know selected count, so output-capacity insufficiency is a
reference-time rejection. Callers may only consume output buffers when the
returned result is ok; rejected reference results are failure evidence, not a
semantic output. If a selected input index cannot be represented as a u32 id,
the reference fails closed as `compute_compact_temp_overflow`.

Compact primitive tests follow the owner-local contract shape: `compact.cpp` is
only the runner, while `compact/` owns planner rejection, deterministic
identity, plan shape, and CPU reference semantics.

Compact rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value, unsupported flag byte width, or unsupported output byte width | `compute_compact_invalid` |
| Zero element count | `compute_compact_count_zero` |
| Zero output capacity | `compute_compact_capacity_zero` |
| CPU reference selected flag count exceeds output capacity | `compute_compact_capacity_insufficient` |
| Temp byte arithmetic overflow, or selected CPU reference index exceeds u32 output id range | `compute_compact_temp_overflow` |
| Missing CPU reference flags, output, or output count pointer | `compute_compact_buffer_invalid` |

## Partition Primitive

See [Compute Partition](./partition.md) for the kernel-owned partition
descriptor, hash, planner, reference, rejection, and non-backend scope.

## Gather Primitive

`GatherDesc`, `GatherPlan`, `GatherHash`,
`GatherResult`, `HashGather(...)`, `PlanGather(...)`,
`ReferenceGatherU32(...)`, and `ReferenceGatherU64(...)` are the kernel-only
planning, identity, and CPU-reference contract for deterministic indexed
gather. It does not execute backend work, does not mutate resident Compute
buffers, does not call backend APIs, and does not make hardware cache,
occupancy, PMU, timing, or speedup claims.

Gather planning is pure and deterministic. It consumes only a
caller-provided `GatherDesc` and must never call node, OS, backend, Compute,
filesystem, clock, Metal, Vulkan, Foundation, driver, allocator, hardware
discovery, runtime cache, or timing APIs. The contract admits only
`GatherElement::U32` and `GatherElement::U64`, nonzero `element_count`,
nonzero `source_count`, and u32-addressable source indices. Gather indices are
always u32; duplicate indices are allowed because every output element writes
exactly its own destination slot. `count_source` is either the descriptor
capacity or one resident U32/U64 logical-count buffer; the latter must be at
most `element_count` before any output is published.

A successful `GatherPlan` records element enum, output element count,
source element count, element byte width, index byte width, status byte width,
temp bytes, pass count, count source, `ok = true`, and reason `ok`. The plan has
`index_bytes = 4`, `status_bytes = 8`, `temp_bytes = 24`, and
`pass_count = 2`. The two status words carry the typed reason and first
rejected ordinal; the remaining 16 bytes are the validated indirect-dispatch
record. The first pass validates the entire active prefix and only the second
pass writes gathered values, so an invalid count or index leaves the output
byte-for-byte unchanged. This status is diagnostic fail-closed evidence for
out-of-range indices; it is not a semantic reduction, output accumulator, or
ordering authority. Every multiply is checked in u64 before admission.

`HashGather(...)` derives the primitive descriptor hash from element enum,
element count, source count, and count source only. The same descriptor facts
produce the same hash regardless of backend runtime state, resident storage
authenticity, pipeline caches, command queues, measured timings, or adapter
selection.

The CPU reference helpers first validate every active index in ascending order,
then traverse that same prefix once more to write
`output[i] = values[indices[i]]`. This deliberate two-pass boundary preserves
the complete caller output on any invalid index; no valid prefix is published
before failure is known. Callers may only consume output buffers when the
returned result is ok. The first out-of-range index is reported as
`first_invalid_index`.

Gather primitive tests follow the owner-local contract shape: `gather.cpp` is
only the runner, while `gather/` owns planner rejection, deterministic
identity, plan shape, and CPU reference semantics.

Gather rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_gather_invalid` |
| Unknown element width | `compute_gather_element_unsupported` |
| Unknown resident count source | `compute_gather_count_source_unsupported` |
| Resident logical count exceeds descriptor capacity | `compute_bounded_count_invalid` |
| Zero output element count | `compute_gather_count_zero` |
| Input capacity not U32-ordinal-addressable | `compute_gather_count_unsupported` |
| Zero source element count | `compute_gather_source_zero` |
| Source count not u32-addressable | `compute_gather_source_unsupported` |
| Output/index byte arithmetic overflow | `compute_gather_temp_overflow` |
| Missing CPU reference values, indices, or output pointer | `compute_gather_buffer_invalid` |
| CPU reference or backend observed an index outside source count | `compute_gather_index_out_of_range` |

## Histogram Primitive

`HistogramDesc`, `HistogramPlan`,
`HistogramHash`, `HistogramResult`,
`HashHistogram(...)`, `PlanHistogram(...)`, and
`ReferenceHistogramU32(...)` are the kernel-only planning, identity, and
CPU-reference contract for deterministic generic bin counting. Histogram owns
cross-element accumulation from a bin-index buffer into a count buffer; it does
not own domain-specific bucket formulas, particle grids, spatial indexing, or
map-local bin computation.

Histogram planning is pure and deterministic. It consumes only a
caller-provided `HistogramDesc` and must never call node, OS, backend,
Compute, filesystem, clock, Metal, Vulkan, Foundation, driver, allocator,
hardware discovery, runtime cache, or timing APIs. The contract admits only
`HistogramIndex::U32` and `HistogramCount::U32`, nonzero
`element_count`, nonzero `bin_count`, and u32-addressable counts.

A successful `HistogramPlan` records index/count enums, input element
count, output bin count, index/count byte widths, input/output byte extents,
status bytes, temp bytes, pass count, `ok = true`, and reason `ok`. The plan
has `index_bytes = 4`, `count_bytes = 4`,
`status_bytes = 4`, `temp_bytes = 4`, and `pass_count = 2`, representing clear
then count. The status word is diagnostic fail-closed evidence for invalid bin
indices; it is not a semantic output or scheduling authority.

`HashHistogram(...)` derives the primitive descriptor hash from index enum,
count enum, element count, and bin count only. The same
descriptor facts produce the same hash regardless of backend runtime state,
resident storage authenticity, pipeline caches, command queues, measured
timings, or adapter selection.

The CPU reference helper clears every output bin before traversing input indices
in ascending order and incrementing `counts[bin]`. Callers may only consume the
count buffer when the returned result is ok; rejected reference results are
failure evidence, not a semantic output. An input bin index greater than or
equal to `bin_count` is rejected as `compute_histogram_bin_invalid`.

Histogram primitive tests follow the owner-local contract shape:
`histogram.cpp` is only the runner, while `histogram/` owns planner rejection,
deterministic identity, plan shape, and CPU reference semantics.

Histogram rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_histogram_invalid` |
| Unknown bin-index width | `compute_histogram_index_unsupported` |
| Unknown count width | `compute_histogram_count_unsupported` |
| Zero input element count | `compute_histogram_count_zero` |
| Zero output bin count | `compute_histogram_bin_count_zero` |
| U32 count output cannot represent the input element count | `compute_histogram_count_overflow` |
| Input/output byte arithmetic overflow | `compute_histogram_bytes_overflow` |
| Missing CPU reference input or output pointer | `compute_histogram_buffer_invalid` |
| CPU reference or backend observed a bin index outside bin count | `compute_histogram_bin_invalid` |

## Scatter Primitive

`ScatterDesc`, `ScatterPlan`, `ScatterHash`,
`ScatterResult`, `HashScatter(...)`,
`PlanScatter(...)`, `ReferenceScatterU32(...)`, and
`ReferenceScatterU64(...)` are the kernel-only planning, identity, and
CPU-reference contract for deterministic limited indexed scatter. It does not
execute backend work, does not mutate resident Compute buffers, does not
call backend APIs, and does not make hardware cache, occupancy, PMU, timing,
or speedup claims.

Scatter planning is pure and deterministic. It consumes only a
caller-provided `ScatterDesc` and must never call node, OS, backend, Compute,
filesystem, clock, Metal, Vulkan, Foundation, driver, allocator, hardware
discovery, runtime cache, or timing APIs. The contract admits only
`ScatterElement::U32` and `ScatterElement::U64`, nonzero
`element_count`, nonzero `output_count`, and u32-addressable output indices.
Scatter indices are always u32. The canonical encoded-owner table reserves one
bit of each u32 entry, so `element_count <= 2^31 - 1`; the output status table
contains one extra slot, so `output_count <= 2^32 - 1`. These are planner
constraints rather than backend-local admission mirrors.

Scatter is limited: `output[indices[i]] = values[i]`, but duplicate
target indices are rejected before any CPU-reference output is committed.
Duplicate writes are not ordered by backend schedule, lane order, atomics, or
last-writer-wins behavior. This keeps the semantic authority independent of
physical execution timing. Output slots not referenced by any input index are
left unchanged by the CPU reference.

A successful `ScatterPlan` records element enum, input element count, output
element count, element byte width, index byte width, status byte width, temp
bytes, pass count, `ok = true`, and reason `ok`. The plan has
`index_bytes = 4`, `status_bytes = (output_count + 1) * 4`,
`temp_bytes = status_bytes`, and `pass_count = 1`. The status words are
diagnostic fail-closed evidence for out-of-range and duplicate indices; they
are not semantic accumulators or write-order authority. Every add and multiply
is checked in u64 before admission.

`HashScatter(...)` derives the primitive descriptor hash from element enum,
element count, and output count only. The same descriptor facts
produce the same hash regardless of backend runtime state, resident storage
authenticity, pipeline caches, command queues, measured timings, or adapter
selection.

The CPU reference helpers validate all indices in ascending input-index order
before writing any output element. The first out-of-range or duplicate input
index is reported as `first_rejected_index`. Callers may only consume output
buffers when the returned result is ok; rejected reference results are failure
evidence, not a semantic output.

Scatter primitive tests follow the owner-local contract shape: `scatter.cpp` is
only the runner, while `scatter/` owns planner rejection, deterministic
identity, plan shape, and CPU reference semantics.

Scatter rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_scatter_invalid` |
| Unknown element width | `compute_scatter_element_unsupported` |
| Zero input element count | `compute_scatter_count_zero` |
| Input count exceeds the u32 owner encoding | `compute_scatter_count_unsupported` |
| Zero output element count | `compute_scatter_output_zero` |
| Output count not u32-addressable | `compute_scatter_output_unsupported` |

## Scatter Reduce Primitive

`ScatterReduceDesc`, `ScatterReducePlan`, `ScatterReduceHash`,
`ScatterReduceResult`, `PlanScatterReduce(...)`, `HashScatterReduce(...)`, and
the typed `ReferenceScatterReduce*` functions own deterministic conflicting
indexed writes. The binding order is values Read, U32 indices Read, optional
U32/U64 logical-count Read, and output Write. `element_count` is physical input
capacity and `output_count` is the fixed result extent.

The semantic order is stable `(target, source ordinal)`. Integer Sum wraps at
32 or 64 bits. Fixed Sum narrows after every addition with the descriptor's
exact overflow policy. Fixed saturation is not associative; for example
`MAX, +10, -20` produces `MAX - 20`, not the result of a reassociated tree.
Min and Max compare in the descriptor's signed, unsigned, or Fixed domain.
Empty logical input is valid and publishes the operation identity for every
output: zero for Sum, numeric maximum for Min, and numeric minimum for Max.

The native physical plan has three dispatches: one complete source preflight
that writes two indirect commands, one parallel output/count-table identity
initialization, and one fold. The preflight is one 256-lane workgroup: lane
`t` scans source ordinals `t, t + 256, ...`, followed by an eight-stage shared
minimum reduction. Its critical path is `O(ceil(N/256) + 8)` and its total work
is `O(N)`. It adds no dispatch, scratch allocation, payload copy, or command
submission to the three-pass plan. Count overflow takes precedence and skips
all index reads; otherwise the shared minimum publishes the exact first
invalid ordinal independently of lane scheduling. The control lane publishes
zero indirect x dimensions on failure and a zero-width fold for zero logical
count. Planning caps input capacity at `UINT32_MAX`, matching the U32
first-error ordinal and indirect-command ABI. Vulkan uses an explicit
final-stride guard, so its U32 loop cannot wrap. On 32-bit carriers, wrapping
Sum and signed/unsigned/Fixed Min/Max use
one hardware atomic lane per active source. Addition modulo `2^32`, Min, and
Max are associative and commutative, so arrival order cannot change the exact
result; contributor-count atomics derive the same conflict total independently
of schedule. Fixed saturating Sum alone retains the strict source-ordinal fold
because per-add saturation is non-associative. 64-bit carriers retain that
source-order path because the supported Metal/Vulkan capability set does not
promise portable 64-bit storage atomics. Valid input is visited once, physical
work is `O(N + O)`, device scratch is `4*O + 16 + 24` bytes, and there are no
copied key/value arrays. `radix_pass_count` is zero, `fold_pass_count` is one,
and `pass_count` is three. The plan is also the sole memory ABI authority:
`segment_bytes = 4*O`, `status_bytes = 16`, `indirect_bytes = 24`, and
`temp_bytes` is their checked sum. Metal and Vulkan allocate from these fields
rather than repeating byte constants. `ScatterReduceFoldParallel(plan)` is the
single ordering-policy predicate: it admits exactly associative 32-bit cases
and keeps Fixed saturating Sum and every 64-bit carrier on the stable ordinal
fold. Identity includes operation, domain, complete Fixed policy, capacity,
output count, and count source.

Before any output write, execution validates `logical_count <= capacity` and
then every active target in source order. The first invalid ordinal is stable.
An oversized count or invalid target leaves the whole output unchanged; no
identity prefix or partial fold is published. The GPU validation status and
CPU reference enforce the same law before arithmetic work. Native atomics are
a physical implementation only for the associative 32-bit cases above;
last-writer-wins and completion order are never semantic authorities.

The CPU reference receives a caller-owned U32 index scratch with at least
`logical_count` entries. Prepared execution allocates it once and retains it;
warm execution performs no heap allocation. After the failure-atomic source
preflight, it sorts only the copied target indices and derives the conflict
count from adjacent equal keys, then folds the original values once in source
order. The CPU bound is `O(N log N + N + O)` for `N` active values and `O`
outputs. Sorting is used only for conflict detection; observable folding
remains one source-order pass, so conflict detection cannot change fold order
or failure publication.

| Gate | Stable reason |
| --- | --- |
| Unsupported operation | `compute_scatter_reduce_op_unsupported` |
| Unsupported numeric domain | `compute_scatter_reduce_domain_unsupported` |
| Invalid/unexpected Fixed policy | `compute_scatter_reduce_fixed_invalid` / `compute_scatter_reduce_fixed_unexpected` |
| Zero physical capacity/output extent | `compute_scatter_reduce_count_zero` / `compute_scatter_reduce_output_zero` |
| Input capacity not U32-ordinal-addressable | `compute_scatter_reduce_count_unsupported` |
| Unsupported output/count source | `compute_scatter_reduce_output_unsupported` / `compute_scatter_reduce_count_source_unsupported` |
| Temp byte arithmetic overflow | `compute_scatter_reduce_temp_overflow` |
| Resident count above capacity | `compute_scatter_reduce_count_out_of_range` |
| Active target outside output | `compute_scatter_reduce_index_out_of_range` |
| Missing active buffer | `compute_scatter_reduce_buffer_invalid` |
| Missing CPU reference values, indices, or output pointer | `compute_scatter_buffer_invalid` |
| CPU reference observed an index outside output count | `compute_scatter_index_out_of_range` |
| CPU reference observed a duplicate output index | `compute_scatter_duplicate_index` |

## Reduce Primitive

`ReduceDesc`, `ReducePlan`, `ReduceHash`,
`ReduceResult`, `HashReduce(...)`, `PlanReduce(...)`,
`ReferenceReduceSumU32(...)`, `ReferenceReduceSumU64(...)`,
`ReferenceReduceCountNonzeroU32(...)`, `ReferenceReduceCountNonzeroU64(...)`,
`ReferenceReduceMinU32(...)`, `ReferenceReduceMinU64(...)`,
`ReferenceReduceMaxU32(...)`, and `ReferenceReduceMaxU64(...)` are the
kernel-only planning, identity, and CPU-reference contract for deterministic
unsigned reduction. It does not
execute backend work, does not
mutate resident Compute buffers, does not call backend APIs, and does not make
hardware cache, occupancy, PMU, timing, or speedup claims.

Reduce planning is pure and deterministic. It consumes only a
caller-provided `ReduceDesc` and must never call node, OS, backend, Compute,
filesystem, clock, Metal, Vulkan, Foundation, driver, allocator, hardware
discovery, runtime cache, or timing APIs. The contract admits only
`ReduceOp::Sum`, `ReduceOp::CountNonzero`, `ReduceOp::Min`, `ReduceOp::Max`,
`ReduceElement::U32`, `ReduceElement::U64`, nonzero element
capacity, a descriptor-owned count or single U32/U64 resident logical-count
input, and a reduction block size that converges. The descriptor count remains
the capacity used for scratch admission; execution validates the resident
logical count before consuming values. `block_size == 1` is admitted
only for a single input element; multi-element reductions require at least two
inputs per tree node so the fixed tree reaches one output.

A successful `ReducePlan` records op enum, element enum, input element
count, element byte width, block size, items per thread, first-pass group
count, pass count, partial element count and width, partial bytes, status
bytes, total temp bytes, count source, `ok = true`, and reason `ok`.

`Sum` and `CountNonzero` use exact 128-bit partials. For capacity `N` and
block size `B`, planning fixes eight input items per thread and

```text
G = min(128, ceil(N / (8 B))).
```

One group completes directly in one pass. Otherwise `G` independent groups
write `G` 16-byte partials and one final group combines them, so pass count is
two, `partial_element_count = G`, and `partial_bytes = 16 G`. Capping `G`
does not discard work: each first-pass thread advances by the full grid width
until it covers every canonical input index.

The stored partial remains 128-bit, but the first-pass input loop does not pay
for a two-word add when a narrower batch has a proved bound. `CountNonzero`
counts once in a private u64 because a lane can observe no more than
`2^64 - 1` inputs. U32 Sum batches at most 256 values before sign- or
zero-extending the batch into the exact two-word accumulator. Its unsigned
batch is less than `256 (2^32 - 1) < 2^40`; its signed magnitude is at most
`256 * 2^31 = 2^39`, so neither private batch can overflow. U64 Sum retains a
two-word add per input. These batching rules change work, not mathematical
order authority: every input contributes exactly once to the associative
addition modulo `2^128`.

`Min` and `Max` keep one input per thread and the fixed tree that repeatedly
replaces `current_count` with `ceil(current_count / B)` until one output
remains. Their partial element width equals the stored element width and their
partial count is the sum of non-final intermediate counts. All operations use
`status_bytes = 4` and `temp_bytes = partial_bytes + status_bytes`; every
admission add and multiply is checked in u64.

`HashReduce(...)` derives the primitive fingerprint from op enum, element
enum, capacity, block size, count source, and the immutable eight-item,
128-group, 16-byte-wide, 256-item narrow-batch planning policy. The same facts
produce the same hash
regardless of backend runtime state, resident storage authenticity, pipeline
caches, command queues, measured timings, or adapter selection. Changing a
physical planning policy therefore cannot reuse an artifact fingerprint from
a different policy.

The CPU reference helpers traverse input elements in ascending index order.
`Sum` writes the exact unsigned sum when it fits the declared output domain.
`CountNonzero` writes the exact count of input elements whose value is not zero,
using the same output domain as the declared element width. U32 helpers fail
when the exact sum or count cannot be represented by the u32 output domain. U64
sum helpers fail on u64 addition overflow; u64 nonzero counts fail only if the
count accumulator would overflow. `Min` and `Max` traverse the same ascending
index order and write the exact unsigned minimum or maximum; comparison does
not use arithmetic overflow status. Callers may only consume output buffers
when the returned result is ok; rejected reference results are failure
evidence, not a semantic output.

Reduce rejection and reference failure reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_reduce_invalid` |
| Unknown reduce operation | `compute_reduce_op_unsupported` |
| Unknown logical-count source | `compute_reduce_count_source_unsupported` |
| Unknown element width | `compute_reduce_element_unsupported` |
| Zero element count | `compute_reduce_count_zero` |
| Zero or non-converging block size | `compute_reduce_block_invalid` |
| Partial/status byte arithmetic overflow | `compute_reduce_temp_overflow` |
| Missing CPU reference input or output pointer | `compute_reduce_buffer_invalid` |
| Sum cannot fit the declared output domain | `compute_reduce_sum_overflow` |
| Nonzero count cannot fit the declared output domain | `compute_reduce_count_overflow` |

Reduce primitive tests follow the owner-local contract shape: `reduce.cpp` is
only the runner, while `reduce/` owns planner rejection, deterministic
identity, plan shape, and CPU reference semantics.
