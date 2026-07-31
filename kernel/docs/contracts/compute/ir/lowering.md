# Compute IR And Lowering Contract

## Fusion Planning

`FusionPolicy`, `FusionPlan`, and `PlanFusion(...)` are the
SDK-free kernel map-chain fusion legality surface. Fusion planning is pure and
deterministic: it consumes only a valid `Graph` descriptor and frozen
caller-provided fusion policy facts. It never calls node, OS, backend, Compute,
filesystem, clock, Metal, Vulkan, Foundation, driver, or runtime cache APIs.
Its cold workspace uses ordinary host allocation proportional to the admitted
graph node count; allocation failure rejects the plan with
`compute_fusion_capacity` before any backend lowering.

Dependency responsibility is explicit and graph-local. The planner derives
fusion candidate dependencies from ordered `GraphNode` buffer refs: an
adjacent boundary `node[i] -> node[i + 1]` is a candidate only when `node[i]`
writes a logical buffer that `node[i + 1]` reads. The candidate remains legal
only when the producer boundary is a single-intermediate map chain: `node[i]`
has exactly one write, `node[i + 1]` has exactly one read of that written
logical buffer, no producer write is declared CPU-visible in `FusionPolicy`,
that written logical buffer has exactly one downstream reader before the next
writer, and both adjacent nodes are marked fusion-supported by their aligned
policy facts. Non-adjacent dependencies are not fused;
preserving graph node order is the execution-shape authority.

Fusion rejection for a candidate edge does not reject the graph. A valid graph
with visibility boundaries, branch consumers, unsupported operations, or no
candidate edges remains an ok unfused plan. Only invalid graph
descriptors or invalid fusion policy pointer/count facts fail the whole plan.
If any candidate boundary is rejected, `reason` records the first rejection
cause even when another boundary fuses. When no candidate boundary is fused or
rejected, `output_graph_id_hi/lo` equals the input graph id. When one or more
adjacent boundaries fuse or reject, the output id is a deterministic
fusion-plan identity derived from the input graph id, frozen policy facts,
original node count, fused node count, rejected edge count, and the ordered
boundary decision stream containing boundary index, decision class,
intermediate logical id, rejection count, and fused flag. It is not a backend
pipeline-cache key and does not prove hardware residency or performance.

Fusion policy has exactly one `FusionNodePolicy` fact per graph node, in graph
order. `supported` states that the already-admitted retained IR for that node
can participate in a fused map region. `writes_visible` states that at least
one producer write crosses the external visibility boundary. `binding_count`
and `ir_node_count` are the exact retained-IR costs used to keep every planned
region within `kMaxComputeBindingCount == 64` and
`kMaxComputeNodeCount == 1024`. A supported node has nonzero bounded costs; an
unsupported node has zero costs. Operation hashes
remain graph identity facts and are not duplicated in the policy. The policy
count must exactly equal the graph node count and is bounded by
`kMaxFusionPolicyNodeCount == kMaxGraphNodeCount == 16384` before any pointer
walk. This outer Program bound is independent of the unchanged 1024-node
per-Map IR bound.

For a Fixed graph, Node marks a Map supported only when its complete fixed
format equals the graph format: integer bits, fraction bits, rounding,
overflow, and approximation must all match. A mixed-policy branch therefore
stays in one graph but creates an unsupported fusion boundary before lowering.
Lowering never retries a smaller region after discovering a policy mismatch,
and fusion never erases a caller-authored quantization law by carrying a value
across two different stored formats.

`PlanFusion(...)` is the sole boundary-decision authority. It records fused
boundaries in a fixed 16,384-bit plan that the Node compiler consumes directly;
Node does not reevaluate fusion legality. Candidate reader lifetimes are built
in two cold vectors sized to the actual admitted node count, sorted by
`(logical_id, writer)`, and counted by one ordered graph sweep with reads
preceding writes at each node. With `N <= 16384` nodes, `B <= 64` buffers per
node, and `R <= N * B` buffer refs, the planner is bounded by
`O(N * B^2 + R log N)` work and `O(N)` cold workspace. It performs no
per-boundary future-graph scan and retains no workspace after planning.
It also carries the current region's binding and IR-node totals. Fusing one
producer/consumer boundary removes one Write binding, one Read binding, one
Write node, and one Read node, so adjoining costs `(b, v)` updates the region
as `(B + b - 2, V + v - 2)`. A boundary that would exceed either canonical
limit starts the next maximal region; lowering never discovers capacity by
failing an already-selected whole region.

A fused region has one intermediate contract and one terminal contract.
Every nonterminal Map has exactly one Write because that Write is the sole
value substituted into the next Map. The terminal Map may have one or more
ordered Writes. Lowering preserves every terminal Write node, binding, graph
buffer reference, and metadata row in canonical declaration order. The
terminal outputs do not become additional fusion intermediates, and Node
projects all of them into the one fused execution step without rebuilding a
different graph.

Fusion planning reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan value | `compute_fusion_invalid` |
| Cold graph-sized planning workspace cannot be allocated | `compute_fusion_capacity` |
| Valid graph has no rejected candidates | `compute_fusion_ok` |
| Candidate boundary is not a single-intermediate map chain, or the intermediate has multiple downstream readers before the next writer | `compute_fusion_dependency_conflict` |
| Candidate producer write is declared CPU-visible | `compute_fusion_visibility_boundary` |
| Either adjacent node is not fusion-supported, is not a Map, or has a mismatched element count | `compute_fusion_unsupported_op` |
| Fusing the next Map would exceed canonical binding or IR-node capacity | `compute_fusion_capacity_boundary` |
| Policy count differs from graph node count, exceeds the graph bound, has a nonzero count with a null pointer, or carries noncanonical per-node costs | `compute_fusion_policy_invalid` |
## IR And DSL

`ComputeIR` and `rund::compute_dsl::bind(...)/def(...).map(...)` are the checked internal operation
identity surface feeding `ComputeMap`. The DSL invokes the mapper with symbolic
values, records declared params, reads, writes, arithmetic nodes, and
`write_handle[i] = value`, then discards the mapper. It is not arbitrary C++
skeleton callback lowering, a callable registry, a shader-string API, or a
surface for captured runtime storage.

Runtime storage is admitted only through declared `.param`, `.read`, and
`.write` bindings. `buffer<T>()` is an IR-only placeholder for binding
declarations; facade runtime execution with unresolved placeholder storage
fails closed. The public DSL write surface is `out[i]` for a declared write
handle; special carrier modes are internal contracts described below. Scatter,
atomics, reductions, and write-combine modes require separate contracts.

DSL/IR validation is part of the contract:

| Gate | Stable reason |
| --- | --- |
| Default/invalid IR | `compute_ir_invalid` |
| Duplicate binding name in one binding kind | `compute_binding_duplicate` |
| Fixed-mode floating parameter | `compute_param_float_unsupported` |
| Mapper emits no write | `compute_write_missing` |
| Forged/missing binding id | `compute_binding_invalid` or `compute_binding_missing` |
| Invalid symbolic value or context mismatch | `compute_value_invalid` or `compute_expression_context_mismatch` |
| Opcode requested outside its admitted scalar domain, including unsigned-order or arithmetic-shift mismatches | `compute_value_invalid` |
| IR node budget exceeded | `compute_ir_too_large` |
| Canonical-IR parse storage cannot be allocated or exceeds a standard container limit | `compute_ir_capacity` |
| Write index is not the symbolic tile index | `compute_write_index_unsupported` |
| Runtime placeholder storage reaches facade binding | `compute_binding_runtime_missing` |
| Forged opcode outside its admitted scalar domain, or a noncanonical unsigned-order opcode | `compute_ir_node_invalid` |

The authoritative DSL has no floating-point scalar selector. Its only scalar
selectors are `i32()`, `u32()`, `i64()`, `u64()`, and
`fixed<IntegerBits, FractionBits, ...>()`; floating parameters are rejected
before mapper invocation and cannot be converted into a fixed expression.
`bind()` starts in an unspecified mode, so omitting an explicit selector is
rejected with `compute_scalar_unsupported`; it never implies a fixed format.

`FixedOnlyOp`, `IrOpDomainValid`, and `CanonicalIrOpForDomain` are the shared
opcode-domain authority used by checked DSL construction, Node expression
lowering, and backend-independent forged-IR validation. The admitted canonical
IR table is:

| Opcode family | Admitted canonical IR domains |
| --- | --- |
| `Quantize`, `NegPositiveFixed`, `MulFixed`, `MulFixedScaled`, `MulUnsignedFixed`, `MulAddFixed`, `DivFixed`, `Recip`, `Sqrt`, `Rsqrt`, `Sin`, `Cos`, `Tan`, `Exp`, `Log`, `Atan2` | `Fixed` only |
| `Abs`, `AbsMagnitude`, `Sign`, `AddSat`, `SubSat`, `ShrArithmeticConst` | `I32`, `I64`, `Fixed` |
| `AddSatUnsigned` | `U32`, `U64`, `Fixed` |
| `Min`, `Max`, `Clamp`, `Lt`, `Le`, `Gt`, `Ge` | `I32`, `I64`, `Fixed` |
| `MinUnsigned`, `MaxUnsigned`, `ClampUnsigned`, `LtUnsigned`, `LeUnsigned`, `GtUnsigned`, `GeUnsigned` | `U32`, `U64` |
| `DivSigned` | `I32`, `I64` |
| `DivUnsigned` | `U32`, `U64` |
| `Neg`, `ShlConst`, `ShrLogicalConst`, equality, bitwise, wrap arithmetic, predicate, and selection operations | all five domains subject to their shape and storage rules |

The checked DSL accepts the domain-generic `min`/`max`/`clamp` and ordered
comparison surface in an unsigned mode, then canonicalizes it to the matching
`*Unsigned` opcode before storage checks, format derivation, append, and graph
fingerprinting. Node uses the same canonicalizer. Consequently an unsigned
graph has one opcode sequence and one fingerprint; forged `U32`/`U64` IR that
retains a base signed-order opcode is rejected as noncanonical. The
`*Unsigned` order opcodes are not Fixed storage overrides: no Fixed backend
lowering owns them, so Fixed DSL and forged Fixed IR reject them. The sole
unsigned-storage Fixed saturation override is `AddSatUnsigned`.

The common validator applies this table and canonicality check before CPU
instruction planning or Metal or Vulkan source generation. The constant
shift builder applies the same gate, so unsigned arithmetic right shift cannot
bypass domain admission.

Integer domain is a value provenance, not a graph-header shortcut. In a
same-width heterogeneous Map, parameter/read/write bindings seed `I32`, `U32`,
`I64`, or `U64` independently. Domain-restricted canonical opcodes add further
constraints, and the common validator resolves constants and intermediate
values with two reverse and two forward bounded node sweeps. Unknown,
domain-neutral values receive the header domain only between the two sweep
pairs. The work is therefore bounded by four linear passes over at most the IR
node limit; it does not rescan every producer for every consumer.

`MergeComputeDomains` is the sole integer merge authority. Equal domains stay
equal; equal signedness selects the wider lane; `I32 + U32 -> U32`,
`I64 + U32 -> I64`, and every merge with `U64` selects `U64`. Fixed and integer
provenance never merge. A general `Select` merges only its true and false
values, never its predicate. The exact `Select(P, 1, 0)` shape inherits `P`'s
domain only when an ordinary `Value` Write to an unsigned target proves that
the Select is the canonical mask owner; a signed result, dead Select, or
malformed mixed arm cannot use that inference. An ordinary `Value` Write requires exact
source/binding domain agreement, except for the pre-existing public canonical
unsigned mask width bridge described below.

`Write.rhs` is the canonical `IrWriteMode` field and has exactly three admitted
values. `Value == 0` is the ordinary public DSL store. The other two values are
internal Flow materialization contracts, not public implicit conversions:

- `CheckedOrdinal == 1` admits only an opposite-signedness, same-width integer
  target. Signed-to-unsigned is exactly
  `Select(Ge(X, 0), X, 0)`; unsigned-to-signed is exactly
  `Select(LeUnsigned(X, INT_MAX_target), X, 0)`. Representable raw values pass
  unchanged and an out-of-range value is normalized to raw zero. The caller's
  Flow guard, not this total storage conversion, owns any semantic overflow
  failure.
- `BoundaryMask == 2` is exactly `Select(Ne(X, 0), 1, 0)`. `X` is an integer
  value and this special mode admits only a same-width signed integer or Fixed
  target. A Fixed target carries that binding's exact I/F, rounding, overflow,
  and approximation policy on the Write. When the target is `U32` or `U64`,
  the same normalized result delegates to the existing ordinary canonical
  unsigned-mask `Value` owner, including its admitted 32/64-bit width bridge.
  The internal request and public mask therefore produce identical canonical
  bytes and hash, a `Value` Write, and absent Fixed metadata. This keeps one
  serialization for identical zero/one semantics while signed and Fixed
  `BoundaryMask` targets remain same-width only. The result is raw zero or raw
  one; it is not numeric integer-to-Fixed scaling.

In both shapes, `X` may be any already-valid pure integer expression.
`CheckedOrdinal` requires its comparison operand and selected true arm to
reference the same canonical `X`; `BoundaryMask` requires that `X` be the left
operand of the exact `Ne(X, 0)` predicate while the selected true arm remains
the raw-one constant. Map-chain substitution or fusion may replace a Read with
an expression but may not duplicate, reinterpret, or reorder the carrier.
Unknown modes, direct cross-domain producers, malformed normalization shapes,
and mode/shape swaps fail before backend source emission. Same-domain Flow
retyping is an identity and emits no carrier Write. The mode is serialized in
the canonical node bytes, therefore it participates in the IR op hash, graph
fingerprint, and program cache identity without a parallel versioned format.

The graph header and binding domains obey the same family boundary. A Fixed
header admits only same-lane Fixed parameters and reads, and only Fixed ordinary
writes. An integer header admits no Fixed parameter or read; same-lane
signed/unsigned heterogeneity remains valid. Its only Fixed write is the exact
same-width `BoundaryMask` raw-zero/raw-one contract above. The independent
public unsigned canonical mask Write retains its admitted width bridge. Both
the checked builder and serialized common validator enforce these exceptions
before backend lowering, so changing only a binding mode cannot forge a
Fixed/integer conversion.

Constant and Index nodes remain compact, untyped serialized values. The
checked builder nevertheless records their effective domain, uses a typed or
operation anchor when constructing them, and includes that domain in local
common-subexpression identity. Equal raw bits used once as signed and once as
unsigned therefore remain distinct nodes. Validation reconstructs the same
constraints from canonical opcodes and bindings. CPU and accelerator lowering
then interpret base ordered operations as signed (or Fixed) regardless of the
header binding order; only the explicit `*Unsigned` variants use unsigned
ordering.

`ReadAt` is the sole dynamic indexed-read opcode. Its `aux` field names the
source read binding, `lhs` names the U32 index read binding, and `rhs` stores
the nonzero source element count. The source and index are ordinary checked
bindings in canonical read order. Validation rejects a non-read source, a
non-U32 index binding, equal source/index bindings, or a zero source count.
CPU, Metal, and Vulkan compute the same byte address

```text
binding_base(source) + zero_extend(index) * stride(source)
```

only after the execution owner has proved `index < source_count` for every
active logical lane. This preserves the exact bits of a materialized Gather
without granting the IR an unchecked pointer or backend-specific indexing
rule. Removing a materialized Gather additionally requires equality between
the Gather validation domain and the indexed Map execution domain. Exact to
bounded fusion is rejected; bounded fusion requires the same count authority.
The first invalid logical ordinal is the minimum failing member of that common
domain and remains part of execution evidence. Direct reads and writes use the
same address owner with the logical lane as their index:

```text
binding_base(binding) + logical_index * stride(binding)
```

Canonical Metal and Vulkan source declares one base and one stride constant
per read/write binding. Runtime view specialization changes only those
declarations; it never searches or rewrites individual address-expression
shapes. Direct reads, `ReadAt` source/index reads, ordinary writes, and wide
writes therefore cannot diverge when a native descriptor requires an aligned
base plus a byte bias. `ReadAt`, its source count, and both binding ordinals
are serialized and therefore participate in operation hash, graph fingerprint,
program cache,
and artifact identity.

Generated integer-division helpers follow reachable canonical nodes, not the
graph-header domain. Metal and Vulkan inspect `DivSigned` and
`DivUnsigned` independently and select only the graph scalar width. A
heterogeneous graph may therefore define and call both signed and unsigned
helpers even when its first binding has only one of those domains. A graph
retains its reachable unsigned-magnitude helper dependency. Source contracts
cover both header directions at 32 and 64 bits, and Node runtime contracts run
the same signed/unsigned quotient goldens on CPU, Metal, and Vulkan.

The canonical identity is the serialized `rund.compute.ir` bytes plus op
hash. Every parameter, read, and write binding records an explicit canonical
numeric mode (`I32`, `U32`, `I64`, `U64`, 32-bit `Fixed<I, F>`, or 64-bit `Fixed<I, F>`) in addition
to element width. Consequently equal-width signed, unsigned, and fixed lanes
cannot collide. Diagnostic names and C++ field tags remain outside semantic
identity. Ordered multi-write Map nodes are valid; each write stays explicit
and each output owns a distinct SoA binding.

Public symbolic values, handles, mapper access objects, and write targets are valid
only during mapper execution; escaped or forged handles must not mutate a new
build context.
During checked DSL construction the canonical builder performs deterministic
local common-subexpression elimination for non-write IR nodes by matching the
exact `(op, lhs, rhs, aux, fixed format, effective value domain)` tuple already
emitted in the same build context.
This is a build-time canonicalization rule, not a backend optimization hint.
It preserves first occurrence order, returns the first matching node id, and
keeps every `Write` node explicit so write observation and mapper admission
remain unchanged. Composite DSL helpers such as projection and rejection
therefore share repeated fixed subexpressions in the serialized IR without
changing tile identity, binding authority, lowering policy, or backend result
bits.

Authoritative Compute results are fixed-point only. The contract admits 32-bit
and 64-bit storage through the single `Fixed<I, F>` public value template.
Supported fixed maps close through checked executable source or instruction-plan
artifacts for Metal, Vulkan, and CPU. Fixed `add`, `sub`, and `mul` retain
their derived precision in IR: addition and subtraction use
`I=max(Ia,Ib)+1, F=max(Fa,Fb)`, while multiplication uses
`I=Ia+Ib, F=Fa+Fb`. No backend may narrow those results before an explicit
`Quantize` node. The fixed IR subset also includes widened unary `neg`, `abs`,
unsigned-lane `abs_magnitude`,
`sign`, signed `min`, `max`, `clamp`, `select`, `eq`, `ne`, `lt`, `le`, `gt`,
`ge`, predicate `not`, `and`, and `or`, storage bitwise `bit_and`, `bit_or`,
`bit_xor`, `bit_not`, and checked constant shifts `shl_const<N>`,
`shr_logical_const<N>`, and `shr_arithmetic_const<N>`, saturating
`add_sat`, unsigned-storage `add_sat_unsigned`, `sub_sat`,
`neg_positive_fixed`, `mul_fixed`, `mul_fixed_scaled`, and
`mul_unsigned_fixed`, with
predicate values normalized to fixed scalar `0` or `1`.
`mul_add_fixed(lhs, rhs, addend)` retains the full product scale and aligns the
addend without an intermediate store. Composite DSL primitives
are not new IR opcodes. Format-aware constants preserve the anchor's declared
`I/F`, rounding, and overflow metadata while literal nodes remain `Exact`.
`fixed_one(value)` is nearest-even numeric one at raw `1 << F`, saturating to
the signed lane maximum only when `I == 1`; `fixed_max(value)` remains the
distinct greatest signed storage value. `saturate(value)` lowers to
`clamp(value, 0, fixed_one(value))`. `step(edge, value)` lowers to
`select(value < edge, 0, fixed_one(value))`. `lerp(lhs, rhs, amount)` clamps
`amount` to `[0, fixed_one(amount)]` and lowers to the existing
`add_sat(mul_fixed(lhs, fixed_one(amount) - amount), mul_fixed(rhs, amount))`
law. `lerp` overloads for 6-argument bilinear and 11-argument trilinear
arities lower to ordered nested `lerp` calls.
`unlerp(lo, hi, value)` lowers to
`select(hi <= lo, 0, saturate(div_fixed(value - lo, hi - lo)))`, with
saturating fixed subtraction for the differences. Thus zero or reversed input
intervals return exact fixed zero as the semantic result. `remap(in_lo, in_hi,
out_lo, out_hi, value)` lowers to `lerp(out_lo, out_hi, unlerp(...))`.
`smoothstep(edge0, edge1, value)` lowers to `fade(unlerp(edge0, edge1,
value))`. `smootherstep(edge0, edge1, value)` lowers to the quintic Bezier
smoothing law over the same normalized `t`: `10*t^3*(1-t)^2 +
5*t^4*(1-t) + t^5`, with the integer factors formed by bounded repeated
`add_sat` over non-negative terms.
`dot` overloads for fixed arities 2, 3, 4, 5, 6, and 8 lower to fixed arity
saturating sums of `mul_fixed` terms. `conv` overloads for tap counts 3, 5, 7,
and 9 are fixed map-expression convolution tap helpers that lower to the same
ordered dot laws over sample and coefficient arguments. They are Accel DSL
helpers, not a skeleton callback operation registry, graph convolution
descriptor, shared-write stencil, reduction, or layout authority. The `Axis`
tag is C++ overload selection only; it is not a ComputeIR value, runtime
branch, plane-space descriptor, or backend layout authority. `mat(Axis::X,
...)` and `mat(Axis::Y, ...)` overload on 2D or 3D matrix-vector argument
count; `mat(Axis::Z, ...)` is 3D-only.
`mat(MatOp::Determinant, ...)` overloads lower either to the 2x2 determinant
`sub_sat(mul_fixed(m00, m11), mul_fixed(m01, m10))` or to the 3x3 fixed
cofactor expansion
`m00*det(m11,m12,m21,m22) - m01*det(m10,m12,m20,m22) +
m02*det(m10,m11,m20,m21)` through ordered `mul_fixed`, `sub_sat`, and
`add_sat`. `mat(MatOp::Trace, ...)` overloads lower to saturating diagonal sums.
`mat(MatOp::Transpose, Axis::X/Y/Z, ...)` lowers to column dot products over
the same row-major inputs. `mat(MatOp::Solve, Axis::X/Y, ...)` lowers to the
2x2 determinant ratios
`mat(MatOp::Determinant, b0, m01, b1, m11) /
mat(MatOp::Determinant, m00, m01, m10, m11)` and
`mat(MatOp::Determinant, m00, b0, m10, b1) /
mat(MatOp::Determinant, m00, m01, m10, m11)` with the existing deterministic
fixed divide; a zero determinant returns exact fixed zero for each component.
These matrix helpers are fixed map expressions only; they are not graph matmul
descriptors, tensor layout authority, shared writes, reductions, iterative
solver authority, factorization authority, or
backend-specific matrix intrinsics. `aff(Axis::X, ...)` and `aff(Axis::Y,
...)` overload on 2D or 3D affine argument count and lower to the matching
matrix row helper plus translation through `add_sat`; `aff(Axis::Z, ...)` is
the 3D z-row form. These affine helpers are fixed map expressions only; they
are not graph transform
descriptors, tensor layout authority, shared writes, reductions, or
backend-specific matrix intrinsics. `mix` overloads for 2, 3, and 4 weighted
terms lower to fixed weighted sums using ordered `mul_fixed` terms and
`add_sat`; helper arguments are not normalized or clamped as weights.
`weighted_mean` overloads for the same fixed arities divide the matching
weighted sum by the saturating sum of weights with the existing guarded
fixed-divide law, so a zero aggregate weight returns exact fixed zero. `poly`
overloads for degree 2 and 3 lower to Horner forms using only `mul_fixed` and
`add_sat`, with coefficient order `c0 + x*(c1 + x*(...))`. These mix and
polynomial helpers are fixed map expressions only; they are not
interpolation-table descriptors, polynomial-fit contracts, reductions, range
aggregations, or layout authority. `lerp` overloads own the ordered linear,
bilinear, and trilinear interpolation laws. `lerp(LerpOp::Smooth, ...)`
overloads for the same arities first transform weights with `fade`, then use
the same ordered interpolation laws. `LerpOp::Smooth` is a C++ overload tag,
not a ComputeIR value, runtime branch, or second interpolation function family.
`bezier` overloads lower to the quadratic or cubic
de Casteljau nested `lerp` law. These interpolation helpers are fixed map
expressions only; they are not lattice descriptors, shared-write stencils,
interpolation-table resources, or layout authority.
`len(MetricOp::Squared, ...)` overloads for 2D and 3D lower to those same dot
laws without fixed `sqrt`. Plain `len(...)` overloads lower to deterministic
fixed `sqrt(len(MetricOp::Squared, ...))`.
`dist(MetricOp::Squared, ...)` overloads lower to squared length over per-axis
saturating fixed subtraction, so distance deltas have one checked overflow law
across backends without requiring a square-root when callers only need squared
metrics. Plain `dist(...)` overloads lower to deterministic fixed
`sqrt(dist(MetricOp::Squared, ...))`. `unit(Axis::X, ...)` and
`unit(Axis::Y, ...)` overload on 2D or 3D vector argument count, while
`unit(Axis::Z, ...)` is 3D-only; each lowers to
`select(len == 0, 0, div_fixed(component, len))`, so zero-length inputs return
exact fixed zero without introducing a backend-specific normalization law.
`absdiff(lhs, rhs)` lowers to saturating fixed `abs(sub_sat(lhs, rhs))`.
`len(Norm::L1, ...)` and `len(Norm::LInf, ...)` overload on 2D or
3D vector argument count;
`dist(Norm::L1, ...)` and `dist(Norm::LInf, ...)` overload on 2D or 3D point
pairs and apply the same laws to per-axis saturating deltas. These norm-tagged
length and distance helpers do not introduce fixed square-root, divide, or
backend-specific norm authority when the squared or absolute metric is
sufficient. `angle(AngleOp::Cosine, ...)` overloads on 2D or 3D vector pairs
and lower to `select(denom == 0, 0,
div_fixed(dot(a, b), denom))`, where `denom` is
`sqrt(mul_fixed(len(MetricOp::Squared, a), len(MetricOp::Squared, b)))`, so
zero-length inputs return fixed zero. The three- and four-input `min` and
`max` overloads lower to nested
signed fixed `min`/`max`.
`median(a, b, c)` lowers to the signed order statistic
`max(min(a, b), min(max(a, b), c))`. `spread` lowers to saturating fixed
`sub_sat(max(...), min(...))`. `clamp_range(value,
a, b)` lowers to `clamp(value, min(a, b), max(a, b))`, so unordered endpoints
still produce a deterministic sorted clamp. `in_range(value, lo, hi)` is an
ordered inclusive predicate `value >= lo && value <= hi`; it does not sort
endpoints. `out_range(...)` is `predicate_not(in_range(...))`.
`bandpass(value, lo, hi)` keeps `value` only when it lies inside the sorted
inclusive interval `[min(lo, hi), max(lo, hi)]`; `bandstop(value, lo, hi)`
zeros that same sorted interval. These range helpers use only signed fixed compare,
predicate, select, min, max, clamp, and saturating subtraction ops and return
predicate values as fixed scalar `0` or `1`. `sum` overloads for 2, 3, and 4 inputs lower to ordered
saturating fixed `add_sat` trees. `sum(SumOp::Abs, ...)` applies fixed
absolute value before the same sum law, and `sum(SumOp::Squared, ...)` applies
`mul_fixed(value, value)` before the same sum law. `SumOp::*` values are C++
overload tags only; they are not ComputeIR values, runtime branches, suffix
functions, graph reductions, scans, or second aggregate authorities.
`diff(from, to)` lowers to `sub_sat(to, from)`, while
`diff(prev, center, next)` lowers to the fixed mean of the two adjacent
first-order differences. Higher finite-difference order uses the same public
name with typed order tags: `diff(DifferenceOrder::Second, prev, center,
next)` lowers to `diff(center, next) - diff(prev, center)`, and
`diff(DifferenceOrder::Third, a, b, c, d)` lowers to the difference of adjacent
second-order terms. The order tags are compile-time overload tags, so invalid
order/arity combinations do not create a second runtime mode or backend
opcode. Absolute difference uses the direct `absdiff(from, to)` authority
rather than a second `diff(...)` selector family. Centered deltas reuse the
statistics `centered(value, center)` authority directly.
These aggregate helpers are fixed map expressions only; they are not unordered
reductions, scans, stencils, or graph collectives. `fixed(FixedOp::Half, value)`,
`fixed(FixedOp::Third, value)`, and `fixed(FixedOp::Quarter, value)` lower to
nearest-even encodings of `1/2`, `1/3`, and `1/4` at the actual declared
fraction count `F`. The constant node inherits the anchor's rounding and
overflow policy but remains `Exact`; neither scalar width nor a Q1 convention
selects its binary point. `mean(lhs, rhs)` lowers to
`add_sat(mul_fixed(lhs, fixed(FixedOp::Half, lhs)), mul_fixed(rhs, fixed(FixedOp::Half, lhs)))`.
`mean(a, b, c)` uses the nearest-even fixed-third coefficient for each term and
combines them with saturating adds. `mean(a, b, c, d)` uses exact
fixed-quarter terms and ordered pairwise saturating adds. These mean helpers
never add all inputs before scaling, so the mean laws have one checked fixed
multiply/add overflow behavior instead of an intermediate sum overflow.
`mean(MeanOp::Abs, ...)` applies fixed `abs` to each input and then uses the
matching fixed mean overload. `mean(MeanOp::Squared, ...)` applies
`mul_fixed(value, value)` to each input and then uses the matching fixed mean
overload. `MeanOp::*` values are compile-time C++ overload tags only; they are
not ComputeIR values, runtime branches, suffix helper names, graph reductions,
or second statistics authorities.
`centered(value, center)` lowers to
`sub_sat(value, center)`, `centered(CenteredOp::Abs, ...)` lowers to fixed
`abs(centered(...))`, `centered(CenteredOp::Squared, ...)` lowers to
`mul_fixed(centered(...), centered(...))`,
`centered(CenteredOp::Cubic, ...)` lowers to the ordered fixed product
`centered(value, center)^3`, and `centered(CenteredOp::Quartic, ...)` lowers
to `centered(CenteredOp::Squared, value, center)^2`. `CenteredOp::*` values are
compile-time C++ overload tags only; they are not ComputeIR values, runtime
branches, suffix helper names, graph reductions, or second centering
authorities. These fixed statistics helpers use only constants, fixed
multiply, saturating add/sub, and absolute value; they do not introduce
division, square root, floating point, or backend-specific rounding authority.
`mean(CenteredOp::Cubic, ...)` and `mean(CenteredOp::Quartic, ...)` overloads
for 2, 3, and 4 inputs compute the matching fixed mean and then return the
fixed mean of the centered powers.
`var(lhs, rhs)` computes the fixed mean with `mean`, then returns the
`mean` of the two squared differences from that mean. `var(a, b, c)` uses
the same law with `mean` and the nearest-even fixed-third coefficient.
`var(a, b, c, d)` uses the same law with exact `mean`. `rms` overloads for 2,
3, and 4 inputs take the deterministic fixed `sqrt` of
`mean(MeanOp::Squared, ...)` at the matching fixed arity. These moment helpers use the same fixed constants, saturating
subtraction/addition, fixed multiply, and fixed sqrt authority as the primitive
operators; they do not introduce floating point, division, unordered
reduction, range aggregation, or a second variance/RMS accumulator contract.
`cov` overloads for 2, 3, and 4 paired samples compute fixed arity covariance
as the fixed mean of paired centered products, using the same `mean`,
`centered`, and `mul_fixed` laws above. `corr` overloads for the same sample
counts compute `sqrt(mul_fixed(var_x, var_y))` as the denominator and return
exact fixed zero when that denominator is zero; otherwise they divide the
matching covariance by that denominator with the existing deterministic fixed divide. These
correlation helpers are fixed arity map expressions only; they do not add
floating point, unordered reduction, range aggregation, atomics, or a second
statistics accumulator contract.
`ratio(numerator, denominator)` lowers to
`select(denominator == 0, 0, div_fixed(numerator, denominator))`.
`proportion(value, total)` reuses the same guarded ratio authority and clamps
the result with `saturate`. `proportion(Axis::X/Y, x, y)` and
`proportion(Axis::X/Y/Z, x, y, z)` select one component and divide it by the
same fixed-arity `sum(...)` total before applying that guarded clamp law. These
axis overloads are fixed-arity component ratios; they do not add exponential
normalization, unordered reduction, or a range-level softmax contract.
`zscore(value, center, scale)` lowers to `ratio(centered(value, center),
scale)`, with saturating fixed subtraction for the centered delta.
`standardized(StandardizedOp::Cubic, ...)` and
`standardized(StandardizedOp::Quartic, ...)` lower to fixed powers of that
z-score using ordered `mul_fixed` terms. `mean(StandardizedOp::Cubic, ...)`
and `mean(StandardizedOp::Quartic, ...)` overloads for 2, 3, and 4 inputs
compute the fixed mean, derive `sqrt(var(...))` as the scale, then return the
fixed mean of the matching standardized powers. Thus zero variance produces
exact fixed zero for each standardized term through the same guarded division
law. These
standardization helpers are fixed arity map expressions only; they do not add
floating point, unordered reduction, range aggregation, or a second
standard-deviation accumulator contract.

Compute DSL formula ownership follows the semantic contracts
above. Statistics constants live under
`dsl/functions/stats/constants.hpp`, mean and centering laws under
`stats/mean/op.hpp`, `stats/mean/value.hpp`, `stats/mean/absolute.hpp`,
`stats/mean/squared.hpp`, centering laws under `stats/center/op.hpp`,
`stats/center/value.hpp`, `stats/center/absolute.hpp`, and
`stats/center/squared.hpp`, and
variance/RMS moments under `stats/moment.hpp`.
Centered higher moment formulas live under `functions/moment.hpp`,
`moment/power.hpp`, and `moment/mean.hpp`; `stats/higher` is not a live
formula owner.
Covariance and correlation formulas live directly under
`corr/covariance/value.hpp` and `corr/correlation/value.hpp`; correlation must
reuse the statistics authority instead of carrying its own mean or variance
law. Standardized mean-moment overloads live under
`standardize/moment/cubic/value.hpp` and
`standardize/moment/quartic/value.hpp` and reuse `standardize/scale.hpp` plus
the standardized power authority. Matrix row helpers live under
`matrix/rows/axis.hpp`, determinant expansion under `matrix/determinant.hpp`,
trace helpers under `matrix/trace.hpp`, transpose-component helpers under
`matrix/transpose/axis.hpp`, and guarded 2x2 solve components under
`matrix/solve/axis.hpp`. Geometry unit helpers live under
`geometry/unit/axis.hpp`, projection under `geometry/projection.hpp`,
cross/orientation/barycentric determinant helpers under
`geometry/determinant.hpp`, guarded geometry ratio helpers under
`geometry/ratio.hpp`, point-line parameter, projection, and distance helpers
under `geometry/line.hpp`, point-plane parameter, projection, and distance
helpers under `geometry/plane.hpp`, and rejection under
`geometry/rejection/axis.hpp`. Range order-summary helpers live under
`range/order.hpp`, where fixed arity `min`, `max`, `median`, and `spread`
overloads share signed fixed ordering authority. The group routers
`stats.hpp`, `corr.hpp`, `matrix.hpp`, `geometry.hpp`, and `range.hpp` are
include routers only. Inside the package-private DSL tree, a router exists only
when it collects two or more independent semantic leaves. A one-to-one
router is not admitted: internal consumers include the declaration-owning leaf
directly. The DSL call surface stays on same-name
overloads rather than numeric-suffix helper names.
`is_zero`, `nonzero`,
`is_neg`, `is_pos`, `is_nonneg`, and `is_nonpos` lower to the matching fixed
compare against exact fixed zero. `all` and `any` lower to nested predicate
`and`/`or`. `keep_if(value, predicate)` lowers to
`select(predicate, value, 0)`, and `zero_if(value, predicate)` lowers to
`select(predicate, 0, value)`. These direct select-mask helpers do not
introduce branches, float truth, or backend-specific predicate authority.
`near(value, target,
tol)` lowers to `absdiff(value, target) <= abs(tol)`, and `near(value, tol)`
lowers to the same predicate with an exact fixed-zero target. `deadzone(value,
tol)` lowers to `select(near(value, tol), 0, value)`, and
`snap(value, target, tol)` lowers to
`select(near(value, target, tol), target, value)`. These tolerance helpers use only fixed
absolute value, saturating subtraction, compare, and select laws.
`clip(value, limit)` lowers to
`clamp(value, neg_positive_fixed(abs(limit)), abs(limit))`.
`huber(value, delta)` lowers to
`select(abs(value) <= abs(delta), 0.5*value*value,
abs(delta) * (abs(value) - 0.5*abs(delta)))` through existing fixed-half,
fixed multiply, saturating subtraction, compare, and select laws.
`positive_part(value)` lowers to `max(value, 0)`, and
`negative_part(value)` lowers to `min(value, 0)`. These piecewise helpers use only signed fixed ordering,
absolute value, neg-positive, clamp, min, and max laws. `proj(Axis::X, ...)` and
`proj(Axis::Y, ...)` overload on 2D or 3D projection argument count, while
`proj(Axis::Z, ...)` is 3D-only; they lower to
`mul_fixed(basis_component, select(dot(basis,basis) == 0, 0,
div_fixed(dot(value,basis), dot(basis,basis))))`, so projection onto a zero
basis vector returns exact fixed zero while all nonzero basis projections use
the existing fixed divide and fixed multiply laws. `cross` lowers to
`sub_sat(mul_fixed(ax, by), mul_fixed(ay, bx))`; `cross(Axis::X/Y/Z, ...)`
lowers to the same determinant law per 3D component. `orient(a,b,c)` lowers to
`cross(b - a, c - a)` with per-axis saturating fixed subtraction before the
determinant. `bary(Axis::X/Y/Z, ...)` uses the same overload selector tag as
other component helpers, but the selector names the result tuple component, not
the input plane coordinate. The overloads lower to the signed orientation ratios
`orient(p,b,c) / orient(a,b,c)`, `orient(a,p,c) / orient(a,b,c)`, and
`orient(a,b,p) / orient(a,b,c)` with the existing deterministic fixed divide;
when `orient(a,b,c)` is zero, each returns exact fixed zero. These barycentric
helpers are fixed map expressions only; they are not topology, indexing,
table-resource, discretization, plane-space, or graph collective authority.
`triple(a,b,c)` lowers to the ordered fixed scalar triple product
`dot(a, cross(b, c))`; it does not authorize a backend to reassociate
determinant terms. `line(LineOp::Parameter, p,a,b)` lowers to
`select(len(MetricOp::Squared, b-a) == 0, 0,
div_fixed(dot(p-a, b-a), len(MetricOp::Squared, b-a)))`,
and `line(LineOp::Projection, Axis::X/Y, ...)` lowers to
`a_component + (b_component - a_component) *
line(LineOp::Parameter, ...)` with saturating fixed add/subtract and fixed
multiply. The parameter is not clamped and does not carry segment,
nearest-search, or graph topology authority.
`line(LineOp::Distance, MetricOp::Squared, p,a,b)` lowers to
`select(len(MetricOp::Squared, b-a) == 0, 0,
div_fixed(mul_fixed(cross(b-a, p-a), cross(b-a, p-a)),
len(MetricOp::Squared, b-a)))`.
Plain `line(LineOp::Distance, ...)` lowers to deterministic fixed
`sqrt(line(LineOp::Distance, MetricOp::Squared, ...))`.
These line helpers are fixed map expressions only; they are not topology,
indexing, table-resource, discretization, nearest-search, or graph collective
authority. `plane(PlaneOp::Parameter, p,a,n)` lowers to
`select(len(MetricOp::Squared, n) == 0, 0,
div_fixed(dot(p-a, n), len(MetricOp::Squared, n)))`, and
`plane(PlaneOp::Projection, Axis::X/Y/Z, ...)` lowers to
`p_component - normal_component * plane(PlaneOp::Parameter, ...)` with
saturating fixed subtraction and fixed multiply.
`plane(PlaneOp::Distance, MetricOp::Squared, p,a,n)` lowers to
`select(len(MetricOp::Squared, n) == 0, 0,
div_fixed(mul_fixed(dot(p-a,n), dot(p-a,n)),
len(MetricOp::Squared, n)))`, and plain `plane(PlaneOp::Distance, ...)`
lowers to deterministic fixed
`sqrt(plane(PlaneOp::Distance, MetricOp::Squared, ...))`. These plane helpers
are fixed map expressions only; they do not add topology, mesh, indexing,
nearest-search, or graph collective authority; a zero normal produces fixed
zero distance and no-op projection. `reject(Axis::X, ...)` and
`reject(Axis::Y, ...)` overload on 2D
or 3D projection argument count, while `reject(Axis::Z, ...)` is 3D-only; each
lowers to `sub_sat(value_component, proj(Axis, ...))`, so rejection shares the
projection zero-basis rule and uses one saturating subtraction law across
backends. `reflect(Axis, ...)` reuses the same projection component and lowers
to `sub_sat(add_sat(projected, projected), value_component)`.
`softsign(value, scale)` first raises the scale floor to
`max(abs(scale), fixed(FixedOp::Half, value))`, then lowers to
`ratio(value, scale_floor + abs(value)*fixed(FixedOp::Half, value))` through the
existing guarded ratio, absolute value, fixed multiply, and saturating add
laws. `softsign(value)` uses the fixed half scale default.
Unit-range softsign behavior is expressed by composing
`saturate(mean(softsign(...), fixed_one(...)))`; there is no separate selector
family, runtime branch, suffix helper, activation/model name, or second
softsign authority.
`activation(ActivationOp::Relu, value)` reuses the signed-part authority and
lowers to `max(value, 0)`.
`activation(ActivationOp::Relu, value, upper)` lowers to
`min(positive_part(value), positive_part(upper))`; the upper bound is an
explicit signed fixed input and is normalized by the same positive-part
authority rather than by a separate bounded-ReLU API.
`activation(ActivationOp::LeakyRelu, value, slope)` lowers to
`select(value >= 0, value, mul_fixed(value, slope))`; the slope is an explicit
fixed input and is not clamped or normalized by the helper.
`activation(ActivationOp::HardSigmoid, value)` lowers to
`saturate(mean(value, fixed_one(value)))`, a fixed piecewise-linear sigmoid
approximation that reuses the statistics mean and saturate authorities.
`activation(ActivationOp::HardSwish, value)` lowers to
`mul_fixed(value, activation(ActivationOp::HardSigmoid, value))`, reusing that
hard-sigmoid authority.
`activation(ActivationOp::HardTanh, value)` lowers to
`clip(value, fixed_one(value))`, reusing the symmetric fixed clip authority instead of
creating a second clamp law. `ActivationOp::*` values are C++ overload tags
only; they are not ComputeIR values, runtime branches, suffix functions,
model names, or second activation authorities. These activation helpers are
map-local fixed formulas only; they do not introduce floating point,
approximation tables, model-specific state, or backend-specific activation
opcodes.
`hash(value)` is a deterministic finalizer over the fixed storage bits:
fixed_lane32 uses xor-shift/multiply rounds `>>16`, `0x7feb352d`, `>>15`,
`0x846ca68b`, `>>16`; fixed_lane64 uses `>>33`, `0xff51afd7ed558ccd`,
`>>33`, `0xc4ceb9fe1a85ec53`, `>>33`. `hash(value, seed)` hashes
`value ^ seed`. `hash(HashOp::Unit, ...)` masks the matching hash with
`(1 << F) - 1`, producing a fixed non-negative value in `[0, 1)` at the
declared binary point without claiming randomness or entropy. The raw hash formula authority lives in
`functions/hash/raw/value.hpp`; fixed storage width changes the round constants
and shifts through scalar-mode dispatch, not through separate fixed-width owner
files.
`fade(amount)` clamps `amount` into `[0, fixed_one(amount)]` and lowers to the cubic
fixed interpolation law `t^2 + 2*t^2*(1-t)` expressed only with `mul_fixed`,
`sub_sat`, and `add_sat`. The public `smootherstep(...)` helper keeps the
quintic curve under interpolation smoothing, not noise or activation owners.
`window(WindowOp::Parabolic, amount)` clamps `amount` into
`[0, fixed_one(amount)]`, forms `q=t*(1-t)` through `mul_fixed`, and lowers to `4q`
through ordered `add_sat`. `window(WindowOp::Triangular, amount)` clamps `amount` into
`[0, fixed_one(amount)]`, forms
`delta=abs(t-fixed(FixedOp::Half, t))`, doubles that distance with `add_sat`, and lowers
to `fixed_one(t)-2*delta` through `sub_sat`. Hamming and Blackman coefficient
integers are canonical Q1.31 source data and are rescaled to the actual `F`
with nearest-even rounding; widening the storage lane does not reinterpret
them as Q1.63. These helpers are current
lane map-local fixed window formulas, not graph-level window/tile collectives
or domain-specific curves.
`noise(cell, amount)` and
`noise(cell, amount, seed)` are deterministic one-dimensional lattice
value-noise helpers: they hash `cell` and `cell + 1` with
`hash(HashOp::Unit, ...)`, then `lerp` those endpoints with `fade(amount)`.
The `cell + 1` step is fixed storage wrap addition, and the helper makes no
entropy or hardware RNG claim.
`noise(x, y, tx, ty)` samples the four fixed-storage lattice corners
`(x,y)`, `(x+1,y)`, `(x,y+1)`, and `(x+1,y+1)` with
`hash(HashOp::Unit, x, y)` style hashing, interpolates across x with
`fade(tx)`, then interpolates those rows with `fade(ty)`.
`noise(x, y, tx, ty, seed)` uses `y ^ seed` and
`(y+1) ^ seed` as the second hash coordinate. The `+1` operations are fixed
storage wrap additions; `noise` is deterministic value-noise, not randomness,
gradient noise, atomics, or backend RNG.
The three stored fixed multiply opcodes use the operands' actual declared
fraction count `F`, never a scalar-width or Q1 binary point. For raw lane bits
`x` and `y`, `mul_fixed` rounds `signed(x) * signed(y) / 2^F` and applies the
declared signed overflow law; `mul_fixed_scaled` rounds
`signed(x) * unsigned(y) / 2^F` and applies the same signed overflow law; and
`mul_unsigned_fixed` rounds `unsigned(x) * unsigned(y) / 2^F`, applies the
declared unsigned bounds, and returns those unsigned result bits in the same
lane width. In all three cases the selected rounding law runs before
`Saturate` or `Wrap`. `mul_add_fixed(x, y, z)` instead retains
`signed(x) * signed(y) + signed(z) * 2^F` at the product scale, aligns the
addend there, and rounds and narrows only at the caller's terminal
`quantize<T>()`; no product store is inserted before the add. Constant shifts
reject at DSL/IR construction when `N` is greater than or equal to the scalar
bit width.
Fixed nonlinear arithmetic admits `div_fixed`, `recip`, `sqrt`, `rsqrt`,
`sin`, `cos`, `tan`, `exp`, `log`, and `atan2` for every declared 32-bit or
64-bit `(I,F)` storage format. Storage-only nonlinear operands are normalized
by a caller-authored explicit `Quantize` node before the operation; the builder
does not convert a widened operand and instead rejects it with
`compute_ir_quantize_required`. These operators lower
only to deterministic integer helper calls. Division by zero returns the exact
math fixed contract value, 32-bit division uses widened unsigned long division,
64-bit division uses explicit limb long division over the widened numerator,
sqrt uses deterministic integer sqrt over the widened radicand, and rsqrt is
the checked composition of sqrt and reciprocal. Trigonometric input is
converted from the declared fraction count into the canonical turn phase,
canonical integer approximation output is quantized back to the declared
format, and `tan` divides the same canonical sine/cosine pair. `exp` and `log`
convert the declared input to the canonical saturated unit representation,
execute the deterministic integer approximation, and quantize its output.
`atan2` converts the canonical full-turn phase output directly to the declared
fraction count. Every nonlinear, division, and `atan2` result records the
`Deterministic` approximation policy in IR; it is never mislabeled `Exact`.
Backend intrinsic sqrt,
reciprocal/rsqrt authority, floating conversion, fast-math authority, shader
128-bit integer types, runtime stats, driver state, OS APIs, timers, PMU
counters, hardware discovery, reductions, atomics, scatter writes, shared
writes, and arbitrary shader source remain outside the admitted operation set.
Dynamic shift operators are not admitted; they remain absent rather than
implemented as rejection-only placeholders. Identity-only or unavailable
Vulkan paths may remain only as rejected or unsupported-path evidence; they do
not satisfy supported Vulkan map execution.

### DSL physical ownership

`include/kernel/program/compute/dsl.hpp` is the product facade, not an
implementation owner. Binding construction, canonical build, operation
storage, definition, expression context, symbolic values, literals, operators,
and mapper access live in their matching one-purpose leaves under
`compute/dsl/`. No aggregate expression include or second declaration
authority exists.

`BuildContext` retains the same two ordered vectors: one for
`ComputeIrNode` and one aligned value-domain row. Its non-template behavior is
compiled once under `src/program/compute/dsl/context/`; public headers carry
only declarations and the templates whose types depend on the mapper body.
Symbolic expression and `ComputeOp` non-template behavior likewise have one
compiled owner. This boundary adds no allocation, graph pass, node copy, or
lock: mapper execution still appends directly to those two vectors.

Canonical order is unchanged. Each mapper action reaches the same append
authority in authored order; local common-subexpression elimination searches
the already appended prefix and returns the first exact match, while every
Write remains explicit. Canonical binding bytes, node bytes, fixed format,
operation hash, and first rejection string therefore remain bit-identical.
The source registry lists every compiled leaf exactly once, and every product
header or source remains below 600 lines.

## Compute Runtime Contract

Kernel remains the semantic authority while Node makes repeated Compute map
execution practical. Warm staged execution, resident bindings, runtime
resource reuse, and measured backend timings are performance and evidence
features; they are not semantic authorities and they do not change planning
purity. Backend timestamp fields such as node `accel_kernel_ns` and
`accel_timestamp_source` are node evidence only; kernel planning, graph
identity, lowering, prepared commit order, and collective ordering never
consume them.

Kernel-owned runtime contracts:

- Warm staged execution keeps the staged binding shape valid while allowing
  node backends to reuse pipelines, command queues, temporary buffers, and
  descriptor state across repeated dispatches.
- Vulkan lowering for supported maps emits deterministic checked Vulkan
  source, not a successful identity-only artifact. A map uses one fixed
  256-lane workgroup shape and one eight-byte `{tile_count, iterations}` push
  constant. Canonical execution publishes `iterations = 1`; a semantically
  proved element-local recurrence publishes its authored positive bound. For a
  window of `N` tiles, node emits `ceil(N / 256)` workgroups and the shader
  rejects lanes `gid >= N` before any buffer access. Every admitted local index
  `i < N` has the unique lane `(floor(i / 256), i mod 256)`, so physical
  grouping cannot duplicate, omit, or reorder a map result. Kernel emits source
  text and stable artifact identity only; node owns SPIR-V compilation, shader
  module creation, push publication, and Vulkan/MoltenVK dispatch.
- Resident binding descriptors are opaque SDK-free binding facts. They carry
  buffer identity, byte extent, element bytes, stride, count, and usage so a
  backend can validate Compute-resident storage without
  exposing platform handles through kernel headers. `ResidentBufferRef`
  remains a trivially copyable descriptor; node-owned authenticity is carried
  separately through binding sidecar pointers.
- Fixed-op expansion is allowed only when the exact fixed integer law is
  specified for source text, helper bodies, artifact identity, and backend
  validators. Unsupported or partly specified ops keep failing closed.

Application-level solver or physics integration remains outside Compute runtime
ownership.
## Lowering Artifacts

`ArtifactKey`, `LoweringArtifact`, and `LowerComputeIR(...)` are pure kernel
support surfaces. Artifact keys contain API, scalar width, numeric domain,
fixed format, executable variant, operation hash, and canonical IR hash. The
variant is an exact orthogonal identity with the closed values `Canonical`,
`Controlled`, and `Recurrence`; graph and operation hashes remain unchanged by
backend specialization. This makes the three executable source families
disjoint without treating a reversible XOR or an implementation suffix as a
collision-free identity;
public executable artifacts also carry the `ExecutionMetadata` and canonical
IR bytes that produced their source. Private product graph tokens use a
destructive retained handoff from the same lowering result: they never own
canonical bytes, and keep only the representation required by their backend.
Lowering validates checked IR bytes and hashes, known APIs, scalar and domain
agreement, numeric policy, node and binding references, canonical writes, and
supported IR ops before emitting an artifact. The three compiled owners below
`lowering/validate/` separate numeric-format, domain/write, and final admission
responsibilities. `lowering/validate.hpp` exposes only the binding-domain and
complete-IR declarations; admission, metadata, fusion, Node, and backend
consumers do not parse or instantiate the validation implementation
independently. This physical boundary changes no validation order, rejection
string, graph identity, or execution schedule. The artifact metadata is built
from the same parsed checked IR used for source emission, so callers do not
rebuild execution metadata for the same IR after lowering succeeds. The thin
`LowerComputeIR` wrapper delegates to one internal `AdmitComputeInput` parse
owner and one `EmitComputeArtifact` owner. Admission owns key/hash checks,
exactly one canonical-IR parse, and lowerability validation; emission consumes
that same parsed fact for metadata, layout, and source. Its core result is a
transient emission with no canonical-vector owner. Public `LowerComputeIR`
attaches one canonical owning copy; source text remains the one emitted payload
owner. Internal authentication compares that source directly against transient
re-emission while admission authenticates the borrowed canonical bytes.
Internal graph admission retains the same key, metadata, and one parsed input
without materializing backend source. Node first fixes the final graph-step
set. Each surviving unfused Map then emits its backend artifact once; a legal
fused chain consumes the retained parsed inputs and emits only the final fused
artifact. No discarded pre-fusion source artifact is created.
Serialization, artifact keying,
fusion, fixed helpers, and backend source generation remain in their split
owners. Consumers include those leaf owners directly; no lowering-common
aggregate makes parse, validation, serialization, formatting, and emission
transitive peers.

`lowering/entry.hpp` contains only the public generic-lowering declaration; its
private transient-emission types live in `lowering/emission.hpp` and therefore
do not add `string`/`vector` parsing to entry-only consumers. The one
implementation owner is `src/program/compute/lowering.cpp`, which alone
includes the CPU, Metal, and Vulkan source-emission leaves. Thus a
backend-emission body edit compiles that one Kernel source owner before the
consuming archive and selected test link; it does not invalidate consumers
through a backend source header.
Changing the public entry declaration or artifact schema remains a real schema
edit and retains its normal consumer fanout.

Physical lowering ownership follows the contract hierarchy instead of encoding
multiple owners in leaf names. Artifact authentication lives at
`lowering/artifact/admission.hpp`; generated-function reachability lives at
`lowering/emission/reachability.hpp`; fixed source calls live under
`lowering/fixed/`; fused-map construction lives under `lowering/fusion/`; and
lowering surfaces are paired under `compute/cpu/`.

Canonical parsing copies each encoded string directly from the borrowed byte
range into its final string owner; it does not allocate an intermediate byte
vector. Every parser allocation is outside a `noexcept` boundary. A failed
allocation or standard-container length rejection is caught by the single
parse guard and returns `compute_ir_capacity`; it cannot terminate the process
or escape as a C++ exception.

Metal lowering emits stable source text for the admitted fixed IR subset.
Vulkan lowering for supported fixed maps emits stable checked source text with
the same identity discipline: scalar width, operation hash, canonical IR hash,
binding layout, helper bodies, and entry point are part of the identity.
CPU lowering emits a stable `CpuPlan` artifact for the
same checked IR identity when an artifact-bearing handoff is requested.
Internal Node CPU graph mint uses generic input admission, emits the artifact
once, discards its textual CPU marker and canonical storage, and retains that
one typed parsed fact for compact preparation. GPU graph mint performs the
inverse representation cut: it keeps source/key/metadata and discards parsed
input. Public supplied CPU execution uses the same `LoweringArtifact` as every
backend, admits once, emits one expected `CpuPlan` for exact comparison, and
only then prepares that runner. SIMD caps remain separate execution evidence.

`AdmitArtifact(...)` is the single detail owner for public
`LoweringArtifact` authentication on CPU, Metal, and Vulkan; Fake uses
the same owner when it exercises a real API artifact. Its declaration-only
surface is `lowering/artifact/admission.hpp`; all authentication bodies and
private comparison helpers have exactly one compiled owner at
`src/program/compute/lowering/artifact/admission.cpp`. Consumers carry no
inline authentication body or second validator. The
owner borrows the submitted
canonical vector, authenticates the exact plan-derived key (including API,
domain, scalar width, fixed format, canonical hash, and policy), parses once,
and emits one transient projection. The projection has no canonical or emitted
byte vector. Admission compares every `ExecutionMetadata` field, exact source,
and canonical identity without a second emitted payload allocation.
Key/kind/fixed-policy and empty-payload rejection happens before parse or
emission. The successful `ArtifactAdmission` token owns the sole
parsed form needed by CPU preparation; GPU preparation consumes the same proof
before compiling source. Backend-specific re-lowering and partial key/source
validators are forbidden.

A valid cold public artifact preparation is parse `1` / emission `1`. Product
graph compilation parses each admitted Map exactly once, decides graph fusion,
and emits each final Map artifact exactly once. It seals the retained result in
a private control-block capability: a source-only deleter type authenticates
the `shared_ptr` control block and recovers the typed token in `O(1)` without a
process-global lookup table, mutex, or weak-entry scan. Warm graph admission returns
parse `0` / emission `0` from its actual validator: CPU passes the retained
typed input to compact preparation, while Metal and Vulkan have no retained
`ParsedIR`. Once CPU instructions or a GPU pipeline/command owner is prepared,
both synchronous run and asynchronous submit consume only that prepared owner
and allocate nothing for artifact authentication. The retained owner accepts
no public artifact payload. Every direct public backend entry continues
through full `AdmitArtifact(...)`, so a mutated or forged caller value cannot
reuse the private proof and fails before pipeline compilation or output
writes.

Generated Metal and Vulkan sources emit the canonical stored-narrow
fixed helper library from the actual checked Fixed IR operations, then retains
only their transitive helper dependencies. Stored `add_sat`, unsigned
`add_sat`, `sub_sat`, and positive-negation helpers and the canonical
`sin`/`cos`/`exp`/`log`/`atan2` approximations are roots only when the IR uses
them; `tan` composes the selected canonical sine and cosine roots before its
wide division. Division, reciprocal, square root, reciprocal square root, and
terminal multiply-add execute through the widened lowering authority and do
not reintroduce stored-narrow arithmetic. Each source selects the lane32 or
lane64 helper library from the artifact scalar and never emits the opposite
library. Generic pair-of-`u32` integer support is separately owned and
has no Fixed naming or semantics, so non-Fixed integer artifacts contain zero
Fixed helpers on all source backends.

Generic 128-by-64 unsigned division scans numerator bits `127..0`, retains the
restoring remainder at 128-bit width until the final exact 64-bit remainder
boundary, and saturates a quotient with any bit at or above `64`. In particular,
bit `127` and a denominator at or above `2^63` must not lose the carry produced
by the restoring subtraction. Kernel fixed nonlinear lowering and node numeric
source consume that helper
instead of maintaining a second Fixed-named or 64-bit-remainder division loop.
Generic unsigned integer square-root helpers similarly own the U64-to-U32 and
U128-to-U64 widened boundaries without acquiring Fixed format policy.

Vulkan lowering contract tests follow the owner-local contract shape:
`lowering/vulkan.cpp` is only the runner, while `lowering/vulkan/` owns base
lane32 fixed source layout, lane64 fixed integer source, expanded fixed operations,
fixed scalar and predicate operations, fixed bit and shift operations, and nonlinear
helper authority. These tests inspect emitted artifact text only as
kernel-owned source semantics; they do not read implementation files or prove
driver, SPIR-V, or runtime backend behavior.

`backend/lowering/emission/reachability.cpp` independently checks Metal and
Vulkan source for non-Fixed-source absence, selected-lane and actual-operation
helper presence, dead narrow-entry-point absence, and monotonic generated-source size
across minimal, stored-helper, and full canonical-helper fixtures.

`BuildFusedComputeMapChainIR(...)` is the bounded kernel-owned executable fusion
bridge for admitted straight-line Map chains of any admitted length at least
two. Its supplied region graph must be proved by `PlanFusion(...)` as one
fused node with no rejected edge; the product graph may contain multiple such
maximal regions separated by rejected, capacity, or non-Map boundaries. The
bridge validates every checked IR hash and exact cost fact against its region
node, collects the parsed sources in graph order, and assembles the final
binding and node vectors once. It does not repeatedly copy a growing fused
prefix. Every source Map passes through the
sole generic admission owner exactly once; Node passes those retained
admissions into the builder, so fusion performs no second source parse.
Intermediate binding names receive stable `f0_`, `f1_`, ... prefixes, and each
boundary Read is replaced by the preceding boundary's single Write value.

Only the final assembled `ParsedIR` is serialized and hashed. The builder admits
that generated fact without parsing the just-produced bytes and derives
metadata directly from it. Node validates final graph bindings against that
metadata and then emits exactly one CPU, Metal, or Vulkan artifact. It does not
emit per-source artifacts, call `LowerComputeIR`, or parse the fused bytes
again. Product retained handoff releases the generated canonical vector and
keeps no final canonical owner, so the transient fused IR and artifact never
own duplicate canonical payloads. For a chain of `K >= 2` Maps, successful
product fusion therefore performs `K` source parses, zero fused-byte parses,
one final artifact emission, and zero canonical-payload copies. The public
result and `LowerComputeIR(...)` behavior remain byte- and
fingerprint-identical. The bridge does not accept arbitrary shader/source text
and does not cover reductions, atomics, scatter, shared writes, non-map region
shapes, or a region containing a rejected fusion boundary. Node owns the
single graph-order scan that splits the product graph at those boundaries and
invokes this bridge once per maximal legal region. The declaration lives in
`include/kernel/program/compute/lowering/fusion/build.hpp`; the complete
implementation has one compiled owner at
`src/program/compute/lowering/fusion/build.cpp`, so changing fusion mechanics
does not fan out through every header consumer.

Lowering and artifact validation reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Unsupported API or scalar | `compute_api_unsupported` or `compute_scalar_unsupported` |
| Non-ok or empty IR | `compute_ir_invalid` or `compute_ir_missing` |
| Operation hash does not match canonical bytes | `compute_ir_hash_mismatch` |
| Malformed canonical bytes or scalar mismatch | `compute_ir_malformed` or `compute_ir_scalar_mismatch` |
| Canonical parse allocation or container-length failure | `compute_ir_capacity` |
| Invalid binding count/kind/payload/duplicate | `compute_ir_binding_count_invalid`, `compute_ir_binding_invalid`, `compute_ir_binding_payload_invalid`, `compute_ir_binding_duplicate` |
| Floating param or wrong param byte size | `compute_param_float_unsupported` or `compute_ir_param_size_mismatch` |
| Invalid node count/ref/op or missing write | `compute_ir_node_count_invalid`, `compute_ir_node_invalid`, `compute_ir_op_unsupported`, `compute_write_missing` |
| Binding scalar/layout mismatch during emission | `compute_ir_binding_scalar_mismatch` |
| Artifact key mismatch before dispatch | `compute_artifact_mismatch` |
| Wrong source kind or non-executable artifact before backend dispatch | `compute_artifact_non_executable` |

Current Metal, Vulkan, and CPU lowering covers deterministic parameter
loads, read loads, writes, and 32/64-bit `Fixed<I, F>` storage. Arithmetic
nodes carry their derived `(I,F)` independently from storage width, use a
signed 128-bit CPU value or two explicit 64-bit limbs in Metal/Vulkan source,
and reject a derived width above 128 bits.
The admitted subset includes
widened `add`, `sub`, `mul`, `neg`, `abs`, unsigned-lane
`abs_magnitude`, `sign`, `min`, `max`, `clamp`, `select`, `eq`, `ne`, `lt`,
`le`, `gt`, `ge`, predicate `not`, `and`, and `or`, storage bitwise `bit_and`,
`bit_or`, `bit_xor`, `bit_not`, and checked constant shifts `shl_const<N>`,
`shr_logical_const<N>`, `shr_arithmetic_const<N>`, `add_sat`,
`add_sat_unsigned`, `sub_sat`, `neg_positive_fixed`, `mul_fixed`,
`mul_fixed_scaled`, `mul_unsigned_fixed`, `mul_add_fixed`, `div_fixed`,
`recip`, `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `exp`, `log`, and `atan2`, plus composite DSL same-name overload families
`dot` (2, 3, 4, 5, 6, 8), `conv` (3, 5, 7, 9), `Axis`, `MatOp`, `mat`,
`aff`, `mix` (2, 3, 4),
`poly` (degree 2, 3), `bezier` (quadratic, cubic),
`MetricOp`, `len`, `dist`, `Norm`, `AngleOp`, `angle`,
`sum`/`sum(SumOp::Abs/Squared, ...)` (2, 3, 4),
`diff`/`diff(DifferenceOrder::Second/Third, ...)`,
`mean`/`mean(MeanOp::Abs/Squared, ...)` (2, 3, 4), `centered`/
`centered(CenteredOp::Abs/Squared/Cubic/Quartic, ...)`, `standardized(StandardizedOp::Cubic/Quartic, ...)`, `var` (2, 3, 4), `rms` (2, 3, 4), `cov`
(2, 3, 4 paired samples), `corr` (2, 3, 4 paired samples), `noise` (1D and
2D), and scalar composite helpers
`saturate`, `step`, `lerp` (linear, bilinear, trilinear arities), `unlerp`,
`remap`, `smoothstep`, `smootherstep`,
`lerp(LerpOp::Smooth, ...)` (linear, bilinear, trilinear arities), `absdiff`,
`min`, `max`, `median`, `spread`, `clamp_range`, `in_range`, `out_range`,
`bandpass`, `bandstop`, `fixed_zero`, `fixed_one`, `fixed_max`, `FixedOp`, `fixed`,
`centered`/`centered(CenteredOp::Abs/Squared/Cubic/Quartic, ...)`, `is_zero`, `nonzero`, `is_neg`,
`is_pos`, `is_nonneg`, `is_nonpos`, `all`, `any`, `keep_if`, `zero_if`,
`near`, `deadzone`, `snap`, `clip`, `positive_part`, `negative_part`,
`huber`, `unit`, `proj`, `HashOp`, `hash`, `cross`, `orient`, `bary`, `triple`,
`LineOp`, `line`, `PlaneOp`, `plane`,
`reject`, `reflect`,
`ratio`, `proportion`, `zscore`, `softsign`,
`softsign`, `ActivationOp`, `activation`,
`WindowOp`, `window`, and `fade` over
those admitted ops. Numeric `add`, `sub`, `mul`, and unary `neg` never use the
storage wrap law. Bitwise operations and constant shifts remain storage-bit
operations because their meaning is explicitly bit-oriented. Logical right
shift operates on unsigned storage and casts back to the fixed lane. Arithmetic
right shift is explicit sign extension: unsigned logical shift of the storage
value, then OR with the deterministic high-bit mask when the original sign bit
is set and `N > 0`; `N == 0` is the identity shift. `min`, `max`, `clamp`,
`lt`, `le`, `gt`, and `ge` use signed fixed-lane ordering. `eq` and `ne`
compare raw fixed storage bits. Predicate-producing comparisons and predicate
ops return deterministic integer scalar values: false is `0` and true is `1`
in the same fixed storage width. Predicate `and` and `or` treat any nonzero
input as true, and `select` treats any nonzero predicate value as true and
otherwise returns the false operand. Clamp is `min(max(value, lower), upper)`
under the same signed ordering. Saturating signed ops clamp to `FixedMin` and
`FixedMax`; `add_sat_unsigned` clamps to the lane's unsigned storage max.
An explicit `Quantize` aligns the binary point, applies its declared rounding
mode, then applies saturating or wrapping overflow to the target 32/64-bit
storage width. Every ordinary fixed `Value` Write must reference a `Quantize`
node whose format matches the write binding; validation rejects any direct
ordinary write from an arithmetic, parameter, constant, or read node. The sole
non-Quantize Fixed write is the internal same-width `BoundaryMask` contract
above, whose exact normalized producer can only store raw zero or raw one under
the target binding's complete policy. The internal DSL inserts the explicit
Quantize IR node only at an ordinary final write serialization boundary; it
does not insert one before a storage-domain operation. The public Flow builder
reaches ordinary serialization only after validating that the user's fixed
output is an unchanged stored input or has an explicit public `quantize<T>()`
root. Thus the serialized graph has one visible numeric narrowing authority and
no widened operand is silently converted. Shader source represents widened values as two 64-bit limbs and
must not use unavailable `int128`/`uint128` spellings, backend float
conversions, or shader overflow as arithmetic authority. For a stored operand
with fraction `F`, division computes `(abs(lhs) << F) / abs(rhs)` with declared
rounding, reciprocal uses the same division with raw `1 << F`, and square root
computes deterministic integer `isqrt(raw << F)`. Reciprocal square root is the
checked composition of those integer laws. They do not use backend intrinsic
sqrt/reciprocal/rsqrt, float casts, or fast math. These ops
do not introduce atomics, scatter, reductions, shared writes, arbitrary shader
source, or dynamic shifts.
Generated symbols hex-encode binding names by kind, so caller-controlled names
cannot inject source/comment structure.

`ExecutionMetadata` and `BuildExecutionMetadata(...)` are the kernel-owned
execution metadata support surface for checked Compute IR. Both it and
`BuildMapGraphSignature(...)` delegate policy, API, scalar, canonical hash,
parse, and lowerability checks to the sole `AdmitComputeInput` owner; neither
keeps a private checked-parser mirror. They consume that admitted parse before
returning the
`ComputeMap`, parameter byte storage, input element byte widths, read count, write
count, and graph binding role/name order consumed by node resident-program
execution. The graph binding order contains checked IR reads in canonical
read-binding order followed by writes in canonical write-binding order;
parameter bindings stay in the IR parameter store. Kernel exposes `BufferRole`
values and binding
names for this order so node can validate graph refs without parsing IR
internals. Raw canonical binding kind numbers and lowering parse details remain
kernel implementation authority.

Read and parameter bindings always use the IR scalar width. A write binding
normally uses that same width. The only mixed-width write admitted by policy
`1` is a sole unsigned canonical mask output: U64-to-U32 narrowing or
U32-to-U64 widening. CPU, Metal, and Vulkan discard zero
upper bits or zero-extend only after the checked IR proves the exact
`Select(predicate, Constant(raw=1), Constant(raw=0))` root. Fixed IR carries
one direct `Quantize` wrapper whose format equals the Select and both constants;
it cannot rescale the normalized raw bit. Admission also requires exactly one
write binding and one Write node targeting it. This is not a general integer
conversion, truncation, extension, or rounding contract: arbitrary arithmetic,
additional writes, signed output modes, and other width-changing roots fail
with `compute_ir_binding_scalar_mismatch` before source emission.

Every backend also compares the execution plan with the `ComputeMap` emitted
by canonical lowering. Mixed-width output is valid only when the IR metadata
itself declares that width; changing a prepared plan's output width cannot
turn an eight-byte program into a four-byte program.
