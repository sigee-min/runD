# Compute SDK

Canonical graph observation, resource hazards, Program caching, and
asynchronous compilation are owned by
[Compute Graph Services](./compute/services.md).

`<rund/compute.hpp>` is the default Compute entry. It keeps target selection,
typed data, fixed execution shape, compilation, execution, and explicit
readback in one `rund::compute` vocabulary. Asynchronous compilation is an
opt-in physical surface reached through `<rund/compute/async.hpp>`; it completes
the same deferred Flow terminal without adding another graph or compilation
authority. Composite expression functions and matrix, transform, factor,
solve, and spectrum stages are reached through `<rund/compute/math.hpp>`. The
math entry extends the same Flow and lowering authority; it is not a second
graph language.
`runD::sdk` is the only application link target.

`<rund/compute/session.hpp>` owns the `Session::compute(Job&)` template and the
four Node-host product values `Request`, `Submission`, `Poll`, and `Completion`
under `rund::compute`. Its transitive support hierarchy is
`compute/session/{request,submission,poll,completion,await}.hpp`; those leaves
each own one declaration or coroutine bridge and are not additional direct SDK
entries. The awaiter is nested as `Request::Awaiter`, so no second root awaiter
name enters the product vocabulary. `<rund/session.hpp>` retains only the
forward declarations needed to state the admission point, so lifecycle-only
translation units do not inherit resident Job and coroutine implementation
headers. A translation unit that calls `session.compute(job)` includes the
Compute Session entry; standalone Flow code does not parse that scheduler
boundary.

`<rund/compute/pipeline.hpp>` owns the focused opt-in Pipeline surface; the
basic `<rund/compute.hpp>` entry deliberately excludes it, while the all-domain
`<rund/rund.hpp>` umbrella composes it. Use `pipeline(device)` to
bind an ordered sequence of already compiled Programs to resident Buffers,
`prepare()` once, then reuse the resulting move-only `Pipeline` through
standalone `run()` or `Session::compute(Pipeline&)`. Pipeline does not add a
second graph language or scheduler. Its declaration-order, hazard, claim,
poison, one-submit, explicit-readback, status, memory, and performance laws are
owned by the [Node Compute Pipeline contract](../../node/docs/contracts/compute/pipeline.md).

## 1. Primary Borrowed Flow

Use `on` with one explicit `Target` and an lvalue contiguous range for the
shortest path. `std::vector` (including custom allocators), `std::array`,
`std::span`, C arrays, and user ranges that can form an exact typed
`std::span` all use this one entry. `Flow<Output(Inputs...)>` borrows the input
view until `collect()` returns; construction performs no payload copy.

The compiler-visible Flow identity uses the public marker
`input::Bound` when its payload is already borrowed and `input::Deferred` when
the recipe will receive payloads at Program execution. Exactly one marker is
present in every Flow type. Neither marker is stored at runtime, so this type
distinction adds zero payload bytes, allocation, dispatch, or copy. Internal
names do not participate in a consumer's type spelling or diagnostics.

The admitted span element type after removing `const` and `volatile` must be
exactly a Compute value type. Scalar conversion is never an input adapter: for
example, an `int16_t` range cannot enter an `int32_t` Flow, and an `int32_t`
index range cannot stand in for `uint32_t`. Volatile storage cannot form the
required read-only span. Proxy ranges such as `vector<bool>`, named or
temporary `initializer_list` storage, and every rvalue range are rejected at
the `BorrowedRange` constraint. This keeps one lifetime and element-identity
rule for primary input, combine, join, complex, gather, scatter, partition,
segmented collectives, matrix multiplication, and solve side inputs.

```cpp compile
#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

rund::compute::Result<std::vector<std::int32_t>> twice() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};
  return rund::compute::on(rund::compute::Target::cpu(), input)
      .map("twice", [](auto value) { return value * 2 + 5; })
      .collect();
}
```

The helper returns the typed Compute outcome unchanged. A failure can never be
mistaken for a valid empty output, and its `code()`, `error()`, and
`exit_code()` remain available to the application boundary.

`on(Target::cpu(), input)` uses the host worker width. Use
`on(Target::cpu(width), input)` when the CPU width must be fixed. Metal and
Vulkan derive their execution width from the selected device.
Calling `on(selector)` without an input returns `FlowBuilder`, the typed recipe
builder used by `input<T>(count)` and `map<T>(name, count, function)`.

`map` builds its bounded typed expression immediately, while Map/Scan steps
remain a resource-free recipe. `collect() &&` is the single expensive
terminal: it opens only the selected backend, translates the recipe once to
the canonical Kernel graph and Program, compiles, allocates buffers, uploads, runs,
allocates the host result, and reads it back. An empty recipe fails with
`compute_flow_empty`; an empty Map name fails with `compute_name_empty`.

## 2. Compose Primary Primitives

Intermediate operations stay lazy, so a collective reads as one typed chain
without exposing buffers or bindings.

```cpp fragment
auto prefix =
    rund::compute::on(rund::compute::Target::metal(), input)
        .map("adjust", [](auto value) { return value + 1; })
        .scan(rund::compute::Scan::InclusiveSum)
        .collect();
```

The primary chain supports `map`, `filter`, `scan`, `reduce`, `sort`,
`argsort`, `compact`, `histogram`, `window`, `gather`, `scatter`,
`scatter_reduce`,
`partition`, `segmented_scan`, `segmented_reduce`, bounded `expand`,
bounded `join`, integer `group_by`, and explicit graph `unroll<N>`. `branch`
exposes immutable typed
stage references, while `record` and `outputs` assemble structure-of-arrays
fields and ordered terminals without a host readback. Side-input stages add
their input type and immutable count to the Flow signature. Gather uses the
index count as its output count; each compiled Program retains every input
count separately.

`Flow` is also the public programmable bounded execution language. It does not
expose Kernel, Accel, Metal, Vulkan, shader modules, native handles, or a
backend callback. A finite bounded data-dependent program is expressed by
functional composition:

- `map`, `select`, and Fixed/integer expressions are the scalar program;
- `gather` is typed indexed resident load;
- `filter`/`compact` and `expand(MaxItems, count, emit)` are stable bounded
  selection and append;
- `partition`, `sort`, grouping, and segmented collectives establish canonical
  work order;
- `unroll<N>(step, converged)` explicitly expands a small bounded active
  worklist in one Program graph;
- `scatter_reduce(indices, output_count, op)` is the deterministic conflicting
  indexed-write boundary.

Every loop bound, append multiplicity, physical capacity, scalar width, and
resource binding is consequently present before `compile()`. No construction
step invokes consumer code at execution time. A resident count is an ordinary
typed Program input/output and a canonical graph dependency, so later work can
dispatch from it without a host readback. `logical_count > capacity` is a typed
execution failure and dispatches no work; counts are never silently clamped.

This is an equivalently expressive bounded DSL, not a claim that each domain
algorithm has a named SDK primitive. The normal-form argument is:

1. A program admitted by this contract has finitely many fixed-width scalar
   registers, finitely many resident arrays with declared capacities, and a
   finite iteration bound. Its execution can therefore be expanded to a finite
   acyclic transition graph.
2. Integer bit operations, shifts, arithmetic, comparison, and `select` encode
   each scalar transition as a typed expression circuit. Fixed transitions use
   the same construction with their declared rounding and overflow policy.
3. `gather` realizes an indexed read. Stable selection/append realizes a
   bounded work queue. `scatter` realizes a unique indexed write, while
   `scatter_reduce` realizes a conflicting indexed write with one specified
   source-ordinal fold. Sort/group operations establish any required canonical
   key order before that write.
4. A data-dependent branch becomes `select` for scalar state or stable Filter
   for work items. Small compile-time circuit expansion uses `unroll<N>`.
   Fixed resident recurrence uses Pipeline `repeat<N>`. Bounded resident
   recurrence uses `windows<Max, Tile>` with one device-resident count and an
   optional recurrent U32 terminal leaf. Both reuse one compiled body Program
   plus one Program-internal workspace. Pipeline orders cross-Program
   transitions, consumes occurrence-local telemetry before route reuse, and
   publishes persistent state only at `commit()`.

By induction over that finite transition graph, every admitted bounded
word-array program has a Flow/Pipeline representation. The converse is
immediate because every Flow node is one such transition. Compile-time graph,
expression, resource, step, and capacity ceilings are product limits, not
implicit semantic narrowing: exceeding one is a typed admission failure and
never selects a CPU callback, private shader, spill allocation, or alternate
backend. Algorithms requiring floating-point behavior outside the public
integer/Fixed numeric contract are not admitted under a different meaning.

At a Program boundary,
`on(device).input<Bounded<T, Count>>(capacity)` reconstructs the bounded
typestate from the ordinary flattened input signature `(T values, Count
count)`. This is deliberately the same two-buffer ABI produced by a
`Program<Bounded<T, Count>(...)>` terminal, so Pipeline can connect two
Programs with `read(values, count)` without a wrapper, copy, allocation, or
host count observation. The count input has one element, is unsigned, and has
the same storage width as `T`. Host-bound `Program::run`/`Job::write` validates
that scalar against `capacity` during the existing flattened shape/type walk,
before any upload or dispatch. This is one `I`-input validation pass, not a
second bounded-metadata pass. Overflow is consequently always
`compute_workset_overflow`, independent of whether the first consumer is Map,
Gather, Sort, Scan, or ScatterReduce, and the compiled graph shape and
fingerprint remain unchanged. Pipeline keeps the count resident and uses the
same capacity contract in backend execution; it introduces no host readback.
Inside Map, comparisons produce a typed `Predicate<T>` and `select` chooses
between two expressions without a host branch. `map` derives its next Flow
value type from the returned expression rather than forcing the input type:

```cpp fragment
auto mask = compute::on(Target::metal(), signed_values)
    .map("positive", [](auto value) {
      return compute::select<std::uint32_t>(value > 0, 1u, 0u);
    }); // Flow<uint32_t(int32_t), stage::Exact>
```

The current canonical Map lane supports general type changes only at equal
scalar width. This permits deterministic signed/unsigned or fixed-width result
selection without changing the lane layout. `mask(predicate)` is the sole
width-changing expression: a 64-bit predicate becomes a U32 `0` or `1` by
lowering to canonical Select and a four-byte output store. It exists for
collective masks and segment heads; it is not a general truncating cast.
Public 32-to-64 widening and arbitrary 64-to-32 narrowing remain rejected until
an explicit conversion primitive owns their overflow and rounding contract.
Bounded stages have one internal exception: when a 32-bit payload carries a
64-bit resident count, refiltering widens the canonical boolean mask to u64 so
the original count lineage is preserved. This is still a `0`/`1` mask, not a
general integer conversion; the single-output mask root, graph identity, and
backend artifact must all agree before execution.

Flow owns the complete canonical element-expression opcode surface. Ordinary
expression interning hashes each node's canonical field sequence in explicit
little-endian order through Node's single internal FNV state transition. The
hash is only the open-addressed-table prefilter: full node equality remains the
authority, so collisions cannot merge distinct expressions. Consolidating the
state transition does not change the node sequence, fingerprint, allocation
count, or asymptotic `O(1)` expected lookup cost.

Ordinary operators cover add/subtract, domain-correct multiply/divide, bitwise logic,
comparisons, and predicates through `<rund/compute.hpp>`. Add
`<rund/compute/math.hpp>` for named composite functions: `abs`, `abs_magnitude`,
`sign`, `min`, `max`, `clamp`, `select`, `bit_not`, compile-time `shl` /
`shr_logical` / `shr_arithmetic`, signed and unsigned saturating arithmetic,
the distinct `mul_fixed`, `mul_fixed_scaled`, `mul_unsigned_fixed`, and
`mul_add_fixed` operations, Fixed-only `div_fixed`, `recip`, square root, reciprocal square root,
sine, cosine, tangent, exponential, logarithm, power, and `atan2`.
Integer and Fixed domain constraints are checked by the C++ overload set;
`div_fixed` participates only for Fixed expressions, while integer division
uses ordinary `/` and selects the signed or unsigned opcode from the integer
lane type. Calling `div_fixed` with an integer expression is ill-formed rather
than an alias for integer division. `add_sat` and `sub_sat` admit signed integer
or Fixed expressions. `add_sat_unsigned` admits unsigned integer expressions
and Fixed expressions whose storage bits require unsigned saturation; this is
the only public Fixed unsigned-storage saturation override. Arithmetic right
shift admits signed integer or Fixed expressions, while logical right shift
remains an explicit stored-bit operation.

Canonical IR validation independently rejects every Fixed-only or
domain-mismatched opcode before any backend artifact is generated. Unsigned
`min`/`max`/`clamp` and ordered comparisons use the same public surface as
other domains, but `CanonicalIrOpForDomain` lowers them to the matching
`*Unsigned` opcode before graph fingerprinting. Forged unsigned IR containing
the base signed-order opcode is noncanonical and rejected. Conversely the
seven `*Unsigned` order opcodes are not admitted for Fixed; Fixed order remains
the signed numeric order declared by `Fixed<I,F>`. Backend execution never
calls a host math callback. These expressions lower directly to the same
canonical Kernel IR opcodes on CPU, Metal, and Vulkan; backend names and choice
remain excluded.

The first input does not choose the meaning of every value. Same-width records
may carry signed and unsigned fields together, and each binding seeds its own
numeric provenance. The single merge contract is: same signedness widens,
`I32 + U32` becomes `U32`, `I64 + U32` becomes `I64`, any merge with `U64`
becomes `U64`, and Fixed never mixes with an integer. A general `select`
combines only its two result values; the predicate has an independent domain.
An exact zero/one Select borrows that predicate domain only when a canonical
ordinary Write to an unsigned `mask` target proves ownership; a signed result
does not become an implicit signedness conversion.
Ordinary writes require the result domain to equal the output binding, except
for the public canonical unsigned `mask` width bridge. Public expressions do
not gain implicit mixed-format arithmetic, integer/Fixed conversion, or
signedness conversion from these exceptions.

That separation starts at bindings. Fixed programs require same-lane Fixed
parameters, reads, and ordinary writes; integer programs reject Fixed bindings
while continuing to admit same-lane signed/unsigned inputs. Two internal Flow
materialization modes are explicit in canonical IR:

- `CheckedOrdinal` stores an opposite-signedness integer only after exact
  same-width range normalization. `I -> U` is
  `select(X >= 0, X, 0)`; `U -> I` is
  `select_unsigned(X <= INT_MAX, X, 0)`. Valid values retain their raw bits and
  out-of-range values become raw zero. A separate Flow guard owns any semantic
  rejection.
- `BoundaryMask` stores exactly `select(X != 0, 1, 0)` into a same-width signed
  integer or Fixed target. A Fixed target inherits the complete target I/F and
  numeric policy, but receives raw zero or raw one rather than a scaled numeric
  conversion. An unsigned target uses the pre-existing canonical unsigned-mask
  `Value` owner for that same shape, so identical zero/one semantics never gain
  a second graph identity.

`X` may be an already-valid pure integer expression. Graph fusion may
substitute an expression for a Read, but `CheckedOrdinal` must reuse that same
`X` as the comparison operand and selected true arm, while `BoundaryMask` must
retain the exact `X != 0` predicate and constant `1`/`0` arms. Same-domain
retyping is a no-op. Both special modes are serialized in the Write node and
therefore change the graph fingerprint and program-cache identity; unknown or
malformed modes fail before CPU, Metal, or Vulkan lowering. The
independent public canonical unsigned mask Write retains its defined width
bridge.

Literal bits and indices stay compact in serialized IR, while construction
tracks their typed or anchored provenance. Signed and unsigned constants with
the same bits are not commoned together. Validation reconstructs effective
node domains in four bounded linear sweeps, then CPU, Metal, and Vulkan
all use base ordered opcodes for signed/Fixed values and explicit
`*Unsigned` opcodes for unsigned values. This is an algorithmic validation
bound, not a runtime speed claim.

Backend helper reachability follows those canonical per-node opcodes as well.
The graph header never decides whether a signed or unsigned division helper is
present: Metal and Vulkan emit the helpers required by reachable
`DivSigned` and `DivUnsigned` nodes at the selected 32- or 64-bit scalar width,
and emit neither for a graph without integer division. This keeps both header
orders of a heterogeneous signed/unsigned Map executable without adding a
second domain authority.

Every Flow-backed element input is materialized from the referenced
`FlowValue`, not reconstructed from the C++ lane type. Exact, Bounded, and
Scalar `map`, `filter`, `combine`, branch `zip`, and deferred multi-input
recipes therefore retain the actual `(I,F)`, rounding, overflow, and
approximation policy written by the preceding `quantize<T>()` boundary.
Nondefault policy cannot revert to the type default between graph nodes.

Consecutive Map fusion requires one exact numeric domain and one exact control
domain. Every producer input/output and consumer input/output must have the
same scalar kind and, for Fixed values, the same `(IntegerBits,
FractionBits)`. Their `(count, predicate, capacity, predicate_expected,
iteration)` tuples must also be identical. A `quantize<T>()` format transition
therefore remains a materialization boundary, and two controlled Maps from
different unrolled iterations cannot collapse into one occurrence. The
cross-format and two-iteration negative contracts execute on CPU, Metal, and
Vulkan and require two surviving value Maps with bit-identical output.

A single-output map whose selected expression is exactly one of its input
values forwards that canonical value ID. It does not create a map node,
intermediate buffer, or accelerator dispatch. This is graph construction, not
a runtime optimization or fallback: an expression with any operation remains
a map, and a forwarded graph input requested as a public terminal is still
materialized once by the existing output-copy boundary. A collective following
an identity mapper therefore reads the original binding directly while graph
identity, backend choice, and stored numeric policy remain unchanged by an
otherwise meaningless map name.

The compiler also removes a Gather materialization when that Gather has one
Map consumer. For semantic source `S_j`, index stream `I_j`, and output lane
`k`, the fused Map reads exactly

```text
ReadAt(S_j, I_j[k], count(S_j))
```

before evaluating the unchanged Map expression. All active indices are
validated in increasing logical-lane order before the first output write, so
an out-of-range index remains failure-atomic and reports
`compute_gather_index_out_of_range`. CPU performs that complete validation
before its write sweep. Metal and Vulkan use one fixed validation dispatch
followed by the indexed Map dispatch; the latter is suppressed on failure.
The compiler removes the Gather only when its validation domain and the Map's
execution domain are identical:

```text
D_gather = D_map
```

An exact Gather therefore cannot fuse into a shorter bounded Map. A bounded
Gather may fuse only when both operations consume the same count authority.
For a rejected index stream, every backend retains

```text
overflow_ordinal = min { k in D_gather | I[k] >= count(source) }
```

Standalone Jobs and Pipeline execution project that same ordinal from the one
backend-authored status range; fusion cannot preserve the typed reason while
discarding its position evidence.
There is no element-count threshold, backend callback, alternate expression,
or host readback.

All lowered buffer addresses have one backend-neutral form:

```text
binding_base(b) + logical_index * stride(b)
```

The logical index is `k` for a direct read/write and `I_j[k]` for the
`ReadAt` source. Metal and Vulkan source contains one base constant and one
stride constant per binding. A runtime view changes only those constants, so
indexed and direct addressing cannot acquire separate bias logic. On Vulkan,
if descriptor alignment moves the native descriptor start down to `D`, the
specialized base is the exact remainder `B`; admission requires the same
remainder for every dispatch window. Consequently

```text
D + B + logical_index * stride
  = authored_offset + logical_index * stride
```

for every selected byte. Specialization affects only physical placement;
canonical graph identity, operation order, overflow policy, and stored bits
are invariant.

For `G` Gather values consumed directly by Maps, capacity `N`, and scalar
width `W` bytes, the absent materialization traffic is at least

```text
2 * G * N * W
```

per execution: one `G*N*W` Gather write and the matching Map read.
Materialization also has `G` resources and `G` Gather dispatches that are
absent from this direct path before later graph fusion and arena placement. The canonical Map
still admits at most 16 semantic expression inputs. Each semantic input may
add one deduplicated U32 index binding, so the physical input envelope is 32.
One shared naming owner projects `input`/`inputN` and `output`/`outputN` into
both checked IR metadata and Accel graph references. Pointer identity, source
allocation order, backend, and temporary value IDs do not participate. A
metadata/name mismatch fails graph admission before backend source
compilation.

```cpp fragment
auto heads = compute::on(Target::vulkan(), keys)
    .map("heads", [](auto key) { return compute::mask(key != 0); });
// Flow<uint32_t(uint64_t), stage::Exact> when keys stores uint64_t.
```

`reduce` returns a
`stage::Scalar` Flow, so sequence-only operations such as `sort` are rejected
at compile time while scalar `map` and terminals remain available.
`matrix(shape)` changes a sequence into dynamic `stage::Matrix<>`, while
`matrix<Rows, Cols, Batches>()` carries static dimensions in the stage type.
Transpose swaps static rows and columns, and static `matmul` rejects inner or
batch mismatches during template instantiation. Host extent and dynamic shapes
remain terminal binding checks. Both forms use the same canonical Matrix
primitive. Factor and Spectrum operations exist only on Matrix; Sequence has
no duplicate shape-taking decomposition overloads.
`complex` combines fixed real/imaginary sequences as `stage::Complex`; `fourier`
keeps that category and `.real()` or `.imag()` projects a sequence from the
same canonical Transform node. Compiling the unprojected Complex stage returns
`Program<Outputs<T, T>(Inputs...)>`; its result type therefore determines that
`read<0>()` is real and `read<1>()` is imaginary without a runtime tag or a
second graph execution. One-shot `collect()` on that stage returns
`Result<tuple<vector<T>, vector<T>>>` in the same real/imaginary order. This is
the public typestate rule: the result category of one stage is the C++ type on
which the next operation and terminal result are resolved.
The imaginary side must use the real side's `(I,F)`. It inherits the active
real value's rounding and overflow policy while retaining its own
approximation provenance, so a `Down/Wrap/Deterministic` real input admits an
`Exact` imaginary binding as `Down/Wrap/Exact` without policy erasure.

Reusable graph construction is an ordinary C++ function passed directly to
`.pipe(function)`. `pipe` runs the function once while the lazy recipe is
built, passes an immutable `StageRef<Value, Cardinality>`, and restores the
returned ref as the next owning Flow stage. The function is never
retained as an execution callback and cannot run on a CPU worker or GPU
command path:

```cpp fragment
auto normalize = [](auto values) {
  auto total = values.reduce(Reduce::Sum);
  return values.combine("normalize", total, [](auto value, auto sum) {
    return value * 1000 / sum;
  });
};

auto output = compute::on(Target::metal(), input)
    .filter([](auto value) { return value > 0; })
    .pipe(normalize)
    .collect();
```

The same function can be applied to an input-bound Flow, a deferred fixed-shape
Flow, or an immutable StageRef inside a branch. The returned C++ cardinality
owns the next overload set. Composition cannot erase cardinality, introduce an
ambient backend, or bypass canonical Kernel graph lowering. Use `combine` for
a sequence/scalar dependency; capturing a symbolic scalar in an element lambda
is not an implicit host value conversion.

Raw element functions must be empty, trivially copyable, and const-invocable.
Mutable lambdas and every raw value, reference, or pointer capture are rejected
during template instantiation. Runtime integer and Fixed constants enter an
element function only through `compute::capture(captureless_function,
values...)`; each value is copied and canonicalized into Expr constant bits
while the recipe is built. Pointer, reference, container, callback, device
handle, and symbolic StageRef arguments do not satisfy that API.

The header invokes the function exactly once, during constant evaluation, with
typed symbolic inputs and symbolic capture slots. That invocation produces a
static expression recipe. Runtime graph construction only materializes the
recipe's canonical opcodes, constant bits, input edges, and capture-slot values;
it never invokes the function again. This admits constexpr pure helpers but
rejects a read of mutable global state, a non-constexpr host helper, and any
other operation that needs runtime host state. Neither a callable nor a host
callback is retained in the Graph or invoked during execution. Backend
selection is fixed before recipe construction and is never passed as an Expr
input.

For integer recipes, a literal's expression anchor is compile-time association
only: the literal already has its complete scalar type, so lowering emits the
constant into the recipe's one `ExprState` without recursively materializing
the anchor. Actual operands therefore lower once. For a literal chain of depth
`D`, construction is linear in `D`: each anchored integer literal emits once.
Fixed literals deliberately retain anchor traversal
because the anchor supplies rounding and overflow policy that is not present
in the literal's C++ value alone.

Captured values are symbolic expressions during recipe construction. They may
participate in `select` and expression operations, but cannot steer a host
`if`, loop, overload choice, or template branch. Consequently a runtime capture
cannot expose a hidden host-state callback path.

`branch` is the non-linear construction boundary. Every derived `StageRef`
is immutable, so one source can feed sorting, aggregation, filtering, and
another subgraph without cloning the input or running twice. `record` groups
typed fields as structure-of-arrays metadata; it never uploads a C++ object
layout. Stage-level records accept existing stages. A Map may instead return
`record(field<Tag>(expr), ...)`; that form lowers to one ordered multi-write
Map and stores each field in its own buffer. `record.get<I>()` and
`record.get<Tag>()` recover the exact sequence, bounded, scalar, or nested
record type so an intermediate field can feed another canonical operation.
Tags are compile-time lookup keys and never enter graph identity. `outputs`
fixes terminal order explicitly. Selecting an external input stage as an
output materializes one canonical identity Map; it never aliases an input
binding as a writable output or performs a host-side copy. For a Fixed input,
that identity Map preserves the input resource's exact `(I,F)`, rounding,
overflow, and approximation metadata; it never reconstructs policy from
storage width.

Static Map-record materialization visits fields exactly once in declaration
order before assembling the typed output record. Field storage is indexed
directly; tuple reflection over the complete recursive expression type is not
a second lowering authority. This makes graph-node order independent of C++
function-argument evaluation order while preserving the declared field order.

Map construction admits at most 16 outputs. The builder carries those value
IDs in one 68-byte, trivially copyable inline value while it publishes the
persistent Map route. It does not allocate a transient output-ID vector. For a
Map with `K` outputs, direct Graph construction therefore retains only the
canonical route allocation; Flow construction and its later Graph compilation
each avoid one allocation with at least `K * sizeof(uint32_t)` transient
capacity. This is a bounded
construction-cost property and does not change graph identity, output order,
or execution memory.

```cpp fragment
auto result = compute::on(Target::vulkan(), input)
    .branch([](auto values) {
      auto stats = compute::record(values.count(),
                                   values.reduce(Reduce::Sum),
                                   values.reduce(Reduce::Max));
      return compute::outputs(values.sort(), stats);
    })
    .collect();
```

`zip(a, b, ...)` is the heterogeneous element stage. Exact inputs must have
the same count; bounded inputs must additionally share the same resident count
ValueId. It emits one ordered multi-read Map, and a record result emits one
ordered multi-write Map. `record(a, b, ...)` only groups existing graph values;
`combine` is the explicit sequence/scalar broadcast operation. The linear
Complex construction path is not a second general zip facade.

For a linear two-input chain, a bound Flow uses
`flow.combine(name, side, expression)`, while a deferred Flow uses
`flow.combine(name, side_count, expression)`. Both append one canonical
two-read Map. The side input becomes a positional Program binding; it is not
captured, uploaded per element, or lowered into two one-input Maps. Inputs must
have the same fixed count and scalar width. When both inputs use the same
`Fixed<I, F>` storage, the side binding inherits the active Flow value's
rounding and overflow policy; its own approximation provenance is retained and
the combined result carries the stronger of the two approximation modes. A
different I/F split is rejected instead of being silently reinterpreted or
quantized.

`expand(MaxItems{n}, count, emit)` accepts Exact or Bounded input and has
capacity `input_capacity * n`. For Bounded input the resident logical count is
a canonical scalar dependency; inactive source slots emit zero items without a
host readback. The operation evaluates the per-source count, rejects negative
or over-bound counts with the stable bounded-count contract, enumerates fixed
`(source, local)` slots, selects `local < count(source)`, and uses stable
Partition before emit. Output order is therefore source order followed by
local order on every backend. Work and temporary storage are proportional to
the declared capacity, not the returned logical count; the bound is an
execution cost contract as well as admission.

`join(MaxMatches{n}, right, left_key, right_key, emit)` is an integer-only
bounded stable inner join. Exact and Bounded sides may be combined; independent
resident counts are explicit graph edges and an inactive capacity pair cannot
match. It preserves `(left original index, right original index)` order within
each key and emits keys in canonical ascending order, so
the full order is `(key, left original index, right original index)`. It
rejects any left row with more than `n` matches and returns a resident bounded
sequence. The current canonical lowering examines the
left/right Cartesian candidate space before stable Partition, so its
intermediate cost is `O(left_count * right_count)`; `MaxMatches` bounds the
public result but does not hide that cost.

For Exact or Bounded integer values, `group_by(key).aggregate(...)` performs
stable key sort, active-lineage masking, deterministic segment construction,
and segmented reductions. Inactive capacity never forms a group or contributes
to a grouped collective. Groups are emitted in ascending signed or unsigned
key order, and equal-key values retain source order. The group count and every
aggregate share one resident U32 logical count. Sort positions, compacted head
indices, and that count are all U32. The capacity-wide key Gather masks every
head-index slot at or beyond the resident group count to valid source index
zero; stale compact tail from an earlier many-group resident run is therefore
never dereferenced. An input capacity above `UINT32_MAX`
is rejected from deferred shape metadata with `compute_group_capacity` before
group graph nodes or backend lowering are created. For 64-bit values and keys,
canonical `mask(predicate)`
writes the U32 segment-head stream while a same-width boundary mark feeds
value-domain window selection. The Kernel segmented contract therefore stays
U32 without reinterpreting a U64 buffer. `group.values().map(...).scan(...)
.ordered()` reuses the same stable value order and head stream for a
deterministic segmented transform; Scan resets at every group boundary on
every backend. When that grouped Map itself returns U32, its marks reuse the
already-canonical U32 head stream directly instead of forging a width-changing
value-domain carrier. Its active-count lineage reaches that U32 value stage
through the same public `mask` owner, including from a 64-bit source count; the
same-width-only internal `BoundaryMask` is not widened into a second conversion
authority.
`WindowSpec` carries operation, radius, and `WindowEdge`.
`group.values().window({..., .edge = WindowEdge::Clip}).ordered()` constructs
segment IDs by canonical scan, gathers each fixed-distance neighbor, masks
neighbors from another segment, and merges in left-to-right distance order.
An internal same-width `0`/`1` boundary carrier keeps its boolean bits while
adopting the selected value stage's exact Fixed storage policy; it is not a
numeric integer-to-Fixed conversion and cannot introduce another I/F format.
Its canonical `BoundaryMask` Write shape is `select(X != 0, 1, 0)`, and its
mode and target policy are part of graph identity.
Its work is `O(radius * count)`. Exact and Bounded Flow `window` both accept
`WindowEdge::Clamp` and `WindowEdge::Clip` for `Sum`, `Min`, and `Max`. Clamp
uses the nearest logical endpoint for an out-of-range neighbor; Clip excludes
that neighbor by applying the operation's identity. On a Bounded stage, both
edge modes consume the resident logical count rather than the capacity tail, so
inactive storage cannot affect a window result and no count readback occurs.

An Exact source with physical count zero admits Window Sum for either edge mode
as zero work and returns an empty sequence. Exact zero-count Window Min and Max
reject with `compute_stencil_count_zero` because no element can define their
result. A Bounded source whose resident logical count is zero retains nonzero
physical capacity; all six operation/edge combinations succeed and return an
empty logical sequence.

An empty source is a valid `expand`, `group_by`, or `join` boundary. Its
resident logical count is zero and it contains no value element; no placeholder
element, backend switch, or special host result is introduced. An empty
right-hand join input follows the same zero-result contract.

`filter(predicate)` accepts Exact or Bounded input and returns a real
`stage::Bounded<Count>`. A Bounded input combines the new predicate with its
resident active-count lineage before compaction, so inactive capacity can
never re-enter a later stage. Exact input lowers selected and rejected flags as
two outputs of one predicate Map; CountNonzero Reduce and stable Partition then
own the resident count and ordered payload. The logical count is therefore a
canonical resident output rather than Compact scratch. Elementwise `map`
preserves the logical prefix; one-shot `collect()` and resident `Bounded` Job
read the count only at the explicit terminal and trim the host result. A
type-changing bounded `map` preserves the same resident logical count while
changing the Flow value from `T` to `U` while preserving the same
`stage::Bounded<Count>` typestate. Calling `.count()`
projects that already-resident scalar as `stage::Scalar`; it does not run a
second reduction or download the value before the terminal. Flow compilation
lowers only the selected outputs' reverse dependency closure, so this count
projection also removes the rejected-mask and Partition stages that cannot
affect the scalar. Bounded Reduce and Scan consume the scalar directly in the
Kernel CPU tile/collective plan and in Metal and Vulkan accelerator dispatch.
Their graph and output hashes match, including cross-tile Scan. Stable bounded
Sort also consumes the resident count directly on CPU, Metal, and Vulkan, and
preserves stable source-index tie order. `sort()` and `argsort()` have one
canonical stable ordering law and expose no algorithm-policy object.
Bounded Argsort preserves the original U32/U64 resident-count type through
`stage::Bounded<Count>`. Changing 64-bit keys to 32-bit indices
therefore cannot make the terminal decode the count at the result width.

`compact({.capacity = N})` is likewise an Exact-to-Bounded transition for U32
flags. It returns `stage::Bounded<std::uint32_t>` with capacity `N`, strictly
ordered source indices, and a resident U32 selected count. Underfilled results
contain only their logical prefix; there is no Exact padded form. Count-only
projection retains the Compact capacity guard, an overflow fails the whole run
without publishing a prefix, and bounded downstream operations consume the
same count. `collect()` and resident `read()` transfer the one-element count
and then only the selected prefix, never the unused capacity tail.
An empty U32 source is a valid compact input: default capacity is zero, the
Program uses the canonical backend-free empty path, the value output has zero
elements, and the resident count output remains one U32 whose value is zero.
The host explicitly materializes that scalar zero and the canonical output hash
includes its one-element shape and four zero bytes while recording no backend
download or transfer traffic.
For a nonempty Program that selects zero elements, the terminal reads that
count but performs no value-buffer copy or accelerator download. Empty-Program
introspection keeps these public resource shapes even though its executable
node list is empty, so fingerprints and Program-cache identity agree with the
installed output ABI. The fingerprint still includes the canonical semantic
recipe built by the normal GraphState path: operation, Scan mode, and
primitive-option changes remain distinct even though no backend dispatch is
created.

`source.gather(bounded_indices)` reads only the resident logical prefix and
returns a bounded stage with the same count lineage. Active indices are
validated before output publication. `bounded.indices()` creates a
capacity-shaped ordinal stream carrying that same resident lineage; it does
not manufacture a second count and names worklist-local ordinals, not the
source indices that originally produced that worklist. When source lineage is
required, map the predicate to U32 flags, call `compact` to obtain stable source
indices, and pass that bounded result to `gather`; no host count observation is
introduced.

`unroll<N>(step, converged)` is an active-worklist graph-expansion operation.
Before each body,
stable Filter removes converged items; `step` must preserve the exact typed
worklist, and a final Filter returns the items still active after the last
bound. The first count-aware body node owns that iteration directly: Map,
bounded Scan, and bounded Sort all use the same resident `GraphControl`, with
the collective's existing logical-count binding rather than an added copy or
identity Map. A zero resident count suppresses that body node and later
count-aware primitive work without a host predicate read. This operation is not an
implicit mutable array or a general value phi; authoritative persistent state
is carried explicitly through Program outputs and Pipeline state publication.

`values.scatter_reduce(indices, output_count, Reduce::{Sum,Min,Max})` accepts
an Exact pair or two Bounded stages sharing the same count lineage. It
initializes every output to the operation identity, orders contributors by
`(target index, source ordinal)`, folds each target strictly in source-ordinal
order, then writes each target once. Integer Sum wraps at the declared lane
width. Fixed Sum applies the declared saturation/wrap policy after every
source-ordered addition; it is never reassociated into a tree. Min/Max use the
declared signed, unsigned, or Fixed numeric order. Count overflow and the first
out-of-range active index fail before identity initialization or any result
write, so the operation is failure-atomic. Native completion, subgroup width,
atomic arrival, allocation address, and backend sort implementation cannot
change the result.

Flow exposes LU/QR/Cholesky as distinct typed Factor stages. An unprojected
Factor terminal returns packed/pivot/status for LU and packed/status for QR or
Cholesky from one execution, and resident Jobs support ordered selective
reads. A Factor stage
consumes a matching RHS through
`.solve(rhs, rhs_cols)` (or the deferred Flow overload). The resulting Solve
stage owns values and status; its unprojected terminal returns both from the
same canonical Factor-to-Solve graph. QR uses the separate Q/R payload required
by factor-input Solve, while LU alone carries pivots.

A square Fixed Matrix stage may solve directly from the raw matrix and RHS
without materializing a public Factor stage. Dynamic shape uses
`.solve(rhs, FactorOp, rhs_cols)` and a static shape uses
`.solve<FactorOp, RhsCols>(rhs)`; deferred recipes use the same spellings with
the RHS supplied at Program binding. LU, QR, and Cholesky all lower to one
Matrix-input Solve node with two read inputs and values/status outputs. Rows,
batches, RHS columns, and scalar width must match before backend execution.
For Fixed values, compatibility means equal storage `(I,F)`, rounding, and
overflow; approximation provenance is deliberately independent. An `Exact`
RHS is therefore valid beside a `Deterministic` matrix, and every direct Solve
value output is `Deterministic`.

The CPU matrix-input QR path keeps Q and R in its canonical row-major scratch
and solves from them directly; it does not emit and reread the intermediate
public-layout Q/R payload. Caller matrix, RHS, and output indexing still honor
the selected layout. For `B` square `n`-by-`n` batches and lane width `S`, this
removes `2*B*n*n*S` retained bytes and `4*B*n*n*S` copy traffic bytes from a
Job while preserving batch and reduction order. Public QR Factor output still
materializes the complete selected-layout payload and writes R's lower triangle
as zero at that boundary.

The Factor/Spectrum shape is not a Java-style runtime variant. A static
`Matrix<Rows, Cols, Batches>` carries those dimensions into
`Factor<Op, Rows, Cols, Batches>`, `Solve<Rows, RhsCols, Batches>`, and
`Spectrum<Op, V, Rows, Cols, Batches>`. LU, Cholesky, and Eigen are absent from
overload resolution for a statically non-square Matrix; QR and SVD retain their
rectangular domain. Integer Matrix stages expose none of the fixed-only
decomposition operations. Dynamic Matrix dimensions keep the same transitions
but defer nonzero, square, RHS, and capacity checks to the terminal Result.

Flow `.lu()`, `.qr()`, and `.cholesky()` produce distinct result types
because only LU has pivots. A Factor `.solve<RhsCols>(rhs)` produces a statically
shaped Solve, and `.values()` projects it back to
`Matrix<Rows, RhsCols, Batches>`. `.eigen<V>()` and `.svd<V>()` likewise encode
`Values`, `Thin`, or `Full` vector output in the stage type. Thin SVD vectors
project to `Matrix<Rows, min(Rows, Cols), Batches>`; full vectors project to
`Matrix<Rows, Rows, Batches>`. `vectors()` is absent from overload resolution
for values-only results. Values, optional vectors, and status compile or
collect as one ordered output set. These static and dynamic spellings append
the same canonical nodes and therefore have the same graph identity.
32- and 64-bit `Fixed<I, F>` storage formats LU, QR, Cholesky, factor-input and matrix-input Solve, SVD
Values/Thin/Full, and Eigen Values/Full ordered results match CPU on Metal and
Vulkan in both Standalone and explicit Node-native execution. Factor,
square Spectrum, and
rectangular Full SVD contracts cover graph and output hash parity; their second
resident run reports zero `pipeline_compiles`, `buffer_allocations`,
`download_events`, and `uploaded_bytes`. QR, Cholesky, and Spectrum use only
canonical fixed arithmetic.
Jacobi rotation uses an algebraically equivalent quarter-scaled form so every
intermediate stays representable in the declared `(I, F)` format, and equal singular values use
the same strict comparison and stable traversal order on every backend.
Vulkan QR, Cholesky, matrix-input Solve, SVD, and Eigen shaders use the same
fixed-only arithmetic, quarter-scaled Jacobi rotation, strict ordering, and
deterministic basis completion. Generated 32/64-bit Factor, Solve, and Spectrum
SPIR-V validates for Vulkan 1.1, and the resulting pipelines pass the same
Standalone and Node-native parity contracts as Metal.

Vulkan numeric execution has one parallel topology rather than a serial shader
variant. Fourier transform uses the canonical fixed schedule: one 256-lane
local pass performs bit reversal and the first eight radix stages, then global
passes consume adjacent stage pairs while retaining each intermediate value in
a register. Factor, Solve, and Spectrum use one 32-lane workgroup per batch.
Pivot selection, dot products, norms, and
comparison order remain ascending on lane zero because saturating fixed-point
addition is not associative; independent rows, columns, matrix cells, RHS
columns, rotations, and output cells are lane-strided. Direct LU and QR Solve
reuse the same factor-then-substitute arithmetic order as the reference path;
there is no second Gaussian-elimination or approximate-QR authority.

Arithmetic identity is fixed while an FFT stage exposes up to
`min(256, N/2)` independent butterflies. Batched
algebra exposes one workgroup per batch and up to 32 independent cells inside a
dependency stage. The scalar ordered reductions remain the determinism-critical
span, so these bounds are not a wall-clock speedup claim; small matrices may be
barrier dominated. `accel.kernel-numeric` checks a 512-element 32/64-bit FFT,
9x9 three-batch Factor and three-RHS direct Solve against the canonical CPU
reference, a 9x9 three-batch non-diagonal Spectrum canary, and generated-source
topology. Timing remains owned by `tools/measure/compute/run` and the sealed
performance packet.

All primary stages share one canonical Kernel graph identity. `collect()` remains
the only cost-bearing terminal and never retries on CPU if Metal is
unavailable. Scan uses its collective planner across tile boundaries; it is
not treated as independent per-tile Map work.
Flow-to-Graph lowering preserves the exact canonical rejection code and reason
from Map, Scan, and primitive admission; it does not replace an unsupported
operation option with the generic `compute_graph_binding_invalid` reason.

Graph identity uses canonical graph value IDs, including deterministic IDs for
planner-owned auxiliary outputs. Resident handles, allocation order, backend
object IDs, and device-local binding order never enter that identity. The
Compute bridge supplies the same logical IDs to CPU and accelerator graph
validation; an internal Accel graph derives graph-local IDs only when no
canonical Compute ID was supplied.

Map expression validation and `ComputeOp` construction have one backend-free
owner shared by Flow graph description, graph compilation, and graph identity.
Flow recipes and compiled Graph construction also share private `MapStep` and
`ScanStep` layouts. Map owns its diagnostic name, input/output value-ID routes,
and expressions; Scan owns its input, output, active-count route, and operation.
Neither record has a second Flow-only or Graph-only definition.

Private `compute/type.hpp` is the sole `Type` projection authority. Its
constexpr table maps the six storage types to byte width, validity, Kernel
scalar width, and Kernel arithmetic domain. Device and Buffer state do not
re-export those rules; every implementation that consumes a projection
imports the leaf directly. Map IR canonicalization and primitive graph
construction therefore use the same `type_domain()` result instead of
reconstructing domain switches.

Private `compute/graph/scan.hpp` likewise owns the one constexpr
`Type -> kernel::ScanElement` projection. Canonical graph description and
backend compilation consume that same result, so the `ScanDesc`, hash,
rejection precedence, and runtime control branches are identical in both
paths. The ownership change adds no public type, graph node, or execution
branch.

One-shot collection, direct buffer execution, and resident execution all
compile the same Flow-to-`GraphState` representation. Canonical description
builds each operation once and owns it until a cache miss consumes it; a cache
hit discards that temporary operation with the description. CPU graph
compilation consumes the operation once: it copies only the `ComputeMap`
descriptor, uses the operation's metadata view while validating probe bindings,
and builds the compact SIMD `PreparedRun`. Accelerator graph compilation lowers
the complete graph with one `CompileAccelKernel` call. Program publication
retains the compact runtime owners described below.

Flow liveness records each live Map's used-input and live-output masks while it
validates the expression DAG. Its transient projection plan retains one
1024-bit reachability set, exact node extent, live-node count, and live-root
mask per distinct `ExprState`. Canonical ordering and Graph projection consume
that plan directly. Live roots that share one state are handled as one union:
liveness marks the union in one reverse pass, then projection copies and
renumbers it in one forward pass. There is no second reachability walk,
reserve-count walk, or per-group node-map clearing pass. For a group whose
highest live root has canonical extent `N`, the expression-node work is
therefore exactly two `N` traversals plus at most 16 root visits and 120
bounded group-identity comparisons. Flow binding admission
likewise validates a shared `ExprState` once at its first output occurrence
while retaining root-specific checks in output order; for `E` roots sharing
`N` stored nodes, that changes validation from `E * N` node visits to `N` node
visits plus `E` root checks. These are traversal-count models, not wall-clock
speedup claims.

ComputeOp validation collects fixed-input formats in expression/node order
into one global 16-entry table. The same single structural pass records one
transient 16-entry fixed-input summary per distinct state; after all states
establish the global first-observed formats, cross-state validation examines only
that summary and never rescans a node vector. For `E` roots sharing one
`N`-node state, validation performs `N` node visits plus at most 16 summary
checks. The first observed format and first state occurrence own rejection
precedence; collection is linear in total expression nodes and every lookup is
constant time.

Accelerator Map compilation has one admission and emission authority. It
parses each source Map once and retains that typed admission until the final
graph steps are fixed. The compiler scans graph order once and folds every
maximal legal straight-line Map region of length `K >= 2`; each region
serializes only its final fused IR and emits one backend artifact. It does not
first emit `K` artifacts, repeatedly copy a growing fused prefix, or discard
intermediate artifacts. The planner carries every Map's exact binding and IR
node costs and starts a new maximal region before the canonical 64-binding or
1024-node limit would be exceeded. The fused step removes all `K - 1`
internal Map boundaries; for `N` elements of width `E`, that removes the
algorithmic intermediate round trip `2(K - 1)NE` bytes. This is a traffic
model, not a hardware-cache or elapsed speedup claim. A rejected or capacity boundary
terminates only its current region; the boundary steps remain original while
later legal regions are still fused. In every case, validated graph order is
execution order; runtime preparation only validates final binding occurrences
linearly and never introduces a second graph scheduler or reordering
authority. The compile owner stores only the dispatches removed by each fused
region. Runtime derives public `original_dispatches` as
`final_dispatches + removed_dispatches`, avoiding a second collective dispatch
table or whole-graph recount.

The 1,024 value above is the inner ComputeIR limit, not the outer Program
schedule limit. A Program admits 16,384 ordered graph nodes and derives its
1,048,640 logical-value construction envelope from 64 refs per node plus 64
outputs. The graph, hazard description, and fusion workspace retain only their
actual authored sizes. Thus a large action stays one canonical Program and one
backend submission; only its maximal legal Map regions become distinct kernel
steps, under one graph order and fingerprint.

A CPU Program uniquely owns a dedicated `CpuRuntimeGraph`. Its values contain
type and count; its Map steps contain input/output value-ID routes; and each
Primitive step contains its routes, selected output, operation kind, and one
validated active Kernel plan. Primitive planning is frozen once during
compilation and consumed by warm runs. The caller's Flow graph remains reusable.

Compiled Flow storage follows those logical IDs. Let `V = E union K`, where
`E` is the set of external input/output values, `K` is the set of graph-internal
values, and `b(v)` is the logical payload of value `v`. The canonical storage
plan partitions `K` into deterministic 256-byte-aligned physical arenas `P`.
Only values with disjoint closed live intervals share a range, except for the
single proved same-node destructive Map transition recorded by
`Resource::source`. That transition is admitted only for a pointwise Map:
every read from the aliased source at output ordinal `i` must read source
ordinal `i`. The Map has one eligible dense full-write output; when several
arena-backed same-shape inputs end their lifetime at that node, the smallest
canonical resource ID is the source. A `ReadAt` indexed Map has no destructive
transition because shape equality does not prove identity indexing or exclude
a cross-lane read-after-write race. Active-prefix storage may enter the arena
when its consumer count is the same count or a descendant in the stable Filter/Compact
count lineage. Every partial-write value owns one exact reset range. Several
such ranges may reuse the same aligned offset when their closed lifetimes are
disjoint. The later range is cleared at its exact first-writer frontier after
the earlier range's last use. Physical-owner equality never deduplicates their
logical reset routes: for the reset set `R`,

```text
reset_bytes = sum(r in R, r.bytes)
reset_count = |R|
```

Padding and unrelated arena intervals are not reset. A Program persistently
owns those immutable routes plus the offset and chunk plan, not their physical
payload. Every ordinary execution is one Job authority, and that Job owns its
own `P` storage so concurrent Jobs
cannot alias an intermediate. External input and output routes point to that
Job's binding owners. Accelerator Jobs also own fixed-view binding storage and
prepared backend state. The exact first Write carries `BufferInit::Zero` into
the canonical Kernel graph; the accelerator token derives the post-fusion
execution frontier once, and an invocation supplies no reset coordinates.
Program publication therefore retains no physical internal buffer, binding
scratch, accelerator reset mirror, or second execution implementation.

For `C` writable resident Jobs, the exact graph-and-binding payload is
`C * (sum(P) + 2 * sum(inputs) + sum(outputs))`. The two input terms are the
active and pending-write sets that make `write()` allocation-free and atomic.
The convenience `Program::run()` surface serializes and retains at most one
read-only Job. A host-input cache owns
`sum(P) + sum(inputs) + sum(outputs)`; a caller-Buffer cache owns only `sum(P)`
because the external buffers remain caller-owned. Switching between those two
convenience forms replaces the one cache instead of accumulating another
runner. Bounded or ordered multi-output terminals use a short-lived read-only
Job with the same `sum(P) + sum(inputs) + sum(outputs)` law. Thus no physical
memory term exists until an execution owner exists. `graph::MemoryPlan`
retains `sum(K)` as `logical_bytes`, the maximum closed-interval live sum as
`live_bytes`, `sum(P)` as `physical_bytes`, and the retained Buffer owner count
as `allocation_count`. The reduction is public plan evidence rather than an
inference from allocator behavior. The complete proof and backend offset law
are owned by [Compute Graph Services](./compute/services.md#generic-resource-graph).

A Pipeline consumes the same Program chunk law. Ordinary chunks may share one
arena owner capped at `min(1 GiB, backend storage limit)` with
256-byte-aligned offsets. A Program chunk larger than that ordinary ceiling
remains one offset-zero dedicated owner; it is never subdivided or mixed with
another live chunk. Pipeline planning does not coalesce frozen owners into a
larger backend-limit-sized Buffer. Nonoverlapping oversized values already
reuse one owner inside the Program, and the same owner may be reused again by
a later nonconcurrent Pipeline step. `PipelinePlan::largest_bytes`, `largest_step`,
`largest_iteration`, and `largest_chunk` identify the largest single Program
workspace before that cross-step reuse. This makes an oversized workspace
diagnosable without exposing allocator addresses or creating another size
limit.

`PipelinePlan::peak_bytes` reports the exact Pipeline-owned planned payload:

```text
peak_bytes = state_bytes + transient_bytes + prepared_bytes
```

`prepared_bytes` is the aligned backing payload of the typed dense View and
primitive scratch arenas referenced by prepared backend commands. Multiple
logical View slots and ordered primitive temporary requests are suballocated
from shared owners at the required scalar and selected backend storage
alignment. Sequential uses of different scalar types may share the same
raw-word slot.
The arenas are planned and allocated once by Pipeline, shared by sequential
Programs and recurrence phases, and never privately allocated by a prepared
Job. `scratch_bytes` and `scratch_count` expose the retained scratch payload and
physical page count. Because Pipeline steps are serial, scratch capacity is the
maximum deterministic Program page envelope rather than the sum of every
prepared occurrence. `view_bytes`, `view_step`, `view_iteration`, and
`view_binding` identify the largest logical View requirement without changing
the Program-workspace coordinates. `view_span_bytes`, `view_backing_bytes`, `view_offset_bytes`,
`view_stride_bytes`, `view_element_bytes`, `view_count`, and
`view_alignment` describe the exact selected binding. `DeviceInfo` exposes the
selected backend's `storage_alignment` and per-binding `storage_bytes` limit.
`persistent_bytes` reports referenced caller Buffer storage, and `total_bytes`
is the checked sum of persistent and peak bytes.

`MemoryBudget` compares `peak_bytes` before any Pipeline-owned View, scratch,
state, or workspace Buffer is materialized because caller Buffers already
exist and are not allocated by prepare. Planned payload cannot be mixed
dimensionally with backend allocation rounding, driver metadata, or
transfer-pool high-water marks. Those are reported after prepare by
`Pipeline::memory()`. `memory_snapshot()` labels scratch separately in
Resident and Device categories so logical payload and physical allocation
rounding remain distinguishable. Allocation failures retain the allocator or
backend's typed reason. runD does not invent a platform-memory capacity.

The public Device, Buffer, and Program handles do not mirror backend, buffer
count, or Program shape metadata. One shared state owns those values, and the
handle's observation and binding checks read that same owner. `Device` and
`Program` expose `valid()` and explicit boolean observation. A moved-from
`Device::backend()` fails with `DeviceInvalid`; a moved-from
`Program::backend()` and every execution or resident admission fail with
`ProgramInvalid` before shape validation. Neither handle substitutes CPU,
opens another backend, retries, or publishes neutral metadata as a successful
backend selection. The four typed `Program` result shapes share one internal
handle owner for validity, backend, graph, fingerprint, and memory observers;
specializations own only their shape-dependent admission and result decoding.
Program
input types and extents are one paired, nonempty shape authority with equal
cardinality; execution and resident-write admission do not retain a scalar
first-input fallback.
Publishing a Program therefore adds one shared-state handle in `O(1)` time
rather than copying input/output extent arrays that compilation already
produced.

One immutable value-ID route table is the authority for synchronous CPU,
asynchronous CPU, convenience accelerator, and resident accelerator binding. Lookup
is O(1) by `value_id - 1`; hot execution borrows raw buffer pointers while the
Program, Job, or CPU run retains ownership, so lookup does not increment a
`shared_ptr` reference count. This execution-only route does not enter graph
identity, ABI serialization, or output hashing. A CPU Map Job consumes this
route table once to freeze its owner addresses, byte strides, and port counts.
Stable warm runs compare the active input-owner set in `O(1)` and reuse that
projection; a successful resident `write()` causes exactly the next run
admission to rebuild it for the newly published owner set. Accelerator
compilation accepts external value-ID plus shape references and therefore
allocates no physical `E` placeholder. The compiled kernel token freezes scalar
width, element count,
policy, and required read/write capability for every binding; run admission
validates the actual owner and that frozen shape before forming a resident
reference. Internal references carry only canonical logical identity and shape
until a Job allocates their physical `K` buffers. Pointer-bearing graph node/ref arrays are compile-local and are
released before the Program is returned. Accelerator Programs and Jobs persist
only the owners assigned by this boundary.

## 3. Compile Once and Run Resident

Build a deferred Flow with `compute::on(target).map<T>(name, count, fn)` and
call `.compile()` when the same fixed-shape graph runs repeatedly. Compilation,
buffer allocation, and ordered initial active-input upload submission occur
before the warm loop. A full Vulkan upload does not wait on a host fence: its
command slot retains staging and native target storage through completion. If
all eight slots are active, upload waits on the slot condition instead of
failing or creating another backlog. The first execution evidence boundary
drains that queue order before resetting warm counters. Full-overwrite
input-binding allocation does not zero bytes that the guaranteed
complete active upload or pending `write()` immediately overwrites. Resident
setup allocates two input binding sets but does not copy the initial payload
into the inactive set, so a later multi-input write can commit atomically
without allocating in the warm path. In contrast, every successful public
`Device::buffer<T>(count)` publishes all `count * sizeof(T)` bytes as zero on
CPU, Metal, and Vulkan, including when native capacity came from a reuse pool.
The clear occurs only at the cold logical-ownership transition; a proven
full-overwrite `Device::upload` skips it and writes exactly the caller payload.
Allocation initialization is not reported as upload/download payload traffic.
`Job::run()` submits only the already prepared resident graph,
and `Job::read()` is the explicit host download boundary.

Bounded and ordered multi-output `collect()`/`Program::run()` are read-only
one-shot terminals, not short-lived writable resident Jobs. Their private
execution owner allocates only active inputs, physical outputs, and graph
internals, then prepares that binding set once. Destruction releases all three
sets after the terminal read. Calling `resident()` remains the sole way to pay
for and retain the inactive input set and its separately prepared backend state.

The single-output convenience cache is thread-safe but intentionally
serialized per Program: concurrent callers cannot mix host inputs, Buffer
identities, or receipts. `Program::memory()` joins the same gate, so observation
is a coherent snapshot rather than a race with cache replacement. Independent
throughput work should use explicit resident Jobs, which do not share that gate.
The cache borrows its enclosing Program to avoid an ownership cycle; every
returned Buffer-backed `Run` takes its own strong Program reference and remains
readable after the original Program handle is destroyed.

`Stats::output_hash` remains zero after `run()` because host-visible output
evidence must not introduce an implicit readback. A single-output `read()`
records that output's content hash. For an ordered multi-output Job,
`read_all()` records one canonical hash over output index, scalar type, element
count, and content hash after every leaf has been read. Selective `read<I>()`
keeps the aggregate hash zero until all leaves have been observed; read order
cannot change the final hash. Every backend performs its canonical byte-order
hash while copying from resident or staging output into the typed result, so it
does not reread the completed result payload. An all-zero result derives the
same FNV-1a value by exponentiation by squaring rather than scanning the bytes a
second time.

The prepared resident surface admits CPU graphs through the canonical Kernel
plan. Metal and Vulkan prepare every valid row in the checked-in
primitive/domain matrix with reusable pipelines, binding state, scratch, and
completion slots. A missing prepared implementation is a contract failure,
reported through the canonical `compute_lowering_invalid` boundary rather than
a supported resident mode. Invalid shapes, domains, bindings, and unavailable
explicitly selected backends still fail closed with their stable reason and
never enter an allocating fallback path.
Metal Sort owns its native histogram, prefix, base, and stable-scatter pipeline
set before the warm loop. Partition owns its prepared scan pipeline set and
preserves the payload width independently from its u32 flags,
including 64-bit integer and 64-bit Fixed<I, F> bit patterns on every backend.

```cpp fragment
auto program =
    rund::compute::on(rund::compute::Target::cpu(4))
        .map<std::int32_t>("twice", input.size(),
                           [](auto value) { return value * 2 + 5; })
        .compile();
if (!program) { return; }

auto job = program->resident(input);
if (!job) { return; }
for (std::size_t iteration = 0; iteration < 100; ++iteration) {
  if (!job->run()) { return; }
}
auto output = job->read();
```

### Batch small accelerator Jobs

`compute::Batch` reduces fixed native submission cost when several small,
independent resident Jobs are ready at the same boundary. It holds at most 64
Jobs in inline storage and accepts only Jobs from the exact same opened Device.
Graphs, Programs, signatures, shapes, and pipelines may differ. The shared
operation is one command submission; Batch does not fuse graphs or dispatches.
CPU Jobs fail with `BatchCpuUnsupported` because CPU execution has no native
GPU submission to amortize.

```cpp fragment
auto device = rund::compute::open(rund::compute::Target::metal());
if (!device) { return; }

auto first_program =
    rund::compute::on(*device)
        .map<std::int32_t>("first", input.size(),
                           [](auto value) { return value * 2; })
        .compile();
auto second_program =
    rund::compute::on(*device)
        .map<std::int32_t>("second", input.size(),
                           [](auto value) { return value + 7; })
        .compile();
if (!first_program || !second_program) { return; }

auto first = first_program->resident(input);
auto second = second_program->resident(input);
if (!first || !second) { return; }

rund::compute::Batch batch;
const auto first_added = batch.add(*first);
if (!first_added) { return first_added.exit_code(); }
const auto second_added = batch.add(*second);
if (!second_added) { return second_added.exit_code(); }
const auto completed = batch.run();
if (!completed) { return completed.exit_code(); }

auto first_output = first->read();
auto second_output = second->read();
```

`run()` claims every Job before changing any Job phase or output. Empty, full,
duplicate, cross-Device, CPU, busy, and invalid prepared inputs have distinct
typed Batch reasons. If a claim fails, no Job executes and all preceding Job
evidence remains unchanged. Metal uses one command buffer and commit; Vulkan
uses one bounded primary command slot, executes each Job's preparation-time
secondary command, and calls `vkQueueSubmit` once. Completion and failure
projection remain in `add` order, including an exact overflow from one Job
while independent later Jobs still finish after the already submitted command.

`Batch::stats()` is the sole owner of the shared execution submit, queue
pressure, and backend timing. Those values come from the completed Batch
command, never a reset/read pair over adapter-global counters; running a Batch
cannot drain unrelated prepared Jobs merely to isolate telemetry. In the
snapshot immediately after `run()` and before readback, each `Job::stats()`
retains its own graph, dispatch, and result evidence while shared execution
submit/timing fields stay zero. Output hashes
still appear only after the corresponding explicit Job read; an accelerator
read may record its own transfer command and does not retroactively duplicate
the Batch execution submit. Batch graph and output hashes stay zero because
Batch does not invent an aggregate graph or result.

For `N` Jobs with fixed submit-and-wait cost `S` and per-Job warm wrapper/device cost
`C`, serial cost is `N(S + C)`, Batch cost is `S + NC`, and speedup is bounded
by `N`. At capacity 64, 30x requires `S / C >= 1856 / 34`, approximately
`54.59`; this is not a claim without a same-input measurement. The official
Compute route records serial and Batch end-to-end wall medians over the same
prepared 64-Job set using four AB and four BA pairs after both paths are warm;
oracle reads happen only after timing. Submit-wait, kernel, and
`max(wall - submit_wait, 0)` are diagnostic medians. Vulkan already records
each Job's dispatch/barrier stream once, reducing warm host command construction
from `Theta(D)` for `D` native commands to one primary execute wrapper; it does
not change kernel work. A remaining dominant host residual points to the Metal
encoding path or higher-level orchestration, while dominant submit-wait
confirms submission amortization as the next structural lever. The exact runtime contract is
[Compute Batch](../../node/docs/contracts/compute/batch.md).

Memory observation is explicit and allocation-free. `Program::memory()`,
`Job::memory()`, and `Device::memory()` return a fixed-size
`MemoryStats` snapshot. Its host, coroutine-frame, tile, resident, staging,
device, and transfer categories are independent dimensions rather than an
additive total. Each category has current, peak, cumulative, reused, and budget
bytes. In particular, resident describes the logical byte extent kept bound by
the Job, while host/device describe the physical SDK-owned storage. Node-native
Compute attributes the actual coordinator coroutine-frame extent and arena
reuse to the same Job; a completed task returns frame current to zero without
discarding peak or cumulative history.

Resident `CpuGraphRun` intermediate buffers are Job-owned resident storage, not
tile scratch. `Job::memory()` therefore includes them in resident and physical
host totals, and `memory_snapshot()` emits one `MemoryUse::Internal` row per
retained `K` value. `Program::run()` uses the same Job implementation through
one serialized cache. Before its first convenience run, a Program snapshot has
no physical Internal row. Afterward, `Program::memory()` includes the cached
Job's physical buffers, prepared state, and tile scratch exactly once; replacing
the cache also replaces those rows. An `E`-only execution has no Internal row.

CPU executor memory has one Kernel-owned byte oracle:
`ComputeTileExecutor::retained_memory()`. `Program::memory()` counts every
compiled Map and collective plan executor once, then adds the one cached Job
when present. `Job::memory()` counts only that Job's run executors; it does not
repeat immutable Program plans.
The PIMPL State and optional asynchronous RunContext are Host/Metadata, while
the Workspace capacities plus failure and per-worker tile arrays are
Tile/Scratch. Operation-specific vectors remain Tile/Scratch. Observation
walks each CPU owner once and uses actual vector capacities, so a snapshot does
not reconstruct private Workspace shapes or execute separate metadata and
scratch passes. All retained executor capacity products and sums saturate at
the maximum unsigned 64-bit byte count. The observation remains allocation-free
and stable across warm execution.

Program Host/Metadata is an all-owned tree, not a partial `sizeof` estimate.
The root includes `ProgramState` and every top-level vector capacity, then the
nested capacities of `graph::Info` resources, nodes, accesses, dependencies,
barriers, inputs, and outputs. A CPU Program adds one `CpuGraphProgram`, its
uniquely owned `CpuRuntimeGraph` and nested route vectors, every live `CpuProgram` or
`CpuCollective`, and each compact prepared plan plus Kernel tile executor. If
`V(x) = capacity(x) * sizeof(x::value_type)`, `I` is the prepared instruction
vector, `F` is its fixed-format vector, and `E` is the executor byte oracle, one
Map contributes exactly
`sizeof(CpuProgram) + V(I) + V(F) + E.state_bytes +
E.async_context_bytes` to Host/Metadata and
`E.workspace_bytes + E.failure_slot_bytes + E.worker_tile_bytes` to
Tile/Scratch. The descriptor, dispatch pointers, counts, flags, and scratch/tile
scalars are inline in `sizeof(CpuProgram)`. Dynamic Map ownership is exactly
`V(I) + V(F)` plus the executor extents above. For runtime graph `G`, its exact
logical owner extent is
`sizeof(G) + V(G.values) + V(G.steps)`, plus `V(inputs) + V(outputs)` for every
live Map and `V(inputs)` for every live Primitive step; the one active Primitive
plan is inline in the step variant. Outer vector capacity accounts for inline
elements exactly once; only live elements are walked for nested allocations. A
vector term uses actual retained capacity. A string contributes zero dynamic
bytes when its data is
inside the string object and otherwise contributes `capacity + 1` logical
character slots including the terminator. Allocator rounding and bookkeeping
are excluded. Every multiplication and addition saturates at `2^64 - 1`.

For `N = I.size()`, `M = N + 1`, SIMD lane count `L`, the raw per-worker Map
scratch request is
`M*sizeof(uint8_t) + M*sizeof(ValueVec) + M*L*sizeof(WideScalar) +
alignof(ValueVec) + alignof(WideScalar) + alignof(uint8_t)`, rounded up to
`sizeof(std::max_align_t)` words. Instruction planning is one `O(N)` Program
preparation operation; a run binds its fixed views in `O(binding_count)` and
physical tiles consume the prepared schedule. The exact owner and scratch
derivations are recorded in the
[Node Compute memory contract](../../node/docs/contracts/compute/memory.md).
These are structural bounds, not wall-clock benchmark claims.

An accelerator Program adds one `AccelProgram` and the authenticated immutable
`AccelKernel` token owner. All run-binding capacity and prepared resources
belong to a Job. A retained convenience Job is nested once in Program memory;
resident Jobs report the same owner classes in Job scope. Token accounting covers
the token object, graph role/shape/visibility/alias vectors, one final
execution-step vector, every nested final lowering artifact and
execution-metadata string/vector, and binding-index overflow storage. A fused
token's retained owner is that final tree plus one inline precomputed
original-dispatch-count scalar. The token is authenticated through its private
`shared_ptr` control-block capability and measured once at compilation;
`memory()` reads the frozen byte result without a process-global token table, mutex, or
weak-entry lookup. Shared Device/context owners, separate context and resident
registries, allocator bookkeeping, and their control structures are not
Program owners and are excluded. The kernel token itself has no registry entry.
A live Job's internal `P` payload remains in the Host or Device physical
category and is never repeated in Host/Metadata.

The scope is `Program` for Program ownership, `Job` for resident Job ownership,
and `Backend` for Device/backend ownership. A default, invalid, or moved-from
owner returns an all-zero snapshot with backend `Unavailable` and scope
`Unspecified`; `available()` is false and that value never describes live
memory ownership. A live snapshot has a concrete backend, a concrete scope,
and `available()` true.

`memory()` itself performs no readback and does not mutate execution evidence.
Transfer current is zero outside an active operation, while cumulative records
the bytes moved by explicit resident setup, `write()`, and `read()`. Resident
setup transfers the caller payload exactly once into the active input set. For
inputs `i`, its transfer byte count is
`sum(count_i * element_bytes_i)`, even though the Job retains both active and
inactive input storage. A zero budget means no byte budget was configured for
that scope. Program and
Job-owned fixed allocations report their committed extent as both
current and budget; accelerator backend device and transfer categories report
the frozen device-capability byte limit. Backend-private
transient staging remains distinct from resident storage and is reported only
when its allocator exposes a measured byte extent; allocation counts are never
misreported as bytes.

Metal staging counters come from their actual transient lease/release paths,
including reuse. Vulkan has three disjoint allocator classes: mapped coherent
staging, unmapped device-local resident payload, and unmapped device-local
internal scratch. Pool reuse includes the class in its key. Explicit Buffer
transfer normalizes arbitrary byte ranges to Vulkan's four-byte copy alignment;
semantic byte telemetry excludes padding and internal boundary preservation.
The public resident handle is the Vulkan native-buffer owner; the adapter hash
registry is weak, drops the row on final release, and cannot retain dead
payload. Full uploads submit their staging owner into the bounded completion
ring, while downloads and unaligned edge preservation remain host observation
points.
Internal status atomics also stay in device-local scratch. Completion copies
only their reduced first U32 into a mapped four-byte staging owner before the
host reads the reason, so discrete GPUs do not publish workgroup status through
host-coherent memory.
Allocator classes, retained Job memory, warm stability, and collective scratch
ownership are covered by executable Vulkan contracts. On Apple those contracts
execute through MoltenVK and prove the translated path, not native Vulkan
throughput.

For allocation-level inspection, pass bounded caller storage to
`memory_snapshot(entries)`. The result reports how many entries were written,
the total available row count; `truncated()` derives exactly `written < total`.
Entries identify their category, semantic use, stable local index, and byte
counters without names or owning memory. No snapshot storage is retained by
Program, Job, or Device. For a complete Program or Job snapshot, summing all
rows in each category reproduces that category's summary counter exactly; this
is the executable boundary between Host/Metadata and physical Internal rows.
The memory contract checks that equality without mirroring the implementation's
ownership formula. It separately proves zero retained delta for diagnostic Map
label changes, positive compact-plan delta for deeper IR, a route/`graph::Info`
binding delta for an additional external input, topology delta for an additional
step, exact compact runtime-graph ownership, allocation-free observation,
stable convenience/Job warm memory, and a representative authenticated accelerator
token owner.

When values change but the graph and every input shape stay fixed, update the
existing resident bindings instead of constructing another Job:

```cpp fragment
if (!job->write(next_input)) { return; }
if (!job->run()) { return; }
auto next_output = job->read();
```

For a multi-input Program, `write(first, second, ...)` accepts the complete
typed binding set. Resident setup allocates and prepares two input binding sets,
but initializes only the active set from the caller payload. Preparing the
inactive set authenticates its storage and bindings; its payload has no semantic
meaning before a successful `write()`. A write validates every type and count,
overwrites the complete inactive CPU or accelerator set, and commits by swapping
the active binding set. Only a successful commit invalidates the preceding
output; a failed transfer preserves both active inputs and output.
`write_stats()` reports transferred bytes and CPU copies or accelerator uploads.
It performs no compilation, buffer allocation, or backend change.
Writing a queued, running, or already-writing Job fails with
`compute_job_busy`.

Inside a native scheduler task, await the same submission directly:

```cpp fragment
auto result = co_await session.compute(*job);
if (!result) { co_return; }
```

Constructing or discarding the awaitable request has no execution side effect.
Submission and scheduler registration happen in the coroutine awaiter's
`await_suspend`; only an actual `co_await` admits the coordinator. This keeps
the standard coroutine suspension point as the sole owner of NodeHost
execution admission.

The Node-host type path is one sequence:

```text
Request -> Submission -> Poll | Completion
   |                         ^
   +---- direct co_await ----+
```

`Session::compute(job)` returns `rund::compute::Request`; `submit()` returns a
move-only `Submission`; `poll()` returns a `Poll` snapshot,
`wait_for(duration)` returns a bounded-wait `Poll`; and `wait()` or direct
coroutine await returns `Completion`. These names share
the existing `rund::compute` product namespace with `Job` and `Result<T>` but
do not duplicate either value-result or execution authority.

The same resident `Job` can be submitted explicitly to a configured Node
runtime without rebuilding its graph, artifact, bindings, or buffers:

```cpp fragment
auto report = rund::run(config, [&](rund::Session& session) {
  auto result = session.compute(*job).submit().wait();
  if (!result) { return; }
});
if (!report) { return; }
auto output = job->read();
```

`Job::run()` is the synchronous StandaloneHost terminal.
`Session::compute(job).submit()` is the explicit NodeHost terminal. NodeHost
owns scheduling and lifecycle only; it cannot compile a graph, create or
upload a buffer, select another backend, or read output. A submitted task
retains the opaque Job state, so moving or destroying the public Job wrapper
does not invalidate accepted work. `read()` while queued or running fails with
`compute_job_running`.

Both terminals enter the same prepared submit/completion owner. The standalone
terminal adds only a stack-owned atomic wait for that completion; it has no
backend-specific prepared-run hook, global telemetry reset, or alternate
evidence projection. CPU may complete inline, while Metal and Vulkan publish
through their native completion paths. The release/acquire completion edge
makes the one resulting `AccelEvidence` snapshot visible before either terminal
returns.

CPU synchronous execution and NodeHost progression also consume one
`start -> finish` graph-step transition owner. That owner alone prepares the
next canonical step, preserves a Scan's local-before-correction pass, and
decides completion. The two terminals differ only in whether each prepared
pass is run inline or submitted to the WorkerBackend; they cannot diverge in
step order, control observation, cancellation boundary, or statistics
publication.

`submit()` admits one coordinator in the Session's bounded Scheduler and
returns without making `wait()` the execution-start trigger. `poll()`, an
external `wait()`, and `co_await session.compute(job)` all observe that one
Scheduler completion cell; Compute has no second future-like terminal flag.
`rund::compute::Poll::submitted` reports Session admission, while
`rund::compute::Poll::backend_submitted` becomes true only after the prepared CPU or
accelerator backend accepts the execution. Both fields are snapshot
observations. `reason()` is the poll's only failure authority; `code()` and
`error()` are derived from it. A live admitted task reports `Reason::Ok` until
terminal publication. An immediate admission failure is already terminal, so
coroutine await never constructs a second join for work that was not admitted.
`rund::compute::Completion` exposes that same `reason()`/`code()`/`error()` projection with
the execution statistics and derives `exit_code()` for an application terminal.
CPU tile completion or an accelerator callback wakes the exact parked
coordinator once. Cancellation is resolved on that resumed task before the Job
can publish success. Cancellation acceptance and terminal publication compete
through one atomic state transition: a successful `cancel()` therefore cannot
be followed by a successful result.
CPU cancellation is observed before submission, after each deterministic tile
epoch, between graph steps, and between collective passes. The same
`compute_cancelled` reason is selected independently of worker completion
order. Accelerator cancellation is observed before dispatch or after backend
completion; an already submitted command is allowed to finish and its result
is discarded.
Session telemetry also applies to NodeHost Compute. Each admitted terminal
emits one `rund::telemetry::Event` with common source, level, session, scope,
and the nested `compute` projection for its typed code, backend, graph, worker,
tile, dispatch, command-submit, buffer-allocation, kernel-time, and
submit/wait evidence. The sink runs
under the Session reentry guard before the task is externally complete and
before its task slot is released. Selection cost, inactive projection, and
Basic/Detail parity are owned by the
[Telemetry contract](../../node/docs/contracts/telemetry.md).

The projection additionally carries buffer reuse, the saturating sum of this
run's explicit upload/download boundary bytes, and `Stats::graph_read_bytes`.
`graph_read_bytes` is derived once from `graph::Info::read_bytes` as the sum of
canonical read-access byte ranges and is propagated unchanged on CPU, Metal,
and Vulkan. It is exact graph-plan evidence, not inferred post-fusion
hardware traffic. `Event::findings()` maps that raw authority to
`Cause::GraphRead` without storing a duplicate mutable counter.
`Job::profile().findings()` exposes the same bounded actionable projection to
Standalone Compute users; it does not require a Session or introduce a second
finding model.

A warm resident `run` performs no pipeline compilation, SDK-owned heap growth,
backend buffer allocation, upload, download, or implicit readback.
GPU drivers may create an opaque command object for each submission; that
process-wide allocation is distinct from SDK warm growth and backend buffer
allocation evidence. Before the first run, `Job::stats()` already reports the
selected backend with zero execution counters. It then reports the latest run
counters, graph identity, and the output hash only after `read()`. If an
accelerator run reaches backend execution and then fails, `stats()` retains
that failed run's measured counters and graph identity while its output hash
remains zero; it does not substitute an all-zero success-shaped snapshot.

Direct buffer execution returns a value-like `Run` receipt backed by fixed
inline storage. Its private state is constructed in place, so the public
receipt does not allocate a pimpl and a warm `Program::run(Buffer, Buffer)`
retains its zero-allocation contract. Copying a receipt copies its scalar
observation state and shared resource owners in place; it does not allocate a
second execution-state owner. A later read updates only that receipt's copied
observation state, preserving value semantics. The complete state and all
lifetime operations have one compiled Node owner rather than a public-header
layout definition.

`Stats` is the single public execution-evidence snapshot. Its names include
their unit or resource owner: `pipeline_compiles`, `buffer_allocations`,
`download_events`, `dispatches`, `command_submits`, `uploaded_bytes`,
`command_capacity`, `command_inflight_peak`,
`command_capacity_rejections`,
`downloaded_bytes`, `pipeline_cache_hits`, `pipeline_cache_evictions`,
`buffer_reuses`, `descriptor_pool_creations`, `descriptor_set_allocations`,
`descriptor_reuses`, `original_dispatches`, `final_dispatches`, `fusions`,
`fusion_rejections`, `internal_roundtrip_bytes`, `external_roundtrip_bytes`,
`reset_bytes`, `reset_commands`, `graph_read_bytes`, `kernel_ns`,
`kernel_samples`, `shader_compile_ns`,
`spirv_compile_ns`,
`pipeline_create_ns`, `descriptor_setup_ns`, `submit_wait_ns`, and
`readback_ns`. `dispatches` counts physical backend algorithmic dispatches,
including Metal/Vulkan device-side gather/scatter required to lower a strided
View for a dense-only primitive;
`command_submits` counts physical accelerator queue submissions. Multi-pass
execution can therefore report several dispatches in one submit. A prepared
multi-step graph reports its physical execution submissions separately from a
later explicit readback copy.
`reset_bytes` counts payload initialized by exact first-write frontiers;
`reset_commands` counts the physical reset operations consumed by that run.
They are execution evidence, unlike the canonical compile-time
`MemoryPlan::{reset_bytes,reset_count}`.

Vulkan's three command-envelope fields share one adapter-owned slot-ring
authority. Capacity is immutable for that Device, peak is the largest number
of simultaneously claimed slots in the snapshot, and rejection count is the
saturating number of precise bounded admissions. `Job::profile()` exposes
`command_pressure()` as the raw peak/capacity rational; it does not infer queue
pressure from wall time or store a second classification.

Prepared execution captures these fields at the command boundary rather than
subtracting two adapter-global snapshots. One Job reports its own physical
submit and dispatch count, submit interval, timestamp sample, capacity, and
observed occupancy. Independent Job evidence combines counts by unsigned
saturating addition and combines capacity/occupancy by maximum. Thus telemetry
collection cannot insert a fence wait or serialize the command envelope it is
measuring.

Accelerator values are projected once from the run's canonical backend
evidence. `fusions` is the checked difference between original and fused
operation counts; no second fusion counter or stored ratio exists. Internal
and external round-trip bytes remain separate, and callers derive a total only
when they need one. `download_events` is owned by successful public Run reads;
`downloaded_bytes` includes the backend execution evidence plus bytes delivered
by those reads. `WriteStats::uploads` remains the exact owner of explicit Job
write events and is not inferred from a nonzero byte total.

`readback_ns` is cumulative monotonic wall time for both execution-internal
status readback and public result-materialization reads. A public read measures
its actual elapsed interval and adds it to the owning `RunState`; it does not
reuse the kernel-completion snapshot, so a later `stats()` call cannot lose
materialization time that occurred after completion. The same `RunState`
accumulates actual post-kernel allocation, reuse, and submission deltas; all
three remain visible through the later `stats()` call.

All cumulative `Stats`, `WriteStats`, Program-cache, and `MemoryStats` source
counters use unsigned saturation at `UINT64_MAX`; none wraps modulo `2^64`.
`UINT64_MAX` is absorbing. In particular, releasing a memory gauge already at
that value keeps it saturated because the unrepresented excess is unknowable.
Snapshot subtraction with either endpoint saturated is likewise saturated.
The public telemetry values can therefore treat `UINT64_MAX` consistently
as non-exact source evidence.

CPU reports its applicable dispatch, worker, tile, SIMD, graph, output-read,
and host-allocation values. Accelerator-only fields are zero on CPU. Such zero
means unavailable or not applicable, not a measured zero-duration accelerator
event. Likewise, `kernel_samples == 0` makes `kernel_ns` unavailable; backend
timings are local diagnostic evidence and are never a portable speedup claim.
`kernel_timing_available()` is the stateless check for that sample condition;
it stores no second availability flag.

### Job profile

`Job::profile()` captures one owning `telemetry::Profile` from the Job that
owns the execution. The profile exposes its exact snapshots only through the
read-only `device()`, `execution()`, and `memory()` accessors; derived accessors
read those snapshots each time and store no mirrored counter or classification.
Device identity is the Program's selected device, execution is the Job's latest
successful or failed run evidence, and memory always has `MemoryScope::Job`.
Before the first run, execution contains the selected backend and zero
counters. An invalid or moved-from Job fails with `compute_profile_invalid`.
A queued, running, or input-writing Job fails with `compute_profile_busy`;
capture never substitutes backend-only zero counters for an in-flight run.

Capture creates the immutable owning `DeviceInfo` string snapshot, then copies
the fixed-size `Stats` and `MemoryStats` values while holding the Job state
gate. Prepared accelerator memory uses one serialized meter snapshot rather
than five independently sampled atomics, so all fields belong to one epoch and
`current <= peak` remains invariant under concurrent completions.
After capture, every `Rate` or `Share` accessor is `O(1)`, `memory_usage()`
selects one of seven fixed categories, and `largest_time()` performs exactly
seven candidate comparisons. No derived accessor allocates or mutates the raw
snapshots.
`findings()` performs one fixed-capacity projection through the compiled
Session telemetry owner and returns at most five entries in
`Allocation`, `Copy`, `Scan`, `Queue`, `CriticalPath` order. Its raw formulas
are

```text
copy    = sat(uploaded_bytes + downloaded_bytes)
scan    = graph_read_bytes
queue   = command_inflight_peak / command_capacity
prepare = sat(shader_compile_ns + spirv_compile_ns
              + pipeline_create_ns + descriptor_setup_ns)
work    = submit_wait_ns when nonzero, otherwise kernel_ns
finish  = readback_ns
```

The queue finding exists only at `capacity != 0 && peak >= capacity`; no
percentage threshold or backend guess exists. A Profile without a backend
timing source reports unavailable critical-path evidence. The seven-stage
`largest_time()` remains the precise raw timing inspection path; findings add
typed remediation without storing another counter or timing authority.
When Work is the Compute maximum, at least one command and one kernel timestamp
sample were observed, and `submit_wait_ns > kernel_ns`, the exact raw-counter
predicate reports `SubmitOverhead -> BatchJobs`; it does not mislabel
submission cost as graph work. Equality, missing submission/sample evidence,
and kernel-dominated work retain `ReduceGraphBound`. Raw kernel and submit/wait
durations are projected into a Session Event only at Detail. This is a
recommendation to amortize the observed per-submit interval across independent
resident Jobs, not a promise that batching removes queue contention or
dependency work.
The support `compute/telemetry.hpp` boundary forward-declares the finding
result; the sole direct Compute entry completes it. Internal Job and Profile
translation units therefore do not inherit the finding template body merely
to state `Profile::findings()`.
An unknown forged `MemoryCategory` fails with
`compute_profile_memory_category_invalid`; it is not reinterpreted as zero
budget or another category.

`Rate` retains the rational pair `(numerator, denominator)`. A zero denominator
makes `available()` false and `value()` returns `nullopt`; it is not converted
to zero, infinity, or a backend-dependent default. `value()` is only a
`long double` presentation conversion. The integer pair remains the traceable
authority. A `UINT64_MAX` component is conservatively non-exact because the
generic value cannot distinguish an exact limit from an absorbing cumulative
counter after prior overflow; `saturated()` exposes that state and `value()` is
unavailable. Thus every available Rate is derived from exact, unsaturated input
components.

`Share` retains `(selected, other)` rather than storing `selected + other`, so
two individually exact 64-bit terms can have a mathematical denominator above
`UINT64_MAX` without an integer overflow. It is unavailable when both terms
are zero or either term is saturated. The raw terms remain present in both
cases.

The derived formulas are fixed:

```text
kernel_time           = kernel_ns / kernel_samples
dispatches_per_submit = dispatches / command_submits
pipeline_cache        = pipeline_cache_hits /
                        (pipeline_cache_hits + pipeline_compiles)
buffer_reuse          = buffer_reuses /
                        (buffer_reuses + buffer_allocations)
descriptor_reuse      = descriptor_reuses /
                        (descriptor_reuses + descriptor_set_allocations)
dispatch_reduction    = max(original_dispatches - final_dispatches, 0) /
                        original_dispatches
internal_traffic      = internal_roundtrip_bytes /
                        (internal_roundtrip_bytes + external_roundtrip_bytes)
memory_usage(C)       = memory.C.current / memory.C.budget
```

`dispatches_per_submit()` is the literal quotient of the two named counters.
Explicit readback can add a submission without adding a compute dispatch, so
the quotient is not labeled as command-encoding efficiency.
`dispatch_reduction()` compares like-unit algorithmic dispatch counts and does
not use `fusions`, whose owner is the difference between original and fused
operation counts rather than dispatch counts.

`largest_time()` compares the seven cumulative raw timing counters—shader
compile, SPIR-V compile, pipeline creation, descriptor setup, submit wait,
kernel, and readback—and returns every stage tied for the maximum. Kernel time
is admitted by `kernel_samples != 0`; the other timing counters are admitted by
a nonzero measured duration because they have no independent public sample
counter. There is no threshold or tie breaker. The function deliberately does
not add the timing fields: backend clocks can cover different scopes, so a
synthetic total could double-count work. The result identifies the largest
observed counter in this snapshot, not a causal diagnosis or a portable
cross-device speedup.
If the maximum duration is `UINT64_MAX`, `Focus::saturated()` exposes that
the exact duration and ordering among other capped stages are unknowable; every
stage tied at the absorbing value remains included.

```cpp fragment
auto profile = job.profile();
if (profile) {
  const auto &device = profile->device();
  const auto &execution = profile->execution();
  auto kernel_ns_per_sample = profile->kernel_time();
  auto device_budget = profile->memory_usage(compute::MemoryCategory::Device);
  auto largest_observed_time = profile->largest_time();
  auto actions = profile->findings();
}
```

The executable primitive/domain/backend matrix has one authority: the
Standalone Compute contracts under `node/tests/contract/compute/`.
`backend.cpp`, `flow/numeric/modes.cpp`, `collective/modes.cpp`,
`boundary/modes.cpp`, `expression/`, `fixed/multiply/declared.cpp`,
`fixed/predicate/wide.cpp`, `bounded.cpp`, `graph/services.cpp`, and
`flow/shape.cpp` own cross-backend parity, exact and bounded transitions,
fixed-policy goldens, Graph identity, and static Matrix shape. These executable
contracts are the authority; there is no parallel inventory that restates
their case membership or expected row counts.

The registered `memory.cpp`, `backend.cpp`, `bounded.cpp`, and
`collective/modes.cpp` sources are ordered case runners. Their semantic
translation units live below matching folders and retain one `local`/`model`
fixture authority per case. The split preserves each case's domain sequence,
CPU-oracle-first backend comparisons, fingerprints, failure precedence, return
codes, and registry row while letting a changed memory, movement, bounded,
empty, cancellation, or overflow oracle compile independently.

The registered `map.cpp` and `flow/shape.cpp` sources are likewise thin case
exports. Map's canonicalization, liveness, envelope, execution, and carrier
oracles live below `map/`; `map/local/model.cpp` is the sole carrier/backend
fixture authority. Static Matrix compile-time typestate lives in
`flow/shape/surface.cpp`, exact runtime order lives in `flow/shape/run.cpp`,
and `flow/shape/local/model.cpp` solely owns Matrix/Solve graph construction
and backend comparison. Their registry rows, assertion order, diagnostics,
and result codes do not move with the physical split.

The registered `flow/numeric.cpp` and `flow/numeric/modes.cpp` sources are
thin case exports. `flow/numeric/local/model.cpp` is their sole backend order,
failure-precedence, diagnostic, and result-code owner; `core.cpp`,
`format.cpp`, and `modes/{algebra,golden,policy,transform}.cpp` own the
semantic oracles. Both cases intentionally share that one compiled cohort:
the model references both ordered runners, so the harness links all numeric
leaves for either case instead of creating a second execution authority. The
split preserves both registry rows and bit/hash parity while a semantic edit
recompiles one leaf.

`flow/primitives.cpp` is a CPU-only semantic oracle containing only composition
laws that need a whole Flow: pipe, ordered outputs, combine,
branch/record/repeat, indices, group, join, and relational records. It does not
open an accelerator or repeat physical backend parity. `bounded.cpp` is the
only filter/compact/expand law owner,
including worker identity, inactive tails, reduction rewrites, invalid counts,
and warm reuse. `expression/part/integral.hpp` owns unsigned 32/64-bit high-bit
comparison, min/max, clamp, and mask behavior. `flow/shape.cpp` owns exact
compile-time Flow and Program result types. Matrix/Transform numeric execution,
collective behavior, and 255-element executor tails remain in their stronger
domain/backend owners and are not mirrored in the Flow UX oracle.

The registered `flow.cpp` case is a thin ordered runner. Its surface, shape,
device, expression, record, composition, basic, parity, and selected-backend
contracts are one-word translation units below `flow/contract/`. Shared typed
fixtures and result-hash oracles have one `local.hpp`/`model.hpp` authority.
The split reduces edit invalidation while preserving the same selected backend
sequence, return codes, graph/output comparisons, and registry owner.

NodeHost has no parallel math matrix. `Session::compute(Job<Signature>&)`
extracts the same opaque `JobState` for every signature, while
`Session::compute(Pipeline&)` extracts the Pipeline's one type-erased operation
owner; both reuse the same Request/Submission/Poll/Completion lifecycle.
`runtime.compute-accel` therefore uses one
representative resident Job per accelerator backend to prove admission,
completion, warm-zero evidence, telemetry, concurrency, retained lifetime, and
balanced external-completion accounting. A warm accelerator may complete
before its coordinator reaches the suspension point, so zero park/wake pairs
are valid; this contract rejects only an unmatched pair. The deterministic
positive park/wake transition is owned by the scheduler host-I/O contract,
which controls completion timing instead of inferring it from device latency.
`runtime.compute-host`, terminal/cancel contracts, and
`runtime.compute-lifecycle` own the corresponding CPU and failure envelopes.
The host concurrency case primes the exact four-leaf lane segment before
counting the overlapping Compute submission, so its zero-allocation result is
a warm-path property and cannot vary with which host schedules the first cold
leaf batch.
The host contract also runs one zero-active bounded Scan through blocking and
submitted CPU transports and requires identical graph, read-byte, dispatch,
worker, tile, and SIMD evidence. That row proves the shared step owner and the
zero-work accounting boundary; it is not a second arithmetic matrix.
Expression, fixed-format, movement, Matrix, and graph-service rows are not
repeated through Runtime because Runtime cannot observe those semantic
dimensions. The focused `runtime.compute-pipeline` and
`runtime.compute-pipeline-accel` rows own only Pipeline admission, poll/await,
cancel/wait/close, retained lifetime, claim release, poison publication, and
the shared completion evidence; the standalone Pipeline contract remains the
numeric, hazard, fingerprint, memory, and backend execution authority.

## 4. Advanced Execution Resources

Flow is the only public graph-construction language. Advanced callers may open
a `Device` to inspect memory or own explicit buffers, then bind the same device
to Flow with `on(device)`. This does not expose symbolic node construction or a
second compile path.

```cpp fragment
auto device = rund::compute::open(rund::compute::Target::vulkan());
if (!device) { return; }

auto program = rund::compute::on(*device)
    .map<std::uint32_t>("adjust", count,
                        [](auto value) { return value + 1; })
    .scan(rund::compute::Scan::InclusiveSum)
    .compile();
```

`on(device)` freezes the existing device identity in the Flow recipe. It is
useful when the caller also needs `Device::buffer`, `Device::upload`, or
`Device::memory`; it does not reopen the backend at `compile()`. The ordinary
path remains `on(Target)`. A moved-from Device is invalid:
`Device::backend()` and a Flow built with `on(device)` return
`compute_device_invalid`, and compilation performs zero backend-open attempts.
There is no CPU placeholder execution or retry. `Device::valid()` and explicit
truth conversion expose owner validity. `Backend::Unavailable` is reserved for
unavailable observation snapshots; it is not a selectable execution backend.

Multiple external inputs use `input<T>(count)` followed by typed
`zip_input<U>(count)`. External input identity is the ordered Flow signature:
type, count, Fixed numeric policy, and declaration position. Inputs do not
accept a diagnostic name. `branch` receives immutable `StageRef` values for
those inputs. Their counts are independent; only a direct elementwise `map`
over the complete input set requires equal counts. Gather, scatter, partition,
segmented operations, bounded join, records, and ordered outputs stay inside
Flow. `outputs(a, b, ...)` fixes declaration order and supports at most 16
logical leaves. `read<I>()` downloads only leaf `I`; `read_all()` returns the
typed nested tuple without rerunning the graph.

Zero-element Map and Scan recipes are valid no-work executions. They keep the
explicit backend and canonical graph identity while submitting zero CPU tiles
or GPU dispatches. Matrix, factor, solve, and spectrum require nonzero declared
dimensions. This is a semantic shape contract, never an element-count-based
backend or serial/parallel policy.

## Selection and Errors

Every `on` and `open` call names one target. `Target::cpu(width)`,
`Target::metal()`, and `Target::vulkan()` are the only selection factories;
zero CPU width means the host or Session width. The same `Target` value is
accepted by both verbs, and an already-open `Device` is the other execution
path. `Backend` is observation-only in Device info, statistics, and telemetry
and cannot be passed to either selector. The selected backend is frozen in
Flow, Device, and Program state, and execution never retries elsewhere.

`Device::info()` returns `Result<DeviceInfo>` and creates an owning identity
snapshot. `DeviceInfo` has `backend`, `name`, `driver`, `driver_details`,
`storage_alignment`, and `storage_bytes`; its strings remain valid after the
Device or backend state is destroyed. The numeric fields are the immutable
offset alignment and largest single storage binding used by backend admission.
Accelerator snapshots copy `AccelDevice::backend_info.device_name`,
`driver_name`, and `driver_info` into the three identity fields without
substituting one field for an absent field. CPU snapshots are derived at the
call boundary
from the existing `CpuCaps` and worker selection: `name` is
`cpu/<scalar|sse2|avx2|avx512|neon>`, `driver` is
`rund/built-in-pool` or `provided-worker-backend`, and `driver_details` is
`workers=N;lane:bytes=N;fixed:lane32:lanes=N;fixed:lane64:lanes=N`. Colons carry
the property hierarchy without inventing width-specific Fixed type names. The
CPU Device stores no permanent identity-string mirror.

The four CPU detail labels occupy 60 bytes and four decimal `uint32_t` values
occupy at most 40 bytes, so the complete detail payload is bounded by 100
bytes. Construction reserves that exact upper bound before appending any row.

For total payload length
`S = size(name) + size(driver) + size(driver_details)`, an identity call takes
`Theta(S)` time and `Theta(S)` logical owning payload (with constant-size string
and allocator metadata). That copy is required by
the lifetime contract; the implementation introduces no intermediate payload
copy and moves the three completed strings into the `Result`. Allocation
failure returns `Capacity/compute_device_info_capacity`; a null, moved-from, or
internally inconsistent Device returns
`Invalid/compute_device_info_invalid`.

The Apple Silicon release contract requires CPU, Metal, and Vulkan/MoltenVK.
Its tests treat an unavailable selected accelerator as failure; they never skip the
row or reinterpret `Code::Unavailable` as a passing result. `Unavailable`
remains the public runtime diagnostic for an explicitly selected backend that
cannot be opened outside that supported release environment.

On Apple Silicon, `Backend::Vulkan` still names the Vulkan API and lowering
contract, while `DeviceInfo::driver == "MoltenVK"` identifies the physical
Vulkan-to-Metal translation path. Its output parity, SPIR-V, descriptor,
command, barrier, and same-path regression evidence remain Vulkan evidence;
its elapsed time is not native Vulkan-driver throughput. The direct
`Backend::Metal` path with driver `Metal` is native Metal throughput evidence
for that exact Apple device and environment. A timing delta between those rows
compares complete Metal and MoltenVK stacks and must not be presented as Metal
versus native Vulkan. The installed measurement interpretation contract lives
in [Performance Method](./performance/method.md#compute-backend-method).

Fallible operations return `Result<T>` or `Status`. `Reason` is the sole
failure authority. Applications may branch on its derived small `Code`
category, while `error()` derives the precise stable diagnostic text used by
evidence tooling. The C++ value type selects the exact integer
or fixed domain; Device selection carries no independent scalar mode that can
conflict with it. Intermediate graph operations do not return `Result`: the
owning Flow records the first logical builder failure and later dependent
expressions cannot replace that reason. The terminal publishes that preserved
failure.

`Code` has one current vocabulary:

| Code | Meaning |
| --- | --- |
| `Ok` | The operation completed successfully. This is the only code for which `ok()` is true and `error()` is empty. |
| `Invalid` | The object, option, state transition, or request itself is invalid. |
| `Unsupported` | The requested operation or policy is outside the implementation contract. |
| `Unavailable` | A selected backend, adapter, device, or host resource cannot be opened in the current environment. |
| `Capacity` | A bound, allocation, identifier space, or checked representation cannot be admitted before execution. |
| `Compile` | IR admission, lowering, shader, pipeline, or Program compilation failed. |
| `Binding` | Input count, type, shape, format, resource, or Program binding does not match the compiled contract. |
| `Transfer` | Upload, download, readback, or host/device transfer failed. |
| `Execution` | Submission, execution, cancellation, a runtime numeric/active-set bound, or resident Job lifecycle failed. |

`Status` stores one two-byte `Reason`; it stores neither a `Code` mirror nor a
borrowed diagnostic view. `reason()` returns that authority. `code()`, `ok()`,
truth conversion, `error()`, and `exit_code()` are pure projections;
`exit_code()` is zero only for `Reason::Ok` and one for every failure. Failure
construction is `Status::fail(Reason)` or `Result<T>::fail(Reason, Location)`.
The location is optional cold-path evidence owned by `Result<T>` and does not
enlarge the hot execution `Status`. Its three canonical U32 coordinates are
the logical Pipeline step, physical recurrence iteration, and Program graph
node; an unavailable coordinate is `Location::none`. `transform` and
`and_then` preserve the whole failure, including location, without invoking
the callback.

There is no `Code + string_view` overload, so a temporary diagnostic cannot
dangle and callers cannot combine an unrelated category and text. A forged
underlying enum value, or `Reason::Ok` passed to a failure factory, normalizes
to `Reason::ReasonInvalid`. Applications may persist the derived string when
evidence storage requires it, but that string is never the in-memory failure
owner.

Foreign Kernel and accelerator diagnostics cross one typed operation boundary.
Let `S` be the non-success diagnostics in `reason.def`, `L` its exact lookup,
and `d(b)` the public Reason fixed by operation boundary `b`. The projection is

```text
P(b, text) = L(text)  when text is in S and L(text) is not ReasonInvalid
             d(b)     otherwise
```

Open, allocation, transfer, compilation, execution, and tile boundaries use
`AdapterUnavailable`/`BackendUnsupported`, `BufferCapacity`,
`TransferInvalid`, `LoweringInvalid`, `BackendFailed`, and
`TileBackendFailed`, respectively. Exact canonical reasons such as numeric
overflow, cancellation, and `DeviceBusy` survive unchanged. Raw
`accel_*`, `cpu_*`, and host-backend strings never become public
`ReasonInvalid`, and no substring classifier or mirrored alias table exists.
The projection performs the same `O(n)` fixed-schema hash lookup as the public
text projection and allocates no storage. Reason lookup, expression interning,
shader-source prefiltering, and SPIR-V identity share one internal FNV byte
transition implementation while retaining their own frozen seeds and framing;
the shared mechanism is not a shared semantic identity domain.

Factor, Solve, and Spectrum per-batch status values are typed execution data,
not foreign backend diagnostic strings. Their central primitive-status mapper
preserves the public U32 status Buffer and projects the ten exact canonical
execution Reasons owned by the
[Pipeline status contract](../../node/docs/contracts/compute/pipeline.md#state-and-publication).
An unknown status ordinal is `ReasonInvalid`; replacing a known or unknown
semantic status with generic `BackendFailed` is forbidden.

The checked-in `reason.def` schema is the sole list of precise failures. The
public enum, stable diagnostic projection, and canonical half of the foreign
operation-boundary projection are generated from that list. Duplicate numeric
values or diagnostic hashes
fail compilation, and a compile-time category check proves that each encoded
ordinal range is contiguous. The large text projection is compiled once in
the Compute library rather than instantiated through every SDK translation
unit.

`graph::Info` reports Fixed resource policy through the public `Rounding`,
`Overflow`, and `Approximation` enum types. Callers compare those values
directly; the underlying byte representation is not a public policy language.

The public scalar domain is closed to `int32_t`, `uint32_t`, `int64_t`,
`uint64_t`, and `Fixed<I, F>` where `I + F` is 32 or 64. Flow input and Device
buffer/upload overloads participate only for those forms.
Fourier Transform, Factor, Solve, and Spectrum Flow operations
participate only for `Fixed<I, F>`; an integer Matrix remains valid,
but fixed-only transitions are absent at compile time. Floating-point input or
an integer decomposition is therefore a static type error, not a graph that is
accepted and rejected later by `compile()`.
Fourier element count must be a nonzero power of two. A zero or non-power-of-two
count fails at the Flow terminal with
`compute_transform_count_not_power_of_two`; it is not padded, truncated, or
lowered through a backend-specific fallback.

`Result<T>::transform` maps a success value, `and_then` chains another Compute
Result, and `value_or` supplies an explicit alternative value. A failure
bypasses the callable and preserves its exact `Reason`. These operations do not
convert expected failures into exceptions.

`compute::Result<T>` and `task::Result<T>` are concrete domain-facing families
over one generic value-or-failure owner. Neither keeps a second status mirror.
Success constructs the value
directly in its active alternative, avoiding an intermediate payload move. If
a throwing user move invalidates that alternative, every checked Compute
observer remains total and reports `Code::Invalid`, `Reason::ValueInvalid`,
`compute_value_invalid`, and exit code 1. It never throws from `code()`,
`reason()`, or `error()`, and `operator->` returns null while no value is active.

`Fixed<I, F>` stores a two's-complement raw integer scaled by `2^-F`.
`Fixed<I, F>` is the only public fixed value spelling. Storage width does not
imply an I/F split: every fixed Flow value and primitive descriptor carries its
complete format and numeric policy explicitly, and unknown formats or modes
fail closed instead of inventing an I/F format.
`from_ratio` constructs a stored value with widened integer arithmetic and an
explicit `Rounding` mode:
`TowardZero`, `Down`, `Up`, or `NearestEven`. `ratio<N, D, Mode>()` provides the
same law at compile time. Conversion saturates at `min()`/`max()`; a zero
denominator fails with `compute_fixed_ratio_zero_denominator`. There is no
implicit or floating-point constructor, and the `from_raw` and `from_ratio`
factories reject floating-point arguments rather than converting them to an
integer first. Expression arithmetic retains derived I/F precision;
`quantize<T>()` is the only conversion to a stored fixed format.
A typed Fixed literal always keeps `T`'s stored I/F split. When a literal is
paired with one Fixed expression by arithmetic, comparison, min/max, clamp, or
one-literal select, it inherits that expression's rounding and overflow policy
while remaining `Exact`; it is never reinterpreted at the widened expression's
binary point. A select whose two result branches are literals uses the declared
stored `T` policy and does not inherit policy from its predicate. Fixed helper
constants and their internal storage boundaries follow the same rule, so a
caller-authored non-default policy is not replaced inside a composite helper.
`fixed_zero(value)` is raw zero, `fixed_one(value)` is the nearest-even stored
encoding of numeric `1`, and `fixed_max(value)` is the lane's greatest signed
stored value. They are distinct contracts: for `Fixed<I, F>` with `I > 1`,
`fixed_one` has raw bits `1 << F` while `fixed_max` remains
`2^(I+F-1)-1`; for `I == 1`, numeric one saturates to that maximum because
positive `1` is not representable. `fixed(FixedOp::Half/Third/Quarter, value)`
likewise encodes `1/2`, `1/3`, or `1/4` at the declared `F` with nearest-even
constant rounding, inherits the anchor's rounding and overflow policy, and
remains `Exact`. Normalized helpers (`saturate`, `step`, interpolation and
smoothing, hard activations, and normalized windows) use `fixed_one`, never
`fixed_max` as an alias for one.

Expression helper modes have one selector owner per family. The callable
surface uses values such as `FixedOp::Half`, `MetricOp::Squared`,
`GeometryOp::Distance`, and `ActivationOp::Relu`; their overload-dispatch tag
types are nested under that selector instead of becoming independent
`rund::compute` names. Geometry helpers consume `GeometryOp` as their sole
mode authority.

Signal-window coefficients are canonical Q1.31 integers but are rescaled to
the caller's actual `F` with nearest-even rounding before entering the graph;
64-bit storage is not interpreted as an implicit Q1.63 coefficient. Unit hash
projection masks raw hash bits with `(1 << F) - 1`, so its codomain is
`[0, 1)` at the declared binary point rather than `[0, fixed_max]`. CPU,
Metal and Vulkan consume the same literal bits and fixed-format metadata.
Every derived binary or ternary I/F combination is checked before graph
admission and fails with `compute_fixed_precision_capacity` when its exact
carrier would exceed 128 bits.
A direct identity Map may return an input that is already stored. Canonical IR
serialization still places a same-format, same-policy identity `Quantize`
immediately before its `Write`, so every stored Fixed write has one explicit
lowering boundary. That normalization does not choose a format or lose
precision. Every widened arithmetic result must cross the caller-written
`quantize<T, Rounding, Overflow, Approximation>()` boundary; Flow and Map never
select a target format or narrow such a result automatically.
Storage-domain primitives such as nonlinear and saturating operations accept
only already-stored operands; widened operands fail until the caller
quantizes them. Their declared codomain is a stored width, not an implicit
narrowing of a wider expression. Nonlinear, division, and `atan2` results carry
the `Deterministic` approximation policy, while exact arithmetic carries
`Exact`; callers select the final output policy with the public
`quantize<T, Rounding, Overflow, Approximation>()` boundary.
Quantization first aligns fractional precision with the selected rounding law,
then applies the target overflow law to the rounded integer. `Wrap` therefore
means modulo `2^W` after rounding, where `W` is the target storage width.
The public stored multiply operations are distinct and all scale by the
operand format's actual declared `F`: `mul_fixed(a, b)` interprets both raw
lanes as signed, `mul_fixed_scaled(a, b)` interprets only `b` as unsigned, and
`mul_unsigned_fixed(a, b)` interprets both as unsigned and applies unsigned
storage bounds. Each operation rounds the widened raw product divided by
`2^F`, then applies its signed or unsigned `Saturate`/`Wrap` law. There is no
width-derived binary point. `mul_add_fixed(a, b, c)` keeps the product at
`2F`, aligns `c` to that scale, adds there, and crosses the stored boundary
only through the terminal `quantize<T>()`; cancellation is therefore evaluated
before any product narrowing.

## CPU and Accelerator Execution

The public `Program` type hides two internal engines without merging their
responsibilities. CPU programs use the canonical Kernel Compute plan, Kernel
physical tile planner, existing worker backend, and tile-local SIMD execution.
Accelerator programs use the same canonical identity and fixed-point meaning,
then hand one or a planner-bounded small number of dispatches to the selected
Metal or Vulkan context.

CPU Map tiles are contiguous buffer slices. A standalone CPU Device owns one
worker pool, a Program owns one prepared tile plan, and a warm run neither
creates a pool nor recompiles or reallocates per worker. Standalone execution
never detects an ambient Session. Only explicit `Session::compute(job)` selects
the Session-owned worker backend and its existing scheduler lanes.
Incompatible width or nested synchronous execution fails instead of opening
another worker resource.
CPU graph execution has one step owner. It resolves the immutable value route,
validates a bounded count, binds a Map, selects the active Scan/Reduce phase,
merges collective tiles, advances the canonical step index, and accumulates
`Stats`. Standalone execution and Node-hosted execution differ only at the
worker transport: Standalone blocks in the prepared tile runner, while
Node-hosted execution submits the same prepared pass and later finishes it.
Neither transport owns another Map/Scan/Reduce switch or another statistics
formula.

Every CPU execution path resolves an authored Buffer View through one private,
overflow-free footprint owner. The returned first-byte pointer, base, byte
stride, payload bytes, count, and width are consumed directly by Map,
collective, primitive, reset, staging, and Pipeline execution. Dense-only
operations add only that policy; strided reset and publication perform no
second per-element bounds pass. The exact proof is owned by
[Compute Memory Ownership](../../node/docs/contracts/compute/memory.md).
All exact host byte products used around that boundary share one private
`size_t` checked law. Exact U64 graph placement and saturating telemetry remain
separate arithmetic authorities, so centralization cannot change rejection,
wrap, or saturation semantics.

For graph steps `s = 0..S-1`, both transports publish results only after step
`s` completes before starting `s+1`. A Scan runs local tiles, writes every tile
prefix by the recurrence `p[0] = 0`,
`p[t] = sum(j=0..t-1, total[j])` in increasing tile order, and then runs prefix
correction. Reduce merges tile totals in the same increasing tile order.
Worker completion order therefore cannot change the arithmetic order. Because
the prefix recurrence overwrites every live `p[t]`, initialization traffic is
zero. Reduce has no prefix phase and therefore retains only
its 128-bit tile-total array, not Scan's second prefix array. At prepared tile
capacity `T`, a Reduce Job uses one fewer allocation and `16*T` fewer bytes
than a Scan Job.
The shared step path allocates no route, binding, or phase storage during a
warm run.
The opaque implementation is a tagged split: CPU devices store frozen
`CpuCaps` plus the Kernel worker backend, CPU buffers store 64-byte-aligned
host memory, and accelerator devices/buffers alone store Accel context and
resident handles. `CpuProgram` owns one immutable `ComputeMap` descriptor, one
compact prepared SIMD instruction/format plan, and one tile plan. Each resident
Job prepares its own tile-run state, SIMD counters, and
fixed-capacity map and primitive scratch before the warm loop. Independent
Jobs from one CPU Program therefore do not share mutable execution scratch;
the same Job still rejects a second concurrent run with `compute_job_busy`.
Accelerator resident Jobs likewise allocate their intermediate graph and
collective buffers during `resident()`. Those buffers and their fixed-view
binding arrays are not shared between Jobs. The mutex-serialized convenience
path retains one ordinary read-only Job; resident runs consume their own
Job-owned bindings and prepared state without locking that cache.
`Stats` reports worker width and participation, tile count and size, SIMD
vector/tail chunk counts, and warm allocation evidence without copying the
Kernel's per-worker distribution or last-tile internals into the public
result.
Collectives preserve cross-tile semantics through their dedicated planners:
scan uses local scan, tile-sum scan, and prefix correction, while reduce uses
tile-local reduction and deterministic tile-index merge. GPU submission count
does not scale with host worker or tile count. Metal encodes all prepared graph
steps into one command buffer per run. Vulkan records the complete ordered step
stream once into a prepared secondary command and executes it from one small
primary wrapper per run. Both preserve compute-to-compute step order and submit
once. Each dispatch still covers its full logical
range; Node-native execution uses the same batch through the device completion
service.
Metal and Vulkan Scan use the same fixed 128-lane lowering. Each lane consumes
one contiguous increasing-index slice of the logical block, the lane totals
use one fixed prefix tree, and the lane then materializes its slice from the
reconstructed prefix. The input count does not select a different workgroup
width or algorithm. This bounds threadgroup scan storage to 1 KiB for both the
double-buffered U32 tree and the U64 tree, while preserving exact stored prefix
bits and per-element overflow checks.
The canonical graph chooses a power-of-two logical collective block no larger
than 256 elements (or 1 for a single element). Counts need not be powers of
two; the last block is bounds-masked. A backend may subdivide that logical work
into a fixed physical tree, but physical grouping is lowering evidence, not
graph identity. Nonuniform tails therefore cannot
change the result or fingerprint.
Boundary contracts execute empty, min/max, exact overflow, and 255-element
tail rows for every primitive/domain combination where those values are
semantically valid. `compute.collective-modes` is the only owner of collective
tail, empty, extrema, and overflow rows. `compute.boundary-modes` is the only
owner of the remaining physical executor tails: movement, matrix, bounded
composition, compact/histogram, transform, factor, solve, and spectrum. Its
93 successful Program rows are
`6*3 + 2*6*3 + 4*3 + 1*3 + 4*2*3`; six additional rows prove that the two
fixed storage widths reject a 255-element Fourier transform on all three
backends. Another 18 Scatter Programs (`6 domains * 3 backends`) each execute
both mixed-failure orderings, for 36 failure runs proving that duplicate and
out-of-range evidence always selects the earliest failing input index. Every
successful shape runs once on CPU as the oracle and once on each selected
accelerator backend. There is no product-Runtime semantic mirror.
Overflowing sum collectives fail with the same stable reason on every declared
backend; Runtime contracts only prove the integration boundary.

The same-manifest scheduler, Map scaling, accelerator completion, and
all-family primitive measurement method, together with revision-scoped dated
packets, is recorded in
[`architecture/verification.md`](../architecture/verification.md#installed-measurement-method).
A packet is closure evidence only when its recorded source manifest matches the
verified tree. Those local measurements apply to the named host and workload,
not a portable performance guarantee.
