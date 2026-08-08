# Node CPU SIMD Runner Contract

## Verification Authority

The public contract cases are:

- `accel.cpu-simd.vector`
- `accel.cpu-simd.dsl-basic`
- `accel.cpu-simd.dsl-geometry`
- `accel.cpu-simd.dsl-stats`
- `accel.cpu-simd.dsl-matrix`
- `accel.cpu-simd.ops`
- `accel.cpu-simd.backend`

The six direct SIMD rows close at `CPU_SIMD`. The backend facade row closes at
`CPU_ACCEL` because it also proves CPU pick, context, and prepared-kernel
behavior. Neither closure includes full catalog composition or native backend
sources and libraries.

Their C++ source authority is split into focused entrypoints and owners under
`/node/tests/contract/accel/cpu/`:

- `vector/contract.cpp`: vector-law contract entrypoint
- `vector.cpp`: vector-law owner router
- `vector/fixed/{lane32,lane64}.hpp` plus
  `vector/{tail,strided,plan}.hpp` and `vector/write/multiple.hpp`: fixed lane 32
  full-vector, fixed_lane64 full-vector, fixed_lane32 tail-chunk, strided and
  multi-write binding evidence, plus frozen executor-selector and scratch-capacity
  evidence
- `dsl/basic.cpp`: hash, noise, normalization, range, mask,
  tolerance, piecewise, and polynomial DSL helper contract entrypoint
- `dsl/geometry.cpp`: vector, squared metric, metric, projection,
  and cross/reject DSL helper contract entrypoint
- `dsl/stats.cpp`: statistics, moment, and correlation DSL helper
  contract entrypoint
- `dsl/matrix.cpp`: linear, matrix, affine, mix, and interpolation
  DSL helper contract entrypoint
- `ops.cpp`: fixed_lane32 scalar, bit, arithmetic, nonlinear, composite,
  fixed_lane64, and ordering operator contract entrypoint
- `backend.cpp`: CPU pick, prepared facade, and forged artifact
  rejection contract entrypoint
- `ops/32/{scalar,bit,arithmetic,nonlinear}.cpp`: focused fixed_lane32 operator
  coverage by arithmetic family
- `ops/32/composite.cpp`: fixed_lane32 composite DSL helper coverage
- `ops/64/contract.cpp`: fixed_lane64 operator coverage router
- `ops/64/`: fixed_lane64 work fixture, DSL expression, reference oracle, and run
  body
- `hash.cpp`: fixed_lane32/fixed_lane64 deterministic hash and unit-hash coverage
- `fixed.hpp`: shared fixed_lane32/fixed_lane64 scalar interpolation and normalization
  oracle
- `noise.cpp`: fixed_lane32/fixed_lane64 deterministic fade and one-dimensional
  `noise(...)` overload coverage
- `noise.hpp`: shared fixed_lane32/fixed_lane64 deterministic noise oracle
- `noise/grid.cpp`: fixed_lane32/fixed_lane64 deterministic two-dimensional `noise(...)`
  overload coverage
- `norm.cpp`: fixed_lane32/fixed_lane64 deterministic unlerp, remap, and smoothstep
  coverage
- `vec.hpp`: shared fixed_lane32/fixed_lane64 deterministic vector helper oracle
- `vec.cpp`: fixed_lane32/fixed_lane64 deterministic dot, length, and distance coverage
- `sq.cpp`: fixed_lane32/fixed_lane64 deterministic squared length and distance coverage
- `metric.cpp`: fixed_lane32/fixed_lane64 deterministic L1 and L-infinity metric coverage
- `range.cpp`: fixed_lane32/fixed_lane64 deterministic min, max, median, spread,
  sorted clamp-range, sorted band pass/stop, and ordered inclusive range
  predicate coverage
- `stats.cpp`: fixed_lane32/fixed_lane64 deterministic fixed half, fixed third, mean,
  centered, and absolute-centered helper coverage
- `moment.cpp`: fixed_lane32/fixed_lane64 deterministic squared-difference, variance,
  and RMS helper coverage
- `stat.hpp`: shared fixed_lane32/fixed_lane64 deterministic arithmetic/statistics
  helpers used by statistics, linear, matrix, and affine oracles
- `corr.cpp`: fixed_lane32/fixed_lane64 deterministic covariance and correlation helper
  coverage
- `linear.cpp`: fixed_lane32/fixed_lane64 deterministic `dot(...)` and `conv(...)`
  overload coverage
- `matrix.cpp`: fixed_lane32/fixed_lane64 deterministic small matrix helper coverage
- `affine.cpp`: fixed_lane32/fixed_lane64 deterministic small affine helper coverage
- `wide.hpp`: shared fixed_lane32 to fixed_lane64 test input widening helper
- `mix.cpp`: fixed_lane32/fixed_lane64 deterministic weighted mix helper coverage
- `poly.cpp`: fixed_lane32/fixed_lane64 deterministic polynomial helper coverage
- `interp.cpp`: fixed_lane32/fixed_lane64 deterministic interpolation helper coverage
- `mask.cpp`: fixed_lane32/fixed_lane64 deterministic zero/sign predicates, three-way
  predicate folds, and branchless keep/zero mask coverage
- `tol.cpp`: fixed_lane32/fixed_lane64 deterministic near, near-zero, deadzone, and snap
  coverage
- `piece.cpp`: fixed_lane32/fixed_lane64 deterministic clip, positive part, and negative
  part coverage
- `proj.hpp`: shared fixed_lane32/fixed_lane64 deterministic unit/projection oracle
- `proj.cpp`: fixed_lane32/fixed_lane64 deterministic unit and projection coverage
- `cross.hpp`: shared fixed_lane32/fixed_lane64 deterministic cross/reject oracle
- `cross.cpp`: fixed_lane32/fixed_lane64 deterministic cross and rejection coverage
- `pick.cpp`: CPU backend pick, generic backend execution, and generic artifact
  rejection
- `forged.cpp`: CPU `LoweringArtifact` mismatch rejection before output writes
- `order.cpp`: ordering/min/max/clamp operator coverage
- `prepared.cpp`: prepared tile phase facade through the CPU backend

## CPU SIMD Runner

`rund::node::accel::RunCpuSimd(...)` is the node-owned deterministic CPU
execution surface for kernel compute artifacts. It consumes a canonical kernel
`ComputeIR`, frozen kernel `CpuCaps`, a checked kernel `LoweringArtifact`, and
the existing staged `BindingSet` bridge used by the fixed map DSL.

This runner is direct SIMD code. Compiler auto-vectorization is not the
execution authority. The current executable lane law uses the repository-owned
128-bit math vector carriers:

- fixed_lane32: `math32::simd::I32x`, four 32-bit lanes
- fixed_lane64: `math64::simd::I64x`, two 64-bit lanes

The runner validates caps and the common CPU artifact admission before looking
at binding storage. A forged artifact key, plan text, metadata, or canonical IR
payload rejects before any output write.

The admitted execution shape is fixed_lane32/fixed_lane64 map IR with one write binding,
staged output storage, no resident output, no sequence tile remapping, and
the admitted fixed map operations:

- `param`, `read`, `constant`, `write`
- `add`, `sub`, `mul`, `neg`
- `abs`, `abs_magnitude`, `sign`
- `min`, `max`, `clamp`, `select`
- `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- predicate `not`, `and`, and `or`
- `bit_and`, `bit_or`, `bit_xor`, `bit_not`
- checked constant shifts
- `add_sat`, `add_sat_unsigned`, `sub_sat`
- `neg_positive_fixed`
- `mul_fixed`, `mul_fixed_scaled`, `mul_unsigned_fixed`, `mul_add_fixed`
- `div_fixed`, `recip`, `sqrt`, `rsqrt`
- composite DSL same-name overload families `dot`, `conv`, `mat(Axis, ...)`,
  `mat(MatOp, ...)`, `aff(Axis, ...)`, `mix`, `poly`, `bezier`, `len`,
  `len(MetricOp::Squared, ...)`, `dist`, `dist(MetricOp::Squared, ...)`,
  `Norm`, `mean`, `var`, `rms`, `cov`, `corr`, `noise`, and scalar helpers
  `saturate`, `step`, `lerp`, `lerp(LerpOp::Smooth, ...)`, fixed-arity
  `lerp(...)` overloads, `unlerp`, `remap`, `smoothstep`, `absdiff`, `min`,
  `max`, `median`, `spread`, `clamp_range`, `in_range`, `out_range`,
  `bandpass`, `bandstop`, `fixed(FixedOp, ...)`, `centered`, `is_zero`,
  `nonzero`, `is_neg`, `is_pos`, `is_nonneg`, `is_nonpos`, `all`, `any`,
  `keep_if`, `zero_if`, `near`,
  `deadzone`, `snap`, `clip`, `positive_part`, `negative_part`, `hash`,
  `hash(HashOp::Unit, ...)`, and `fade`
  lowered to the existing fixed clamp, select/compare, saturating add/sub,
  bitwise, constant-shift, wrap-add, fixed divide, fixed sqrt, and fixed
  multiply instructions

Unsupported policy, malformed IR, unsupported scalar, dynamic shifts, float
authority, atomics, unordered reductions, resident CPU output, and forged
artifacts fail closed. Wider strategies such as AVX2 and AVX-512 are observed
as host evidence, but canonical `CpuEntry()` admission narrows the executable
profile to the direct 16-byte `Sse2` or `Neon` lane shapes until wider direct
lane runners are checked in. This narrowing is setup/admission policy; it is
not a hot-loop branch and it never depends on workload size or timing.

Execution walks semantic tile indexes in ascending order. Full chunks execute
with the fixed SIMD lane count. Tail chunks use the same vector law with
zero-padded inactive lanes and store only live lanes, so tail handling does not
introduce a second scalar formula. The runner reports processed tile count,
full vector chunk count, tail chunk count, selected strategy, and fail-closed
rejection count.

`cpu/simd/prepare.cpp` is only the prepare orchestrator. Kernel's detail-only
CPU SIMD admission validates caps and delegates the one canonical parse and
`ValidateLowerableIR` pass to generic Compute input admission. Focused owners
under `cpu/simd/prepare/` validate lane caps and bindings, build a temporary
binding plan, and lower it into one compact `PreparedRun`. The
temporary parse and binding plan die when preparation returns. The retained
owner contains only ordered `PreparedInstruction` records, the once-prefix
size, read/write counts, domain, strategy, and the index-use flag. Each
instruction freezes its node, value index, immediate, element width, and
bounded one-byte full/tail executor selectors. One 64-bit field packs the
bounded binding slot in its low 32 bits and the three source fractional widths
in its next three bytes. Field ordering therefore keeps the complete
instruction record at exactly 56 bytes on the supported 64-bit ABI;
compile-time layout and round-trip assertions own that retained-memory bound.

The internal Compute Program calls
`PrepareCpuSimdDispatch(ir, caps, bindings)`. That route performs one admission
parse and zero artifact emissions. Public `RunCpuSimd` uses the same
`LoweringArtifact` as every backend: it performs one admission parse, emits one
deterministic expected `CpuPlan`, and compares the complete
key/source/metadata/canonical payload before preparing the same compact plan.
Public validation therefore remains parse `1` / expected
emission `1`; internal Program preparation is parse `1` / emission `0`, and
repeated runs of that prepared dispatch are `0` / `0`.

CPU execution never re-lowers and then calls a second parsing preparer. It
borrows the artifact canonical bytes, uses one admission and one transient
emission for full identity comparison, and passes the same parsed admission to
compact preparation. The transient emission owns key, kind, metadata, and
source but no canonical vector; no conversion artifact or copied echo exists.
The focused CPU pick contract checks transient admission results rather than
retaining diagnostics in `CpuAdapter` or `CpuSimdDispatch`: valid execution is
exactly `1/1`, internal
preparation is `1/0`, two prepared executions add `0/0`, source forgeries
reject after `1/1`, and forged operation identity or canonical hash rejects
before parse/emission at `0/0`.

For submitted canonical size `C` and generated source size `S`, transient
authentication borrows the canonical vector and compares the artifact's source
directly with the generated source. Its additional retained canonical/source
payload is therefore zero bytes (allocator bookkeeping and capacity rounding
excluded). Final artifact construction owns exactly one canonical payload and
one emitted source payload. It never allocates an equal source-byte mirror.

`RunFixedLane32` and `RunFixedLane64` remain thin typed wrappers over
`cpu/simd/run/body.hpp`. Focused run-body owners provide memory operations,
arithmetic, predicates, bitwise and fixed-point math, the fixed opcode executor
table, and tile traversal. The wrappers provide scalar width, vector carrier,
constant decoding, and direct math bindings through `32/config.hpp` and
`64/config.hpp`; shared cleanup in `run/clear.hpp` keeps the two widths under
one traversal law. Stable `Param`, `Constant`, `ReadUniform`, and pure dependent
instructions, including pure `Quantize` and constant shifts, occupy the
prepared prefix and execute once. `Read`, `ReadAt`, `Index`, `Write`, and their
dependents occupy the loop suffix. A `CpuSimdBindingView` is bound once per Map
run, so tile callbacks change only `begin` and `count`; they do not rebuild or
copy an instruction plan.

Read and write admission proves final addressability before preparation. Full
contiguous and full strided reads use their specialized loaders, tails zero-pad
only inactive lanes, and writes store only live lanes. This specialization is
based on frozen IR and stride evidence, never workload-size timing. Strategy
selection is a fixed `ComputeScalar -> RunFixed*` table before traversal;
preparation maps every admitted value in `[0, IrOp::ReadUniform]` plus the three
specialized full-chunk read/write paths into one bounded executor table.
Compile-time assertions bind its exact extent to
`IrOp::ReadUniform + 1 + 3`. The tile loop performs one direct table lookup from
the frozen selector. It does not switch on binding mode or rebuild function
dispatch per vector chunk. Tail selectors always name the canonical opcode
path, so zero-padding and live-lane stores remain the sole tail law.

Preparation walks the stable-once prefix followed by the per-vector suffix,
computes every value's final use in that actual execution order, and assigns
one reusable physical scratch slot. `ParsedNodeResourcesFor` is the only value
edge classifier: `ReadAt` binding ordinals, write modes, and constant-shift
immediates are never SSA references. A pure Quantize or constant shift is in
the stable prefix when its real operands are stable. A stable value referenced
by the repeated suffix is pinned through the whole run; dynamic values have no
loop-carried edge.

Each prepared instruction freezes the three source fractional widths before
SSA identities are remapped, so physical slots are independent of fixed
format. An executor first evaluates all operands, then commits through
`set_raw` or `set_wide`. A result may therefore reuse one unpinned operand slot
whose final use is that instruction, including across format boundaries;
duplicate operands are retired once. Write has no destination slot.

For the fixed admitted instruction schedule, before any dead-code elimination,
let `P` be its commit-demand peak. Immediately before instruction `i`, demand is
the number of values whose contents must remain available, plus one exactly
when `i` produces a value and no unpinned operand dies at `i`. A dying operand
may carry the completed result because every executor evaluates operands before
commit; otherwise the commit requires a free slot. The deterministic allocator
chooses that dying operand, then a free slot, and grows only when neither
exists. It therefore uses exactly `P <= N` slots, and fewer slots cannot execute
that same non-DCE schedule without overwriting a still-required value or its
current result commit. This does not claim that the admitted schedule itself is
globally minimal under dead-code elimination.

For `L` SIMD lanes, Fixed execution requests exactly
`P*sizeof(uint8_t) + P*sizeof(ValueVec) + P*L*sizeof(WideScalar) +
alignof(ValueVec) - 1`. `ValueVec` size is a multiple of `WideScalar`
alignment, so the first aligned plane also aligns every following plane.
Non-Fixed
execution has neither a 128-bit materialization plane nor its validity plane
and requests exactly `P*sizeof(ValueVec) + alignof(ValueVec) - 1`. The
allocation is rounded up to
`sizeof(std::max_align_t)` words. Execution preflights the declared byte
requirement before laying out typed regions, so the exact rounded allocation
succeeds and an allocation one word shorter rejects as
`cpu_simd_scratch_invalid` before any output write. Instruction arrays and
plan-stability state are compile-owned, not worker scratch. Every stable-prefix
executor is a commit barrier: its first failure terminates the prefix before a
later executor can observe or publish an uncommitted result. Preparation is
`O(N)` and each tile reuses the frozen plan. These are algorithmic and storage
bounds; wall-clock claims still require measurement.

## CPU Backend Admission

Canonical `CpuEntry()` selection through `PickFromCatalog` is a real backend
admission path when host feature observation can freeze executable CPU caps.
The public `PickAccel(Api::Cpu)` supplies that entry from the full catalog;
focused CPU contracts inject the same single entry without compiling the
catalog or native adapters. A successful pick returns:

- `api = Api::Cpu`
- frozen generic `ComputeCaps` with `ComputeApi::Cpu`
- frozen `cpu_caps` with backend `Cpu`
- a usable `ComputeBackendDispatch`
- SDK-free backend info naming the selected CPU SIMD strategy
- an owner lifetime handle

The backend dispatch validates the frozen plan header, caps, artifact kind,
artifact identity, dispatch windows, and staged bindings before execution. It
requires staged outputs, rejects resident inputs/outputs and sequence remap in
this CPU path, builds only a non-owning identity header over the submitted
generic CPU artifact's canonical bytes, performs the generic comparison, and
prepares the internal compact runner from that same admitted parse. It then
executes one full-range pass after proving the submitted dispatch windows are
contiguous ascending coverage of that same range.

## CPU Native AccelKernel Executor

`Api::Cpu` also admits `AccelContext` and `AccelKernel` execution over
host-resident context buffers. This path is native CPU execution, not a
device-command-stream imitation and not a silent fallback from another backend. `Map`
steps reuse the checked CPU SIMD artifact path above after context-buffer
authentication. `Scan`, `SegmentedScan`, `SegmentedReduce`, `Sort`, `Compact`,
`Gather`, `Histogram`, `Partition`, `Reduce`, `Scatter`, `Stencil`, `Transform`,
`Matrix`, `Factor`, `Solve`, and `Spectrum` steps execute the same deterministic
kernel primitive laws through focused CPU primitive owners. Sort preserves stable
ascending key order, but the
CPU backend implementation uses the stable radix path from the
frozen `SortPlan` instead of the quadratic contract-reference insertion
routine or a comparison-sort side path. The CPU sort pass count is derived from
the declared key domain (`sort.key_bits`) and never from workload-size
thresholds. Temporary radix keys, values, bucket counts, and offsets are
context-owned reusable CPU scratch, so warm resident sort rows do not allocate a
fresh comparison-sort workspace on every run. That scratch is execution
resource state only; it is not semantic state, it does not affect graph or
kernel identity, and it follows the frozen radix pass count. Every admitted step
kind shares the same graph order, owner, binding, descriptor-hash, evidence,
and fail-closed reason contracts as Metal and Vulkan; only the private
executor changes.

Node's CPU graph executor admits and projects a primitive exactly once in
`compute/cpu/run/primitive.cpp`: it resolves the frozen binding range into an
eight-entry stack view, validates each byte range, dispatches one primitive,
and projects its stable reason. Algebraic kernels are compiled separately in
`compute/cpu/run/primitive/algebra.cpp`; that leaf owns Stencil, Transform,
Matrix, Factor, Solve, and Spectrum execution and their semantic-status
projection. `execute.cpp` owns reset only. No inline implementation header or
second port resolver remains, so changing an algebraic reference does not
reparse or rebuild reset execution and does not add an allocation, payload
copy, workload threshold, or alternate calculation order.

The CPU implementation includes only the authority needed by that executor.
`cpu/kernel/run.hpp` is the one declaration owner for CPU kernel run, prepare,
prepared run, and submit. Its implementation consumes the backend-neutral bound
run schema plus the CPU collective contract. Each CPU primitive translation unit
consumes its own shape, binding, and reference leaves; it does not include a
root operation header that also declares Metal or Vulkan execution.
`backend/ops/table.hpp` owns the backend dispatch-table schema,
`backend/catalog/entry.hpp` owns one catalog row, and each backend-local
`ops.hpp` owns only that backend's entry factory. No header co-locates dispatch
schema with the cross-backend inventory. Every table accessor has one caller.

CPU/backend identity and output parity are contract-test concerns. Performance
claims require a focused measurement made from the current revision; they are
not emitted by the normal semantic suite.
