# Compute Pipeline Contract

This page owns the shipped `rund::compute::Pipeline` product contract. Pipeline
prepares one frozen declaration-ordered sequence of compiled Programs over
caller-owned resident Buffers, then reuses that prepared sequence without
rebuilding its cross-Program plan.

## Scope

`rund::compute::Pipeline` is the prepared execution owner for a frozen,
ordered sequence of already compiled Programs whose caller-owned Buffers may be
shared across Program boundaries. It exists for simulation ticks and other
resident multi-stage work that needs modular Programs without one native GPU
submission per Program.

Pipeline is domain-neutral. It knows Program identity, Buffer identity, scalar
and Fixed policy, access role, order, resource conflicts, execution state, and
evidence. It does not know bodies, components, worlds, frames, physics,
temperature, electricity, checkpoints, or gameplay meaning.

The current authorities remain:

- Flow owns one Program's typed calculation graph;
- Program owns graph identity and compilation;
- Pipeline owns cross-Program binding, order, resource visibility, one prepared
  execution, and its publication state;
- Batch owns one submission for independent disjoint Jobs;
- Session owns bounded asynchronous admission and lifecycle;
- the caller owns simulation state meaning, checkpoints, restore, and replay
  policy.

Pipeline is an execution plan over compiled Programs. It is not a new Compute
IR, a second graph compiler, a scheduler, a fusion pass, or a native-handle
escape.

## Product Surface

The product surface has one durable owner type. `read(...)` and `write(...)`
are transient typed binding packs; they own no execution state and are consumed
immediately by `then(...)`, `repeat<N>(...)`, or `windows<Max, Tile>(...)`.

```cpp fragment
using Real = rund::compute::Fixed<32, 32>;

auto prepared =
    rund::compute::pipeline(device)
        .then(integrate,
              rund::compute::read(position, velocity, force),
              rund::compute::write(next_velocity))
        .then(advance,
              rund::compute::read(position, next_velocity),
              rund::compute::write(next_position))
        .prepare();

if (!prepared) {
  return prepared.exit_code();
}

rund::compute::Pipeline tick = std::move(prepared).value();
auto completion = co_await session.compute(tick);
if (!completion) {
  return completion.exit_code();
}
```

Per-declared-step profiling is one explicit cold Pipeline choice:

```cpp fragment
auto prepared =
    rund::compute::pipeline(device)
        .profile(rund::compute::PipelineProfile::Steps)
        .then(bounds, rund::compute::read(position),
                      rund::compute::write(extents))
        .then(broadphase, rund::compute::read(extents),
                          rund::compute::write(pairs, pair_count))
        .prepare();

std::array<rund::compute::PipelineStepProfile,
           rund::compute::PipelineIterationCapacity> steps{};
auto profile = prepared->profile(steps);
```

`PipelineProfile::None` is the default. `PipelineProfile::Steps` is fixed
during `prepare()` and requests bounded raw evidence for the same prepared
execution. It does not create another Pipeline, split a Program into a Job,
add a semantic stage name, or alter declaration order. Profiling mode,
timestamps, and observed counters are diagnostic state and are excluded from
the Pipeline fingerprint, Replay identity, output and snapshot hashes,
publication generation, and terminal failure identity.

`Pipeline::profile(steps)` consumes only caller-provided row storage and
returns `Result<PipelineProfileSnapshot>`. The result reports `written` and
`total`; `truncated()` is exactly `written < total`. A ready Pipeline prepared
with the default mode fails with `ProfileUnavailable`. An invalid or
moved-from owner fails with `ProfileInvalid`, and an in-flight owner fails with
`ProfileBusy`. A selected backend that cannot supply a truthful timing still
returns the work and memory row with timing explicitly unavailable; it never
substitutes a measured zero or another clock.

`then(...)` consumes one Program followed by exactly one read pack and one
write pack. Program signature expansion determines the required scalar leaf
types and counts at compile time. Rvalue Buffers are rejected. Pipeline retains
the Buffer owners, so moving the caller's handles cannot change prepared
identity.

`repeat<N>(...)` is the fixed-count resident recurrence authority. The
Program's flattened output types must exactly match the prefix of its input
types; those leaves are loop-carried state and any remaining input leaves are
loop invariants. `N` is a positive compile-time execution bound. The caller
binds the initial state once and the final state once:

```cpp fragment
auto tick = rund::compute::pipeline(device)
                .repeat<8>(solver,
                           rund::compute::read(seed, constants),
                           rund::compute::write(result))
                .prepare();
```

Preparation compiles no second graph. It allocates one private resident
scratch bank for the carried output leaves and exactly one Program-internal
workspace for the complete recurrence. The seed route and the two alternating
bank routes bind that same workspace; they never allocate occurrence-local
copies of Program internal values. Preparation cold-freezes those routes so
the final iteration always publishes to the declared output bank.
The recurrence is one declared Pipeline step and therefore consumes one of the
64 public steps. Its physical prepared iterations consume the separate 1,024
iteration envelope. CPU executes the frozen entries in the same order; Metal
and Vulkan preserve that exact logical order. A recurrence whose complete
physical body is one element-local Map is lowered to one native dispatch per
pre-existing device-capacity window: one lane loads its carried and invariant
inputs, evaluates iterations `0..N-1` in that order, retains carried values in
registers, and writes only the final state. Other Program shapes retain the
canonical ordered command stream with
resource visibility barriers between carried writes and subsequent reads.
There is no host callback, count readback, graph rebuild, compilation, or warm
allocation between iterations.

This lowering is selected only by a semantic proof, never by `N`, element
count, device model, timing, or a tuning threshold. All occurrences must be the
same status-free Map artifact and dispatch plan; every carried output must feed
the matching next input view; all other input views must be invariant; and
carried, invariant, and final output views must satisfy the no-alias proof.
The derived shader is mechanically specialized from the Program's canonical
lowered source and receives a fingerprint distinct from the ordinary Map. It
is not a second graph or calculation authority. If an eligible source cannot
be transformed with exact symbol and replacement cardinality, preparation
fails closed rather than silently recording the slower recurrence.
The compiled recurrence owner is physically separated into `match`, `source`,
and `build`: semantic and resident-view equivalence is proved once, exact
source transformation consumes that proof, and final assembly alone publishes
the immutable recurrence. None of those leaves owns another eligibility path.
Before that complete element-local proof is established, a multi-step,
controlled, collective, aliased, or otherwise nonlocal body is outside
register-recurrence eligibility and keeps its one canonical prepared command
stream. Profiling never changes either classification.

Backend executable-cache admission compares the complete `ArtifactKey` and
the exact specialized source bytes. Source hashes may prefilter that equality
but never authorize a hit. Resident-stride specialization therefore leaves the
canonical graph hashes unchanged; controlled and recurrence variants use the
orthogonal variant field, while exact source equality distinguishes every
layout specialization. Two different shader texts cannot alias even if every
finite hash field collides.

Metal and Vulkan executable caches share this one cache law. A source or
function-name hash selects the collision bucket in expected `O(1)` time, then
the complete `ArtifactKey`, exact source or name bytes, and Vulkan binding
cardinalities authorize the entry. Hash-table iteration is never an execution
or eviction authority. Building `P` distinct pipelines therefore performs
expected `O(P)` indexed admissions instead of a linear scan per insertion and
`Theta(P^2)` aggregate comparisons. Vulkan prepared Map and direct runtime
paths call the same acquire implementation; neither retains a mirrored cache
path. The Metal source-library LRU is intentionally separate and its linear
promotion scan is capped at the fixed 16-entry retention bound.

Cache publication is transactional. A native executable becomes visible only
after both its stable owner and index entry exist; allocation failure destroys
the unpublished Vulkan handles or releases the unpublished Metal owner and
leaves no index row or successful-compile counter. Adapter destruction clears
the non-owning index before destroying its owning executable sequence.
Vulkan's executable is a move-only RAII owner: move clears every native handle
from the source, and its destructor is the sole handle-destruction authority.
Its shader module belongs only to synchronous executable construction and is
destroyed before cache publication. The executable cache therefore retains
the `VkPipeline`, layout and descriptor state required by warm execution,
without extending construction-module lifetime. The detailed native owner
chain is defined by the
[Accel Runtime Resource contract](../accel/runtime/resources.md#prepared-execution).

Vulkan descriptor reuse has no cache-wide hot sweep. Each operation advances
one adapter epoch in `O(1)`; only a collective pipeline actually used in that
epoch lazily resets its cursor and reusable-set bound. Long-lived prepared
leases remain explicit bits and are skipped by the same allocator. The sole
full walk is the exact `2^64` epoch-wrap maintenance boundary, which resets
identity tags before reusing epoch one and cannot alter descriptor contents or
execution order.

On Metal, every planned RAW, WAR, or WAW frontier remains one explicit ICB
command barrier. A frontier after command `i` is encoded as `setBarrier` on
command `i + 1`, which makes all preceding indirect commands complete before
that command begins. Controls and Pipeline-private fixed-grid lowerings live in
the same ICB; the ordinary compute encoder owns only the trailing visibility
barrier after the one full-range execution. Therefore a dependency path with
`k` internal hazard frontiers has exactly `k` ICB visibility boundaries,
independent of device scheduling or recurrence ownership.

The body graph has size `G`, Program-internal physical memory size `I`, carried state
has size `S`, invariant input payload per element is `C`, output payload per
element is `O`, authored logical binding width is `L <= 64`, element count is
`E`, and
the recurrence bound is `N`. Preparation retains one `Theta(G)` compiled body,
one `Theta(I)` chunked virtual arena, and `Theta(S)` carried scratch; the frozen
logical schedule and its exact binding/hazard evidence are `Theta(N * L)`.
They are cold, bounded metadata rather than duplicated graph compilation or
payload storage. `I` is the canonical `MemoryPlan::physical_bytes`, not the sum
of all authored intermediate extents. Ordinary physical owners are capped at
`min(1 GiB, backend storage limit)`. The same canonical placement algorithm is
used on CPU, Metal, and Vulkan; the frozen target limit may only lower that
ceiling.

For `W` pre-existing device-capacity windows, the canonical multi-dispatch
route performs `N * W` dispatches and `N - 1` visibility boundaries. Its
authored commands expose `Theta(N * E * (S + C + O))` payload loads and
stores. The proved element-local Map route performs `W` dispatches, has no
inter-iteration visibility boundary, and exposes
`Theta(E * (S + C + O))` payload loads and stores while retaining
`Theta(N * E)` arithmetic. Dispatches and program-visible payload operations
therefore decrease linearly in `N`. Physical memory traffic and wall time
remain measurements because compiler register allocation, spills, cache,
occupancy, arithmetic cost, driver submission, and device behavior are not
algebraic constants. A body with more than one physical dispatch or any
collective dependency keeps `Theta(N * D)` physical dispatch work.

The 1,024 bound admits multiple bounded solver phases while keeping the
command stream statically finite. A bounded resident count may suppress later work
through the body's existing controls, but it does not invent a second host-side
termination decision. Profile rows retain the logical step index plus the
zero-based `iteration` and authored `iteration_bound`, so executed/skipped
control evidence remains attributable without treating native scheduling as a
semantic loop authority.

Register recurrence is not a numerical reassociation. For each element `i`,
the canonical state transition is `x[k + 1, i] = F(x[k, i], c[i], i)`. The
specialized lane evaluates the same `F`, with the same operation, Fixed
quantization, rounding, overflow, and iteration order, before making
`x[k + 1, i]` the next carried operand. Induction on `k` therefore proves
bit-identical carried state; independence across `i` proves that eliminating
cross-dispatch scheduling cannot change an operand. No reduction tree,
cross-element read, atomic, or schedule-dependent operation is admitted by the
proof.

Internal memory reuse is not a numerical optimization choice. Every internal
Flow value is produced before its same-Program consumers, every Pipeline step
is separated from a later shared-arena user by the frozen visibility frontier,
every recurrence occurrence completes before the next carried-state read, and
Pipeline admits only one execution attempt at a time. Therefore the fixed
recurrence `x[k + 1] = F(x[k], c)` and every ordinary cross-Program transition
observe the same operand bits and Program operation order as separate storage.
Owner identity and byte offset never enter Fixed arithmetic, reduction order,
status priority, or publication. Transactional primary and alternate routes
also share internal memory because the parity gate selects exactly one route
set for an attempt.

Memory observation charges Pipeline-shared arena payload to `shared_memory`
exactly once. Step rows report only their JobWorkspace route metadata,
Jobs, and tile/staging owners. The shared partition plus all rows therefore
reconciles with complete Pipeline-owned memory without
multiplying arena bytes by step count, recurrence `N`, or transactional route
count.

### Resident Window

`windows<Max, Tile>(...)` is the bounded resident recurrence authority. The
body has the same output-prefix recurrence law as `repeat<N>`, followed by two
runtime-owned U32 inputs: the resident count and the increasing window
ordinal. The count is bound once through `window(count)`; it is not repeated
in the user read pack.

```cpp fragment
auto prepared =
    rund::compute::pipeline(device)
        .windows<516096, 1024>(
            fold,
            rund::compute::window(active).until<1>(7u),
            rund::compute::read(accumulator, terminal, geometry),
            rund::compute::write(result, stopped))
        .prepare();
```

`until<Index>(expected)` is optional. `Index` names one flattened recurrent
output-prefix leaf and therefore the corresponding recurrent input leaf. That
leaf must be one U32 scalar. At the start of occurrence `k`, equality with
`expected` replaces the authored transition with the identity transition for
every recurrent leaf. A terminal value produced by occurrence `k` therefore
freezes occurrence `k + 1` and every later occurrence. The expected value,
leaf index, `Max`, and `Tile` are frozen into Pipeline identity; a pointer,
address, host callback, or observed timing never is.

Pipeline consumes each Program's immutable `MemoryPlan` and publishes one
`PipelinePlan` through the contract owned by
[Compute Graph Services](../../../../docs/reference/compute/services.md#resident-windows).
The explicit `windows<Max, Tile>` arguments own resident-window shape.
One optional `MemoryBudget::bytes` applies to the complete
`PipelinePlan::peak_bytes`. The plan and allocator are the same cold
authority: Pipeline shares the plan-owned physical rows across steps, and
backend preparation allocates those rows without recomputing their sizes or
offsets.

For the explicit template arguments `Max` and `Tile`, and
`K = ceil(Max / Tile)`, preparation freezes the `K` resident-window entries in
canonical increasing `k` order. One device-resident count `C` supplies:

```text
base(k)  = k * Tile
count(k) = min(Tile, max(0, C - base(k)))
```

The host never observes `C` between steps or windows. `C > Max` is one device
structured `compute_bounded_count_invalid` failure: every payload dispatch is
suppressed, no output or generation is published, and the canonical first
failure is observed at completion. For `C <= Max`, the windows form a disjoint
ordered partition of `[0,C)`. A logical element `i` is evaluated exactly once
at global ordinal `base + local = i`. The body graph, reduction and merge
trees, source ordinals, grouping, and storage extents are fixed by
`resident<Max, Tile>` before Pipeline planning. The window projection therefore
cannot change Fixed quantization, overflow, reduction order, or output bits.

The prepared window entries are part of the existing Pipeline command graph.
There is no caller-visible or count-driven window executor, tile mode,
workload-size branch, alternate graph, or fallback. CPU traverses the same
frozen entries under one Pipeline run. A nonempty Metal or Vulkan attempt
commits the preflight, all window work, status reduction, and publication in
exactly one native submission. It performs no count or payload readback, graph
rebuild, descriptor growth, or warm allocation. `Max` is positive by the
public template contract; a device-resident `C = 0` still submits the control
preflight because the host does not read the count.

A resident active workset uses that same composition boundary; it is not a
second queue API or executor. One preceding Program maps the caller's final U32
flags and applies the canonical stable
[Compact contract](../accel/compact.md), publishing
`U32 ordinals[Max] || U32 count[1]` into caller-owned Buffers. The following
`windows<Max, Tile>` binds that exact count through `window(count)` and passes
the exact ordinal Buffer to Seed in its ordinary read pack. Pipeline freezes
the RAW hazards between those steps, so the complete Compact, resident-count
preflight, `tile_repeat<N>`, status reduction, and publication remain one
prepared execution and one nonempty accelerator submission. There is no count
readback, intermediate submission, warm allocation, queue copy, or
domain-specific runD surface. Zero, underfilled, partial-tail, and full
capacity counts use the same frozen route. Compact overflow is
`compute_compact_capacity_insufficient`; it suppresses the later resident
window, exposes no readable queue/count prefix, and leaves every downstream
Pipeline state pair and generation unpublished.

The caller owns the meaning that precedes those final flags: active predicate,
canonical domain identifier, duplicate resolution, conflict ordering, and any
winner rule. GYEOL therefore resolves those semantics before workset
publication. runD owns only the domain-neutral stable source-ordinal
compaction, checked capacity, resident count lineage, cross-step hazards, and
bounded execution. Encoding GYEOL policy in Compact, `windows`, or
`tile_repeat` would create a second domain authority and is outside this
contract.

Metal hard-cuts the command-table boundary during cold preparation. A warm run
creates Metal's required single-use outer command buffer, performs one bulk
residency declaration over the frozen unique-resource array, executes the
entire reusable ICB with one range call, and commits once. The runD warm path
visits zero command, range, binding, indirect-grid, or recurrence-state rows
and restores zero bytes. This is not a claim of literally zero host work:
command-buffer creation, the bulk Metal residency call, commit, completion,
and the fixed control observation remain. If `D` is the frozen command count
and `R` is the unique resource count, the host boundary is
`T_outer + T_bulk_residency(R) + 0*D`; no term is proportional to the prepared
descriptor schedule. Vulkan records its immutable primary command buffer
during cold preparation and retains its existing one-submit boundary.

Preparation also seals the rank of window entries in every physical-step
prefix:

```text
rank[0]     = 0
rank[i + 1] = rank[i] + (step[i].window != 0)
```

The table exists only when the Pipeline contains a resident window and has
exactly `S + 1` U16 entries for `S <= 1024` physical steps, so its maximum
value is representable without truncation. Completion receives the canonical
verified prefix from the device and adds `rank[verified]` to iteration
telemetry in `O(1)`. It does not rescan up to 1,024 retained `PipelineStep`
rows after every submission.

The body Program's `MemoryPlan` reports the storage implied by its explicit
`Tile`-sized graph. Pipeline planning reuses that one Program workspace across
all `K` occurrences and reports it in `PipelinePlan::transient_bytes`.
Increasing `K` does not clone the Program workspace. Recurrent state and
caller-owned Buffers remain explicit Pipeline owners and cannot be hidden in a
per-window estimate or allocated by a backend side path.

For seed bank `B0`, the two private destination banks `B1` and `B2`, and
authored transition `F_k`, one `ResidentState` selector is the recurrence
authority:

```text
state(0) = { current = B0, stopped = false }
active(k) = !stopped && k * Tile < C && terminal(current) != expected
state(k + 1) =
  active(k) ? { current = B[1 + (k mod 2)], stopped = false }
            : { current = current, stopped = true }
Bfinal = B[1 + ((K - 1) mod 2)]
seal = current == Bfinal ? 0 bytes : copy(bytes(current), Bfinal)
publish = bytes(Bfinal)
```

`C > Max` remains failure rather than a state transition. The selector is
reset once at submission and updated once per occurrence. CPU checks the same
predicate before executing the body. Metal and Vulkan retain one immutable
command stream: their first inactive occurrence may have produced a private
speculative result before the selector observes the stop, but that result never
changes `current`. Metal resets its device-private selector array in the first
ICB command. Every recurrence-owned command binds its owner's
`ResidentState::stopped` word to the Pipeline-private guard ABI; after a
transition stops the owner, every later owned kernel returns before a payload
access. Vulkan retains its device-authored indirect-argument gating. Canonical
status, telemetry, and selector controls remain unowned, fixed-width commands
and therefore still execute; no later inactive occurrence can write recurrent
payload.

For the complete ordered writer set
`W_k = reset_k + gather_k + body_k + scatter_k`, the first stopped occurrence
establishes `payload(W_j) = 0` for every `j > k` before any such later command
can access its arguments. On Metal the physical ICB commands remain present
and uniformly return through the owner guard. Pipeline-private primitives that
normally consume an indirect grid use their cold-frozen maximum grid and keep
their existing logical-count or indirect-word early return, so invalid and
zero logical work remains non-accessing. Standalone Metal execution retains
the original indirect dispatch. Vulkan represents direct dispatch dimensions
in one immutable indirect-argument arena and sets later dispatches to zero;
dynamically produced dimensions are copied into a private gated slot. Transfer
fills are lowered through the same guarded compute reset dispatch while
captured by a resident recurrence, so a transfer command cannot remain as an
unguarded payload writer.

This is bit-preserving by construction: a stopped transition is the state
identity `current' = current`, not an arithmetic operation. It cannot
quantize, round, overflow, reassociate, or change a reduction tree. Prepared
downstream steps have immutable bindings to `Bfinal`, so the first stop seals
the selected bytes into that canonical bank only when its parity differs.
There is no copy for each inactive occurrence and at most one seal copy over
the recurrence lifetime.

For total recurrent payload width `S` and `J` inactive occurrences, the
removed identity propagation cost was `J*S` device bytes. Both accelerator
selectors add at most `S` seal bytes and `O(K)` constant-width control work.
Metal additionally pays one uniform guard load for each physically launched
Pipeline-private kernel thread; Vulkan retains its first-stop indirect gating
work. The user-facing final publication remains one `S`-byte operation. Thus
payload propagation changes from `Theta(J*S)` to `O(S)`, independent of `J`,
while authored arithmetic order and result bits stay unchanged. Physical
inactive-command scheduling is reported and measured rather than hidden as
useful work.

The recurrence bound, count and terminal contract, selector ordinal, logical
bank topology, and publication route are frozen into the Pipeline fingerprint.
The backend gate schedule is a deterministic lowering of that identity and
the backend ABI. Pointer identity and the runtime selector value are not.

#### Nested Tile-Local Repeat

`tile_repeat<N>(seed_program, action_program, fold_program)` composes a fixed
inner recurrence inside each bounded resident window without materializing
maximum-domain recurrent state. It is a declaration value accepted only as the
body of `windows<Max, Tile>(...)`:

```cpp fragment
auto body = rund::compute::tile_repeat<64>(
    seed_program,
    action_program,
    fold_program);

auto prepared =
    rund::compute::pipeline(device)
        .windows<516096, 1024>(
            body,
            rund::compute::window(active),
            rund::compute::read(outer_seed, seed_external),
            rund::compute::write(outer_result))
        .prepare();
```

The declaration has no independent `prepare`, `run`, binding, memory, or
readback authority. It retains each of the three already compiled immutable
Programs exactly once and becomes executable only when the enclosing
Pipeline is prepared. `N` is a positive compile-time inner bound. Let the
flattened type tuples be:

```text
S = seed-external inputs
T = complete tile-local state
P = inner carried state
Q = invariant tail of T, so T = P || Q
O = outer recurrent state

Seed  : (S..., U32 count, U32 ordinal) -> (T...)
Action: (T...)                         -> (P...)
Fold  : (O..., T...)                   -> (O...)

windows read pack  = (O..., S...)
windows write pack = (O...)
```

`P` must be an exact type prefix of `T`; `Q` is the remaining suffix. The
Action updates only `P`, while `Q` is invariant across all `N` inner
iterations. Fold outputs must exactly match the prefix `O` of its inputs.
These equalities include scalar type, Fixed format, flattened leaf order, and
leaf count and are enforced at template admission. The count and ordinal are
runtime-owned trailing Seed inputs and never appear in the caller read pack.
If Action needs either value, Seed must place it in `Q`; there is no hidden
Action input.

For `K = ceil(Max / Tile)`, valid resident count `C`, outer state `O_k`, and
window-local count `c_k`, the canonical order is:

```text
active(k) = k * Tile < C && terminal(O_k) != expected
c_k       = min(Tile, C - k * Tile)
T(k, 0)   = Seed(S, C, k)
P(k, j+1) = Action(P(k, j), Q(k)) for j = 0 .. N-1
T(k, j)   = P(k, j) || Q(k)
O(k+1)    = Fold(O_k, T(k, N))
```

Seed receives the canonical total count `C`, just like the ordinary
`windows` body, and derives `c_k` through `resident<Max, Tile>(C, k)`. If
Action needs `c_k`, Seed retains it explicitly in `Q`. The Action equation
means that only its `P` prefix changes; every evaluation observes the same
`Q` produced by Seed. The next outer window cannot seed until the preceding
Fold has completed. Seed, every Action iteration, and Fold
preserve their compiled Program node order, Fixed policy, reductions, resets,
and status priority. No cross-Program fusion, reassociation, or approximate
termination is implied.

A count of zero executes no Seed, Action, or Fold and publishes the initial
`O` through the ordinary final route. A partial final window receives its
exact `c_k` and ordinal. Seed must place that count in `Q` whenever Action or
Fold needs bounded tail access; the Pipeline preserves the value exactly but
does not infer a new count lineage across the three already compiled Program
boundaries. The Programs must initialize or avoid every capacity-tail byte
they read. `C > Max` fails the resident preflight before every payload command
and publishes neither output nor generation. `until<Index>(expected)` names a
U32 leaf of `O`; equality in the current outer state suppresses that window,
and a value produced by Fold at window `k` suppresses window `k + 1` and all
later windows. It never shortens the fixed `N` Action recurrence of an already
active window.

Failure order is lexicographic in outer window, phase
`Seed < Action < Fold`, Action iteration, and Program node. The first failure
stops all later nested payload work and prevents publication. Before a
prepared route or raw backend status slot is reused, its result is folded into
the canonical fixed-width Pipeline status; route reuse therefore cannot erase
the first failure. Telemetry for an inactive occurrence observes the same
resident stopped state before reading mutable route counters, so it cannot
re-accumulate values left by an earlier active occurrence. Diagnostics,
profiles, and memory-plan locations retain the logical Pipeline step plus
separate outer-window and inner-iteration coordinates. Seed and Fold identify
their phase and have no fabricated inner iteration. The product `k * N + j`
is never a storage, capacity, ordering, or failure authority.

The four nested-work totals are attempt-wide saturating sums across every
nested logical Pipeline step. Entering a later nested step, including one with
different `K` or `N`, never resets or replaces work already attributed to an
earlier step.

Preparation freezes `O(K + N)` route templates plus fixed control routes. It
does not freeze `K * N` Program graphs, Jobs, workspaces, bindings, or state
banks. In particular, the retained owner graph contains one Seed Program, one
Action Program, one Fold Program, one tile-local invariant `Q` bank, exactly
two alternating carried `P` banks, the required outer-state banks, and one
maximum workspace/View/scratch envelope shared serially by all three Programs
and every outer window. `prepared_command_count` retains the checked authored
occurrence capacity `K * (N + 2)` even when a backend proves a smaller physical
stream. Route-template count, authored occurrence count, physically encoded
Program occurrence count, and backend-reported dispatch count are distinct
report coordinates; none may be inferred from another or used to justify
allocating `K * N` Program/Job/workspace owners.

`PipelinePlan::barrier_count` is the exact number of nonzero boundaries in
that compact route-template schedule. The cold planner projects the canonical
`resource::analyze` hazards to one bit per route boundary, adds the boundaries
required when distinct Programs reuse the shared workspace, and freezes that
same vector for `prepare()`. It is independent of authored occurrence count: a
pure nested body has at most `K + N + 2` compact boundaries even though
`prepared_command_count = K * (N + 2)`. Ordinary steps composed before or
after the nested body contribute their own exact route boundaries; no
`prepared_command_count - 1` proxy is valid.

The Action compiled artifact is retained once. Its alternating views require
at most two parity Job/prepared-resource owners, which all outer windows
reuse. The common accelerator compiler classifies the complete Action template
subrange once. It admits a tile transducer only when all `N` occurrences are
the same status-free element-local Map artifact and dispatch plan, the carried
prefix alternates through the exact two banks, every tail input is invariant,
and all internal Action barriers are present. Indexed reads, controls,
telemetry/status routes, collectives, aliasing, a missing barrier, or any
resident-view mismatch keep the canonical scalar route.

An invariant tail binding may be either an ordinary elementwise read or one
canonical `ReadUniform`, but not both on the same binding. The carried prefix
must remain direct elementwise input. Recurrence source lowering derives this
address mode from checked execution metadata: an elementwise invariant is
loaded once per lane from `base + gid * stride`, while a uniform invariant is
loaded once per lane from `base` and then held outside the inner loop. It never
expands the one-element scalar, applies a window offset to it, or reinterprets a
zero stride. Mixed direct/uniform use and any input with no sole admitted read
mode keep the scalar route.

For an admitted transducer, cold Metal and Vulkan lowering encode exactly one
Action Program occurrence between Seed and Fold for each outer window, so the
physical Program-occurrence shape is `K * 3` instead of `K * (N + 2)`. That
Action shader loads one `P || Q` tile, evaluates the authored transition in
the exact order `j = 0 .. N - 1`, retains `P` in registers, keeps `Q`
invariant, and stores only `P(N)`. Fold advances the logical inner-work counter
by `N`; profiles and public plan counts remain authored coordinates. Because
the admitted Action has no status-bearing path, removing its intermediate
failure coordinates cannot hide a possible failure. An ineligible Action
retains all `N` physical invocations and exact inner-failure attribution.

This is the same recurrence law, not reassociation: induction on `j` gives the
same `P(j)` bits because every operation, Fixed quantization, rounding,
overflow rule, and operand order is unchanged, while element locality makes
lanes independent. Seed and Fold remain outside the transducer, the next Seed
still follows the complete preceding Fold, and one boundary canonicalization
publishes the final selected outer bank before any downstream step consumes
it. Metal and Vulkan use one Pipeline submission with no warm count readback,
allocation, compilation, descriptor growth, binding mutation, or fallback.
CPU executes the identical logical order inside one Pipeline run and retains
the same compact prepared ownership. Metal executes one full ICB range; its
device guards suppress inactive payloads without a warm descriptor, range,
binding, or recurrence-state traversal.

`PipelineStats::rebinding_count` counts post-prepare mutations of the retained
Job, Buffer, View, or prepared-owner binding identity. It is zero by
construction because warm execution has no such mutation path. Metal's and
Vulkan's cold preparation, and Metal's one bulk residency declaration plus
full-ICB execution from a fresh single-use command buffer, are execution
operations, not binding mutations. The zero counter is diagnostic rather than standalone
proof: `compute.window` snapshots every unique nested normal `JobState`; a
transactional recurrence in the same binding oracle also captures each normal
and alternate `JobState`. Both capture Program/workspace/Buffer owners, typed
Views, arena bindings, and the available primary/alternate
`PreparedKernelPipeline` owners, then compare the complete snapshot after
successive executions; the nested oracle also compares overflow.

Logical output leaves are the user-facing binding order. Preparation maps them
through the Program's existing logical-to-physical output projection. If two
logical leaves name one physical Program output, both leaves must bind the same
Buffer owner; Pipeline then retains that physical owner once. Distinct Buffer
owners never trigger an implicit copy to satisfy one aliased physical output.
The current nested Seed/Action/Fold bank model requires distinct physical
outputs and rejects a Program with logical output aliases as
`BindingAliasUnsupported`; ordinary Pipeline steps retain the general alias
projection above.

`PipelineBuilder` and `Pipeline` are move-only. Immutable `StateSnapshot` is a
copyable shared evidence value suitable for a bounded checkpoint ring. The transient binding packs
returned by `read(...)` and `write(...)` are move-only as well. `then(...)` is
valid only on `PipelineBuilder` while building.
`prepare() &&` consumes and freezes the builder and returns `Result<Pipeline>`.
A ready Pipeline has no erase, replace, reorder, or
mutable binding view. Changing a Program or Buffer set constructs a new
Pipeline.

The fixed product envelope is:

```text
steps <= 64
flat prepared iterations <= 1024
compact nested route templates <= 2051 (= 2 * 1024 + 3)
nested contribution is O(outer windows + inner iterations), never their product
combined logical input and output leaves per Program <= 32
flat binding occurrences <= 32768 (= 1024 * 32)
nested route binding occurrences <= 65632 (= 2051 * 32)
unique resources <= 2048
authored commands <= the checked logical Pipeline command capacity
physical native commands <= the selected Device's checked command capacity
```

Preparation allocates exact retained storage inside that bound. It never
spills, grows on demand, or switches execution strategy. Capacity failure is
typed and occurs before backend preparation. The common prepared owner checks
the sum of flat entries and compact nested route templates against
`PipelineRouteCapacity`; it neither checks nor allocates from a flattened
outer-times-inner cardinality. Flat-only schedules retain the narrower
`PipelineIterationCapacity`. Each reported nested coordinate retains its own
range proof. The independent Batch capacity is not a Pipeline limit and is
never consulted by this path.

Standalone `Pipeline::run()` and `Session::compute(Pipeline&)` execute the same
private operation authority. Pipeline exposes `stats()`, `memory()`,
`memory_snapshot(...)`, optional `profile(...)`, and explicit
`read(buffer, output)` for declared write targets after a successful run.
Session reuses the existing Request, Submission, Poll, and Completion product
types; it does not add Pipeline-specific future or task types.

`<rund/compute/pipeline.hpp>` is the focused Pipeline entry. Pipeline is
deliberately excluded from the basic `<rund/compute.hpp>` entry and explicitly
composed by the all-domain `<rund/rund.hpp>` umbrella. Its non-template owner is
compiled, and only signature flattening plus typed read/write binding
construction stays in `pipeline/bind.hpp`. It does not include Flow expression,
math, backend SDK, or Session implementation headers. An unrelated
`<rund/compute.hpp>` consumer does not parse Pipeline binding templates; the
measured focused-header and reverse-fan-out evidence must remain inside the
repository header budget.

## Canonical Order

Declaration order is the sole step-order authority:

```text
step 0, step 1, ... step N - 1
```

Pipeline never topologically reorders, groups, fuses, or chooses a schedule.
Dependencies describe memory visibility inside that fixed order; they do not
select it. There is no public dependency, node-handle, predecessor, or DAG API,
so a caller cannot represent a cycle. The erased plan validates every inferred
edge with `before < after` before backend preparation; an invalid internal edge
is `PipelineInvalid`, never a new public cycle reason. This forward-edge check
is defensive validation of the one declaration order, not a second order
authority.

Program-internal node and reduction order remain owned by each Program. CPU,
Metal, and Vulkan observe the same Pipeline step order even though their
physical execution mechanisms differ.

## Resource Identity

Preparation scans step order, then each Program's input leaf order, then its
logical output leaf order. The first occurrence of a distinct Buffer owner
assigns the next canonical resource ordinal. Later copies of that Buffer owner
reuse the ordinal. Pointer values and backend handles are membership lookup
facts only and are never iterated or hashed.

For each canonical resource, every occurrence must agree on:

- exact Device identity;
- scalar lane type and element count;
- Fixed integer bits, fraction bits, rounding, overflow, and approximation;
- typed view offset, count, stride, alignment, and underlying Buffer extent.

The public Buffer's C++ type proves scalar width and `(I,F)` at the typed
boundary. The erased preparation state still retains and compares the complete
Program-interface policy so that two same-width Fixed formats cannot be
silently equated. Shared producer and consumer slots require exact policy
equality; Pipeline performs no implicit quantization, approximation downgrade,
or policy conversion.

Pipeline accepts a Buffer or `buffer.view(offset, count, stride)`. Element-unit
arguments match `Buffer::size()`. Mutable Buffer lvalues yield `View<T>`;
const lvalues yield `View<const T>`, which is readable but cannot enter
`write(...)`; rvalue Buffer view creation is deleted. A whole-buffer view is
`buffer.view()`.

Canonical resource identity remains the underlying Buffer owner. Each access
retains its exact offset/count/stride/element-width tuple. Hazard, frontier,
claim, barrier, CPU pointer projection, Metal offset binding, and Vulkan
descriptor/range preparation consume that tuple rather than its enclosing byte
envelope. Two strided accesses conflict exactly when selected element byte
sets intersect; holes are never treated as reads or writes. The bounded
Diophantine overlap proof and overflow gates are owned by the resource planner,
not reimplemented by Pipeline.

Map ports consume the retained offset and byte stride directly on CPU, Metal,
and Vulkan and remain zero-copy. Vulkan aligns each descriptor base down to the
device's storage-buffer alignment and specializes the Map byte index with the
exact removed prefix. For semantic offset `o`, aligned base `b`, element index
`i`, and stride `s`, the shader address remains
`b + (o - b) + i*s = o + i*s`. Each canonical dispatch window owns its exact
range, so a large Map value need not fit in one `maxStorageBufferRange`.

Dense-only primitive ports preserve the same public View contract without a
host fallback. CPU uses cold-prepared dense storage and allocation-free
gather/publish loops. A standalone Metal or Vulkan Job owns its dense
normalization storage. Pipeline instead projects every required normalization
into its preflight `prepared_bytes` arena; all private Jobs borrow those slots.
Vulkan uses the same normalization for a contiguous View whose byte offset is
not a legal storage descriptor base, because dense primitive shaders do not
own Map's prefix term. Gather, the authored primitive, and scatter remain in
the same prepared command stream and queue submission. They perform no host
mapping, payload upload, payload readback, or warm allocation. Vulkan View
parameters are 48 bytes of push constants recorded with the command and do not
create a Buffer, transfer, or descriptor binding.

A directly admissible contiguous subview passes its byte offset and range to
the native primitive and remains zero-copy. A normalized input gathers exactly
`count * element_bytes`; a normalized output scatters the same amount. The sum
is reported in `internal_roundtrip_bytes`, and the retained dense storage is
reported as Internal memory. Every Metal/Vulkan gather/scatter is also a
physical dispatch, so it is included in `dispatches` and `final_dispatches`,
while `original_dispatches` remains the authored accelerator Program plan. CPU
gather/publish loops do not manufacture device dispatches. The retained backend
transfer list accounts these bytes in the same preparation sweep that measures
its dense storage; the common graph never guesses backend normalization from
logical stride. Offset/stride is therefore never silently discarded, and Map
retains its native strided path.

## Hazard Law

Program inputs are reads and distinct physical Program outputs are writes.
Pipeline rejects same-step read/write aliasing and two distinct physical output
slots bound to one owner because a compiled Program does not currently publish
an in-place safety proof.

Across different steps, all three ordered conflict forms are legal:

| Earlier | Later | Meaning |
| --- | --- | --- |
| Write | Read | RAW: later step consumes the produced value. |
| Read | Write | WAR: later step may overwrite only after the read completes. |
| Write | Write | WAW: the later declared writer is authoritative. |

Read/read requires no dependency or barrier.

Pipeline lowers every occurrence to the canonical resource planner as
`(owner ordinal, step, role, offset, element width, count, stride)`. The planner
compares only accesses in the same alias group, uses the shared exact
Diophantine intersection predicate, emits a dependency once per ordered step
pair, and retains one canonical exact overlap witness for that pair. A
cross-resource witness takes precedence because it also proves arena-alias
visibility.
Further overlapping ranges cannot strengthen the ordering edge. Pipeline marks
the later step boundary directly from the aligned dependency/witness rows. The
fixed command order remains the execution-order proof, so each later step has
at most one physical barrier.

Each alias group has one deterministic AVL interval index keyed by physical
envelope start and canonical access ordinal. Augmented subtree minimum start,
maximum end, and latest ordinal drive a best-first newest-to-oldest query.
Disjoint subtrees are rejected before the exact predicate, and a complete
visibility frontier stops the query before older transitive conflicts are
expanded. The planner does not collect and sort a complete candidate vector,
so dense repeated full-range writes remain bounded by the balanced path to the
latest writer. Dependency and barrier order is bit-identical to the canonical
linear definition. There is no workload-size branch or alternate planner.

Let `A_g` be the number of accesses in alias group `g`, `V_i` the index rows
expanded for access `i`, `K_i <= V_i` the exact envelope candidates, and `S`
the maximum stride. Index construction costs
`O(sum_g A_g log A_g)` and `O(A)` storage. Best-first reporting and exact tests
add `O(sum_i (V_i log V_i + K_i log S))`; element width is bounded by eight
bytes, so each exact test examines at most 15 relative byte deltas before
Euclid's algorithm. A fully disjoint case has every `K_i = 0`, reducing
`A(A-1)/2` exact comparisons to bounded indexed work. Repeated complete-range
accesses have one exact test per access; genuine multi-range dependencies
remain output-sensitive. Preparation remains inside the fixed 2,048-resource
cold envelope, warm execution walks neither footprints nor dependencies, and
at most `N - 1` physical barrier boundaries exist for `N` declared steps.

Backends may coalesce the resource conflicts at one boundary into one physical
barrier but may not weaken their ordering meaning. Program-internal barriers
remain inside the prepared Program step and are not copied into the Pipeline
planner. Their one authority is the Program's canonical binding-role and alias
stream. A cold linear frontier pass retains accesses since the last global
visibility boundary and emits the next boundary iff a shared alias has RAW,
WAR, or WAW conflict; read/read and distinct aliases emit none. Emitting a
global boundary clears that pending frontier. Metal standalone encoders and
ICB commands and Vulkan standalone or Pipeline primary command buffers consume
the same frozen bits. Thus a Program with `D` graph steps emits at most `D - 1`
internal boundaries instead of unconditionally emitting `D - 1`, while
operation order and every conflicting visibility edge remain unchanged.
Multi-command native primitives keep their own internal pass boundaries; the
Program frontier removes only redundant boundaries between graph steps and
never reinterprets a primitive's private scratch or status schedule.

## Fingerprint

The canonical Pipeline fingerprint is a 128-bit value. Identity policy version
`2` hashes:

```text
F(P) = H(
  "rund.compute.pipeline",
  identity policy version = 2,
  ordered step count,
  ordered Program fingerprints,
  ordered logical leaf role and index,
  first-use canonical resource ordinals,
  scalar types, element counts, and semantic byte extents,
  Fixed policies,
  logical-to-physical output projection,
  ordered exact range dependencies as
    (before, after, resource, before mode, after mode),
  ordered overlap witnesses with both complete strided footprints,
  ordered barrier boundaries)
```

It excludes:

- Buffer, Program, Pipeline, and temporary object addresses;
- native resource handles and descriptor identities;
- allocation and construction order outside declared step/leaf order;
- Device, queue, backend object, timestamps, and current contents;
- diagnostic Program names and domain metadata.

Device and process ABI scope the prepared owner just as they scope a Program
cache; they are not repeated semantic fields. Reallocating all Buffers while
preserving the same alias topology produces the same fingerprint. Changing
step order, Program identity, binding role, resource equivalence, extent, or
numeric policy changes it.

Pipeline is not a result cache. Buffer contents, checkpoint identity, logical
time, and authoritative input hashes remain caller or Replay evidence.

## Preparation

`prepare()` is the only cold transition. It performs, in order:

1. validate owner phase, nonempty membership, and fixed capacities;
2. expand every Program signature and logical output projection;
3. validate Program, Device, type, extent, and numeric policy;
4. canonicalize Buffer owners by first use;
5. reject unsupported same-step aliases;
6. build the exact range-hazard plan and fingerprint;
7. create one external-buffer-bound private prepared step per Program;
8. allocate each Program's private internal storage and its backend-owned
   packed status/control storage;
9. pack immutable Metal command parameters into one retained parameter arena;
10. for transactional state, prepare both immutable parity binding/stream
    projections;
11. prepare retained completion state and apply an optional validated snapshot
    seed to both parity Buffers;
12. publish the ready Pipeline only after every step succeeds.

Phase 1 records each canonical resource's first input step and first complete
output step while it already visits the binding. Transactional admission then
proves the pending-overwrite-before-read law and freezes both resource-partner
ordinals in `Theta(A + P)` total work for `A` binding occurrences and `P`
state pairs; it does not rescan all `A` bindings for each pair. The same partner
ordinal constructs alternate claims and private Job owners in linear resource
and binding passes.

The resource planner emits aligned dependency and witness rows: row `i`
contains one ordered step pair and its canonical exact conflict. Pipeline
consumes each row once, sets the later boundary bit, and stores only its compact
dependency description. Planning is `Theta(D)` after exact intersection for
`D` dependencies, uses no pair lookup table, and retains no full overlap
footprints in a ready Pipeline.

The public Context capability is admitted once for the complete prepared
Pipeline. Each private prepared Program already owns its immutable admitted
Kernel token and Context projection; assembly compares the cached kernel,
context, backend-operation, and owner identities instead of repeating complete
Kernel admission for every Program. A forged or cross-Context request still
fails the one Context admission or an exact cached-identity comparison. The
prepared owner pins its token, so there is no independent revocable token lookup
whose removal could make the cached proof stale.

No public Job is synthesized. The Program convenience cache is not entered,
and a private step Job does not allocate the out-of-line public terminal state
that retains `Run` receipts and failed statistics.
Once primary and optional alternate private Jobs own their exact Buffer owners
and Views, Pipeline releases the builder declarations, cold resource ordinals,
and projection tables before native stream construction or snapshot restore.
A ready `PipelineStep` retains only Program, primary Job, optional alternate
Job, and one write-membership bit. The native prepared owner retains exactly
the active private step owners and state pointers rather than a capacity-sized
64-step table. Those steps are not reachable through Job or Batch, so warm
Pipeline execution does not reacquire their individual submission gates: the
Pipeline phase gate and the backend's one-active-stream gate are the complete
concurrency authority.
Preparation failure publishes no executable Pipeline and cannot modify caller
Buffer contents, Program caches, or Batch membership.

## Resource Claim

Preparation proves static compatibility; execution proves exclusive dynamic
use. The exact Device owns one resource-claim gate. A Pipeline request presents
its already sorted canonical resource array under that gate and acquires all
claims atomically:

- reads share only with reads;
- every write is exclusive;
- any conflict rejects the complete request as resource busy;
- no command is submitted and no Pipeline or Buffer generation changes;
- claims remain held from Session admission through terminal publication.

The Device-level gate makes admission `Theta(R)` with one lock acquisition and
prevents lock-order deadlocks. It does not serialize independent device work;
the gate is released immediately after marking claims, while claims themselves
remain as bounded metadata until completion.

Because this claim vector is unique and canonical by construction, successful
admission validates and marks each resource in the same forward pass: exactly
`R` claim visits rather than a validation pass followed by a second `R`-visit
mutation pass. A conflict at zero-based ordinal `j` rolls back the accepted
prefix before releasing the same Device gate, for at most `2j + 1` visits on
that rejected path. No partial claim is externally observable, and the first
canonical failure reason remains authoritative.

Synchronous `run()` and Session admission enter this boundary through one
`run/start.cpp` authority while holding the Pipeline phase gate. It owns phase
rejection, profile/stat reset, claim timing, saturating conflict accounting,
and every per-run terminal field reset before publishing `Running`. Neither
surface carries a mirrored initialization sequence, so a prior completion
cannot leak verified, submission, write, or failure state into the next run.
The boundary adds no lock, allocation, resource visit, or backend operation;
admission remains `Theta(R)`.

Each canonical claim also freezes whether its resource belongs to a
transactional state pair. That bit occupies existing `BufferClaim` alignment
padding, so the checked two-pointer-width claim extent is unchanged. Terminal
failure publication consumes the bit in the existing canonical claim pass:
state writes are discarded and ordinary writes are poisoned without searching
`P` state pairs for each of `W` failed writes. Publication therefore remains
`Theta(R)` rather than `O(R + W*P)`.

The same claim authority covers Standalone and Session execution. Two callers
cannot bypass it by mixing Program, Job, Batch, and Pipeline surfaces. Existing
independent Jobs keep their disjoint-storage fast path; binding an externally
shared Buffer opts into the common claim boundary. The existing public
Buffer-backed Program convenience run must use this claim boundary as well;
otherwise it would remain an untracked writer beside Pipeline.

Explicit Buffer observation uses the same authority. `Pipeline::read` acquires
a read claim for the transfer and releases it only after the transfer reaches
its terminal state. It therefore cannot race a newly admitted writer after the
tick's execution claim has been released. Read conflict is reported as
`BufferBusy`; there is no unsynchronized convenience read beside this path.

## Execution

Standalone and Session accelerator execution feed the same native completion
record to one finish authority. That authority alone validates submit count,
active-step identity, generation control, verified prefix, first failure,
window overflow projection, occurrence telemetry, per-step profile rows, and
generation rebase. The two public invocation surfaces cannot reinterpret the
same evidence or maintain mirrored completion state.

### CPU

CPU invokes the already prepared Program steps in declaration order. Each
Program keeps its current deterministic worker and reduction policy. A step
failure stops before the next step. CPU publishes zero native command submits
and does not route through Batch or an accelerator-shaped fake executor.
Private Map Jobs resolve and validate their immutable Buffer owners, base
addresses, byte strides, and port counts once during Pipeline preparation.
A warm tick reuses those frozen bindings and changes only the authored dynamic
logical count, cancellation observation, and per-run SIMD counters; it does not
walk value-ID routes or repeat static Buffer/View validation. The frozen
addresses remain valid because the private Job retains every owner and exposes
no input-set mutation. Step evidence accumulates directly from the retained
Job `Stats`; conflict count and overflow ordinal are supplied as two scalar
overrides. Step aggregation uses the retained `Stats` object directly, so its
coordinator copy traffic is zero. The reducer reads only the 17 fields a
CPU Job can produce: dispatch, internal-transfer, graph-read, worker/tile/SIMD,
six dynamic-control counters, conflict count, and overflow ordinal. It no
longer probes the other 35 accelerator, cache, timing, transfer, and publication
fields that are structurally zero or Pipeline-owned.

### Metal

Preparation records the immutable compute commands, buffer bindings, and
conflict barriers in one reusable compute indirect command buffer. Warm
execution creates one ordinary command buffer and compute encoder, declares
the retained resources with one bulk call, executes the complete ICB with one
range call, ends encoding, commits once, and observes one completion. There is
no direct-command partition and no warm command/range table. The ICB begins
with one open command that resets both the 128-byte control and the exact
device-private `ResidentState[0..S)` prefix, then one raw-arena reset only when
a private replacement status word exists, and ends with the canonical
completion path. Resetting resident state is not a second attempt command.

Nested-window state transitions use the following one-authority cut. The
change removes physical controls only; authored Seed/Action/Fold order,
failure coordinates, counters, bank parity, and publication remain invariant.

| Boundary | Previous physical transition | Current physical transition | Preserved state |
| --- | --- | --- | --- |
| Attempt open | control open, then a separate resident-state reset | one grid resets control at lane zero and each exact resident-state row at its matching lane | all controls and states are clear before the first guarded command |
| Seed failure | status reducer selected the failure, then a dedicated Seed-close control stopped the route | the status reducer selects the same lexicographic failure and sets `stopped = outer + 1` in the same single-writer lane | exact `failed_outer_window`, unknown inner coordinate, `Seed` phase, and zero completed inner work |
| Successful Fold `i < K - 1` | Fold advance committed counters/bank, then Seed `i + 1` admitted work | Seed `i + 1` first commits Fold `i`, then performs its own admission in the same single-writer control | the same saturated inner/outer counters and `current = 1 + (i & 1)` before Seed `i + 1` reads terminal state |
| Final Fold | one Fold advance before canonicalization/publication | unchanged | final counters and selected bank are visible before publication |

The deferred Fold advance carried by a Seed is cold-proved per resident-state
lineage. Every Fold for that state must agree on both `inner_bound` and
`inner_advance`; every Seed and Action must use that same bound; and advance is
exactly either zero for individually counted Action controls or `inner_bound`
for a proved Action transducer. Any forged or inconsistent shape fails
preparation rather than corrupting work evidence.

For the focused transduced window-repeat shape with `K` outer windows, one
private raw-status reset, one status-bearing Seed per window, one resident
state, and one publication, the exact Metal control count is
`2 + 1 + K + K + 1 + 1 + 1 = 2K + 6`: open/close, raw reset, Seed folds,
Seed preflights, final Fold advance, canonicalization, and final publication.
At `K = 504` this is 1,014 controls. The 5,042 public dispatches include the
two publication dispatches, so the physical ICB command count is
`5,042 + 1,014 - 2 = 6,054`; the removed stream had 7,062 commands.

Every Pipeline-private Metal kernel entry carries one reserved
`device const uint*` guard at Buffer index 30. Apple's 31-entry per-kernel
Buffer limit makes indices 0 through 30 the complete hardware ABI. Pipeline
reserves index 30, so its Pipeline-private executable admits at most 30
application Buffer entries at indices 0 through 29. Standalone Metal execution
retains all 31 entries. An unowned control command binds a frozen zero word. A
recurrence-owned command
binds the exact `ResidentState::stopped` word for its owner; its value is
uniform for the dispatch, and a nonzero value returns before any payload
access or threadgroup barrier. Checked C++ layout fixes `ResidentState` at
eight bytes with `stopped` at byte offset four. A caller kernel that already
uses Buffer index 30 fails Pipeline-private preparation with
`accel_metal_pipeline_unavailable` rather than being silently aliased. This is
an intentional alpha ABI hard cut, not a claim that the public logical IR limit
or standalone Metal capacity was reduced.

Metal's compute ICB cannot retain an indirect-buffer dispatch. Exactly in
Pipeline-private preparation, gather, controlled resident windows, bounded
sort, segmented reduce, and scatter-reduce therefore freeze their checked
maximum direct grid. Their kernels still consume the device-authored logical
count, segment count, active count, or indirect validity words and return
outside live work. Scatter-reduce initialization and fold both test the
control kernel's zero dispatch words, preserving invalid-index failure
suppression before any output access. Normal Job and Batch preparation keep
the original indirect dispatches and unguarded source/cache identity; the
Pipeline-private cache namespace prevents either executable ABI from being
reused as the other.

Private replacement ranges form one contiguous raw prefix; public ranges form
the suffix and are overwritten in full by their import commands, so an
all-public stream creates neither a reset pipeline nor a reset command. The
reducer reads that raw arena through immutable compact metadata directly;
there is no second canonical status arena or canonicalization copy. A
zero-status stream allocates no raw or metadata dummy storage and uses a
dedicated one-thread completion kernel to publish success; it does not dispatch
the 128-thread status reducer or a no-op reset. Each prepared Program status
binding refers to its fixed metadata slice; warm execution performs no host or
per-dispatch status setup.

Indirect compute commands bind Buffers, not transient encoder byte payloads.
Preparation therefore packs every immutable `setBytes`-style Program parameter
block into one retained Metal parameter arena and binds stable offsets from the
indirect commands. Dynamic values remain ordinary resident Program inputs; no
parameter upload or command rewrite occurs in the warm tick.
The cold capture keeps one 31-slot encoder-state table, but each dispatch
snapshots only its active application Buffer bindings, its one guard binding,
and its active threadgroup bindings into flat sparse arrays. For dispatch `d`
with `b_d` application Buffer bindings and `t_d` active threadgroup bindings,
capture storage and ICB construction visit `Theta(b_d + t_d + 1)` rows rather
than copying and rescanning 31 rows of each kind.
Each sparse array doubles from a fixed minimum only when its capacity is
exhausted. For `Q_b` final sparse rows, total relocation is less than `2Q_b`;
dispatch capture therefore remains linear in the complete command stream and never
reserves the exact cumulative size per dispatch.
On the 64-bit Apple ABI, checked layout assertions fix sparse capture storage
at 96 bytes per command, 32 bytes per Buffer binding, and 16 bytes per active
threadgroup binding. Excluding vector capacity slack, `D` commands use exactly
`96D + 32 sum(b_d + 1) + 16 sum(t_d)` transient bytes; the `+1` is the guard.
Because indirect commands do not inherit Buffers, every active binding is still
materialized for every command. Parameter growth zero-initializes only the zero
to fifteen alignment-gap bytes; payload bytes are inserted once and are not
zero-filled before being overwritten.

The retained pipeline-state and resource lists use pointer-identity membership
indexes while preserving first-command order in their vectors. The indexes are
lookup-only and never iterated, so command identity stays deterministic while
deduplication costs expected `Theta(sum(b_d) + D)` instead of comparing each
binding with every previously retained resource.

Normal Metal command buffers are single-use; the reusable indirect command
buffer is the command cache. A selected Metal Device that cannot prepare the
required compute indirect commands or barriers rejects Pipeline preparation.
Internal dependency frontiers are fixed `MTLIndirectComputeCommand` barriers.
The ordinary encoder performs one trailing Buffer-scope barrier after the full
ICB range. For `D` prepared commands, `B` captured binding rows, and `S`
recurrence states, warm runD execution visits exactly zero `D`, `B`, or `S`
rows and restores zero bytes. It allocates no segment list, range table, gate
buffer, proxy-dispatch table, or alternate direct-command stream. There is no
fallback selection: every warm run consumes the same complete ICB. Batch's
independent-Map packing and host pack/unpack path are never used by Pipeline.

The hard cut intentionally moves inactive-command selection to the device.
Metal still schedules the frozen physical commands, whose uniform guards make
inactive payload work non-accessing; it does not promise that inactive command
issue is free. The minimum host envelope also remains explicit: one outer
command-buffer/encoder lifecycle, one bulk residency declaration for the `R`
unique retained resources, one ICB range call, one commit/completion, and one
fixed control observation. This is zero runD schedule traversal, not zero CPU
participation. Performance evidence must report the device issue tradeoff and
must not relabel the Metal API boundary as eliminated host work.

### Vulkan

Each private Program is prepared in Pipeline-only mode and therefore does not
record the standalone secondary or allocate its standalone mapped status
readback. After status ownership and frontier barriers are fixed, Pipeline
records every Program's prepared primitive steps directly into one dedicated
primary command buffer in declaration order and inserts exact compute
visibility barriers at the frozen frontier boundaries. Every primitive command
is still encoded exactly once, while `N` private Programs retain zero secondary
command pools, zero secondary command buffers, and the primary contains zero
`vkCmdExecuteCommands` indirections. Every private raw status source is
canonicalized immediately after its producing primitive. The following fixed
tree fold consumes that canonical slice before any later primitive may reset or
reuse native status storage. This removes native-buffer alias analysis and
keeps one lexicographic failure authority in declaration order. The stream ends
with one deterministic close that reads and overwrites the mapped 128-byte
terminal block directly. The canonical writers
cover `[0, status_entry_count)` exactly, so a preceding arena fill, transfer
barrier, device-only control mirror, and device-to-staging copy would be
redundant. A zero-status stream allocates no arena, binds no arena descriptor,
emits no arena barrier, and uses a dedicated `local_size = 1` summary kernel.
Warm execution resubmits that completed immutable primary command buffer once;
it does not rebuild its dispatch or barrier stream. The command buffers are
recorded without one-time-submit semantics.

When the common compiler supplies a proved tile transducer, Vulkan prepares
one retained recurrence Map resource for that transducer: its specialized
artifact, resident bindings, parameter Buffer, dispatch windows, descriptor
sets, per-binding `InputWindowPlan`, and `iterations = N` push value are all
cold state. Uniform and indexed-source invariants therefore remain one
base-anchored exact input span across all dispatch windows; direct and index
bindings remain window-local, and a mixed address class is rejected. Each
outer window's single representative Action records that same resource under
the exact resident-state owner. Its `W` Map dispatch windows are captured into the
Pipeline's immutable indirect-argument arena, so a stopped owner zeros the
representative Action without a host decision; an active owner executes the
authored Action transition `N` times inside each shader invocation and stores
only the final carried value. The proof requires an uncontrolled, status-free
Map, so the representative command has no intermediate status or telemetry row
to suppress. After Fold completes, its fixed control route advances the
logical inner-work count by `N`; the final outer Fold then canonicalizes the
selected bank before terminal publication or downstream consumption.

Vulkan profile ownership follows the same logical/physical cut. For `K` outer
windows and `W` native Map dispatch windows, the first canonical Action row
owns exactly `K * W` physical dispatches plus the exact summed workgroup,
work-item, and available timestamp evidence for those dispatches. Every later
Action row retains its `K` authored Action occurrences and original dispatch
counts while owning zero physical work. The focused nested-window contract
checks this shape directly in addition to value, failure, counter, publication,
one-submit, and warm-memory parity.

The retained primary uses the same compiled Vulkan command-resource lifecycle
as adapter-ring primaries and standalone immutable secondaries, but not the
same recording policy. Its closed kind fixes a primary level, no reset-capable
pool flag, no one-time begin flag, and one initially signaled fence. Pipeline
preparation calls the shared cold create/begin/end transitions once; warm
execution still reads the retained buffer and fence handles directly and calls
the external submit path. Warm branches and the native call sequence are:

```text
prepare = create_pool + allocate_primary + create_signaled_fence + begin + end
warm    = reset_fence + queue_submit
```

The standalone Kernel kind remains an inherited immutable secondary without a
fence, while each adapter-ring slot remains a resettable one-shot primary with
its own unsignaled fence and timestamp query pool. No command kind is selected
from work size, device identity, timing, or dispatch count.

Secondary and primary command-buffer boundaries are not synchronization. The
Pipeline barriers are therefore correctness requirements, not optional tuning.
One Pipeline owner permits one active submission; reuse begins only after its
completion fence returns the command to executable state. The common prepared
submission retains that owning reference as its sole active-claim authority;
the `PipelineState` view is derived from the owner and no parallel raw pointer
or Boolean mirrors its lifetime. Inside the Vulkan executable,
`submission::State<VulkanPipeline>` is likewise the sole in-flight owner from
accepted submit through completion take; no backend-local active Boolean
mirrors it.

### Common law

- one nonempty GPU Pipeline attempt, including resident-count preflight, means
  one native submit;
- no CPU retry, backend retry, hidden serial route, or backend switch exists;
- no Program compile, SDK-owned Buffer allocation, descriptor growth, payload
  upload, payload download, or Flow reconstruction occurs in the warm run;
- every device status entry stores zero or one canonical typed Reason code;
- one deterministic fixed-tree fold immediately after each status-bearing
  primitive selects the first nonzero local status ordinal while the 128-byte
  control is still open; later folds cannot replace the first failing Program,
  and one terminal close writes generation and verified prefix;
- each bounded-control source is consumed immediately after its owning
  occurrence, before a reused recurrence route may overwrite that mutable
  native source; the resulting values are accumulated on-device into the
  eight legacy U64 telemetry fields and four nested-work U64 fields before the
  single fixed 128-byte host observation;
- the fixed 128-byte control observation is not payload readback and is reported
  separately from payload transfer;
- output payload and hash remain unavailable until explicit `read(...)`.

## State And Publication

Persistent publication is explicit in the same builder vocabulary:

```cpp fragment
auto tick = pipeline(device)
    .state(published, pending)
    .then(integrate, read(published), write(pending))
    .commit()
    .prepare();
```

Each `state(published, pending)` pair must have identical typed shape and
distinct Buffer identity. Steps read the currently published generation and
write the pending generation. Before any read of a pending Buffer, the frozen
step order must contain a whole-buffer overwrite of it. This admission rule is
the proof that a discarded partial generation can be reused on the next tick
without copying the published generation into it.

`commit()` is required exactly once when state pairs exist and seals the
builder. Preparation creates both immutable parity projections: parity zero
reads each first Buffer and writes its second; parity one swaps those owners
while preserving the same Programs, views, hazards, and order. Warm execution
selects one already prepared projection under the Pipeline phase gate. It does
not rebuild a Job, binding set, graph, descriptor set, or native stream.

Only complete success changes publication. After every Program and semantic
status check succeeds, one Device-claim-gate critical section increments the
Pipeline generation, publishes every active pending Buffer generation, flips
parity once, and releases the claims. There is no per-pair visible commit
window. Pre-submit rejection, cancellation, semantic failure, backend failure,
submit loss, and device loss leave the logical parity and generation unchanged.
If writes may have started, the active pending state Buffers are counted as one
discarded generation but are not poisoned. On a live Device, the last published
Buffers remain readable and snapshot-eligible. After `DeviceLost`, resident
I/O is unavailable: `run()`, `read()`, `snapshot()`, and in-place `restore()`
fail with `DeviceLost`, even though the logical publication counters remain
unchanged. Recovery therefore requires a host-owned `StateSnapshot` captured
before loss and builder restore into newly allocated Buffers on a newly opened
Device. Any non-state write that may have started retains the ordinary poison
law because it has no declared rollback owner.

A submitted live-Device failure also rebases the selected native generation
control before the Pipeline becomes ready again. If that constant-time rebase
cannot be proved, the Pipeline becomes poisoned; it never retries with an
ambiguous completion identity.

`snapshot()` copies only the currently published Buffer of each declared pair,
in declaration order, into immutable host-owned evidence. The snapshot binds
the payload to the Pipeline fingerprint, field type/format/count schema,
generation, and a canonical FNV hash. It contains no Device, backend Buffer, or
native-handle owner. `restore(saved)` overwrites both physical parity Buffers of
a compatible ready Pipeline. The builder route
`.restore(saved).commit().prepare()` performs that same validation and restore
after cold preparation, so a checkpoint from a lost Device can seed new
Buffers on a newly opened Device. `restore` is the final declaration before
`commit`: it freezes further `state`/`then` additions, and `commit` remains the
single sealing terminal. Calling `restore` after `commit`, or adding work after
`restore`, is `compute_pipeline_invalid`; no alternative declaration order is
valid. A
partial restore poisons the destination Pipeline; a complete restore publishes
the saved generation at parity zero.
No failed pending generation becomes readable as published state.

CPU copies and hashes each published field directly into its final snapshot
offset. Metal resolves every shared resident owner under one registry critical
section, then consumes the batch through its shared storage with one host
readback scope and therefore creates no staging allocation or command
submission. Restore likewise admits every destination before the first copy
and uses one adapter scope instead of reacquiring adapter and resident locks
`2P` times.
The public `PipelineLeafCapacity` bound also sizes the transient claim and
transfer rows: snapshot uses fixed `P`-row claim/download stores and restore
uses fixed `2P`-row claim/upload stores. These rows are stack-owned, so neither
operation allocates vectors for admission or transfer routing. The immutable
snapshot owner, its retained field schema, and its payload remain the only
required host allocations.
The CPU contract oracle therefore observes exactly three allocations for a
nonempty snapshot (shared snapshot owner, retained field vector, payload) and
zero allocations for an in-place restore of an already prepared Pipeline.
Vulkan resolves all resident owners once and greedily packs adjacent requests
into host-visible staging chunks bounded by the frozen Device staging budget
`L`. A field larger than `L` is the sole member of its chunk. Each snapshot
chunk records all device-to-staging copies and barriers in one command Buffer,
submits synchronously, and copies and hashes directly into the final snapshot
offsets. Restore packs both parity destinations by the same law. Its one-chunk
case remains an ordered asynchronous transfer; a multi-chunk restore completes
each chunk synchronously before reusing staging capacity. Generic overlapping
subranges of the same aligned resident word fall back to one request per
synchronous chunk in declaration order, so preservation bytes from a later
subrange cannot undo an earlier write.

For `P` state pairs and `B` total published bytes, let `K_s` be the number of
greedy chunks over the `P` published ranges and `K_r` the number over the `2P`
restore ranges. A successful Pipeline checkpoint performs exactly `K_s`
snapshot copy submissions and `K_r` restore copy submissions, with cumulative
staging traffic `B` and `2B`; if `B <= L` or `2B <= L`, respectively, the
corresponding count is one. Peak transient staging is bounded by
`max(L, max(b_i))`, not by aggregate `B` or `2B`. Allocation versus pool-reuse
counts and exact reused bytes are reported separately for every chunk.
The unavoidable payload traffic remains `B` device-to-host for snapshot and
`2B` host-to-device for restore. These are exact successful-operation
command/lease-count claims, not wall-clock speedup claims.

`StateSnapshot::fingerprint()` is the compatible Pipeline schema/topology
identity. `StateSnapshot::hash()` is the immutable saved generation's schema,
generation, and ordered per-field FNV payload-content identity. The same
copy/download that materializes each field produces its leaf hash, so snapshot
creation reads each payload byte once while copying it.
Restore recomputes every retained leaf at its trust boundary before the first
destination write. Fingerprint and content identity are
intentionally separate:
two snapshots may restore to the same Pipeline fingerprint while carrying
different generations or bytes and therefore different hashes.

The closed build/ready type and Pipeline lifecycle are:

```text
PipelineBuilder(Building) --prepare() &&--> Pipeline(Ready)
                                          |
                                          +-> Reserved -> Submitted -> Ready
                                                |            |
                                                +-> Ready     +-> Poisoned
```

Static preparation failure never leaves Building as a valid executable value.
Admission failure or cancellation before native submission returns to Ready.
A successful execution advances one Pipeline generation and every declared
write resource generation before releasing claims and publishing completion.

For a Pipeline without declared state pairs, or for write resources outside a
declared pair, failure poisoning is mandatory. Once CPU execution begins
a step with such a declared write, or once a GPU command has been accepted, any
later failure or accepted cancellation poisons every non-state write that may
have started before claims are released and before terminal Completion
publication. When the first failed physical step `f` is known, the cold
resource plan's `first_write` frontier proves that set exactly as
`first_write(resource) <= f`; a later write remains usable because no command
could have reached it. When the failure step is unknown after native
submission, every non-state write is conservatively poisoned.
No surface may consume a poisoned non-state Buffer. Recovery recreates it or
rebuilds the Pipeline from checkpointed inputs.

Transactional state is not implemented by hidden byte copies. The two Buffers
are caller-owned and explicit, and success commits by swapping one parity bit.
The required pending whole-overwrite-before-read law removes both a warm
published-to-pending copy of `Theta(state bytes)` and per-step guard dispatches.
Snapshot and restore are explicit observation/recovery boundaries and their
host/device traffic is reported; they add no traffic to a normal tick.

Terminal reason remains the only failure authority. Pipeline evidence records:

- declared step count;
- verified successful prefix;
- first failed step when backend status makes it knowable;
- an explicit unknown-step marker for submit loss or device loss;
- canonical resource and barrier counts.

Cold preparation failure also retains a public `Location` when the rejecting
owner is known. `step` is the authored logical Pipeline step, `iteration` is
its canonical physical recurrence ordinal, and `node` is the source Program
graph node before accelerator fusion. Unknown fields use `Location::none`.
This is evidence attached to the failed `Result<Pipeline>`; it neither creates
a second Reason nor enlarges the two-byte hot execution `Status`. A native
failure that cannot identify a Program node may still identify step and
iteration, and must never fabricate the missing node.
Public, resident, and Pipeline-private Jobs share one `finish_prepare`
projection, so every failed `Result` preserves an available Program node.

On GPU, later commands may have physically executed after the first semantic
failure. Their outputs remain poisoned and are never described as published.
`verified prefix` is therefore the evidence name; `completed steps` is not.

The status arena is execution metadata, not a second failure authority. Each
active Program owns one contiguous
`PreparedProgramStatusSlice {first, count}` inside the packed arena. The frozen
layout retains `declared_step_count`, `active_step_count`, the strictly
increasing active-to-declared step projection, and total `status_entry_count`,
so a zero-work Program cannot shift public failure identity. A Program may own
several entries, but every entry uses the common U32 canonical Reason-code ABI
and remains device-resident. Each producing primitive's immediate fold is a
fixed-tree integer selection over its minimum local failing ordinal. Primitive
folds execute in declaration order and only the first failing fold can close
the control, so the result is the lexicographic minimum
`(step ordinal, primitive ordinal, status ordinal)` independent of worker or
workgroup completion order.

The sole host control record is exactly 128 bytes: a four-U32 status prefix,
eight U64 bounded-control telemetry fields, four U32 nested failure-coordinate
fields, and four U64 nested-work totals:

```text
PreparedPipelineControl = {
  generation,
  reason,
  failed_step,
  verified_prefix,
  generated_item_count,
  generated_capacity,
  indirect_dispatch_count,
  indirect_work_item_count,
  iteration_count,
  skipped_iteration_count,
  conflict_count,
  overflow_ordinal,
  failed_outer_window,
  failed_inner_iteration,
  failed_nested_phase,
  reserved,
  executed_outer_window_count,
  skipped_outer_window_count,
  executed_inner_iteration_count,
  skipped_inner_iteration_count
}
sizeof(PreparedPipelineControl) == 128
```

`reason` is the raw canonical `compute::Reason` value and zero means success.
`failed_step == UINT32_MAX` represents both success and an unknown failure;
`reason` together with transport-owned `control_observed` disambiguates them.
A known failure has `failed_step == verified_prefix`. Device loss before the
record is visible produces an unknown step instead of fabricating a prefix.
The public projection converts the sentinel to
`PipelineStats::no_failed_step`.

The eight legacy telemetry U64 fields and four nested-work U64 fields are
generated and accumulated by the device in canonical occurrence and primitive
order through the first failing Program, inclusive. The three nested failure
coordinates use `UINT32_MAX` when unknown; the fourth U32 preserves eight-byte
alignment. Each telemetry dispatch reads the state immediately after its
occurrence, then a compute-visibility barrier publishes that accumulation
before the next occurrence may reuse the same route. A Program boundary fold
merges that Program's status-owned telemetry exactly once and records its
failure only when the control was open on entry. Later Program telemetry and
status folds observe the closed control and are identity transitions.

This placement is part of the memory proof. If a controlled value has graph
lifetime `[first, last]`, its observation executes after `last`'s producer and
before the next graph node. The arena may therefore reuse its half-open byte
range after that boundary without extending `last` to Program completion.
Deferring the read to Program completion is forbidden: separated storage once
made that accidentally work, while deterministic suballocation may legally
replace the bytes immediately after their last graph use.

One prepared open command resets reason, failure identity, verified prefix and
the telemetry suffix before any Program command. It also resets Metal step
profiles; Vulkan's recorded profile fill resets the same rows. One terminal
close advances generation and derives verified prefix from the already folded
failure state. Both commands live in the immutable command stream, so sync,
async, first, warm and restored submissions have the same state transition
without a warm-path host seed.

For occurrence values `t_i` and first failing Program occurrence boundary
`j*`, the reported control is the fixed fold
`t_0 ⊕ t_1 ⊕ ... ⊕ t_j*`. When no Program fails, `j* = T - 1`. Every count
field uses unsigned saturating addition and overflow ordinal uses `min`. Both
operators are closed and associative over U64 with identities `0` and
`UINT64_MAX`, respectively; the frozen order additionally fixes profile
attribution. Additions saturate at `UINT64_MAX`.
`overflow_ordinal == UINT64_MAX` means no overflow. Resident count and
predicate sources remain device-resident; the host neither maps nor separately
copies them. Thus status and telemetry have one observation authority and one
host copy per submitted Pipeline.

The device `generation` is the low 32 bits of the accepted native Pipeline
submission generation and advances modulo `2^32`; zero is valid only on wrap.
A successful submitted tick must equal `low32(public_generation + 1)` before
publication. A nontransactional Pipeline owns one control with stride one. A
transactional Pipeline owns independent primary and alternate controls, so
each physical control advances with stride two. At public generation `G` and
parity zero, cold preparation or restore seeds the primary control to
`low32(G - 1)` and the alternate to `low32(G)`. The resulting sequence is
primary `G + 1`, alternate `G + 2`, primary `G + 3`, without a warm-path host
seed. After a submitted failure, public `G` and parity do not change; the
selected transactional control is therefore rebased to `low32(G - 1)` (or the
single nontransactional control to `low32(G)`) before retry. Pre-submit
rejection and zero-work/no-submit do not advance or rebase device control.
Zero-work success still advances the public U64 generation but observes zero
control bytes.

Let `C` be the number of prepared primitive status import or canonicalization
commands, `Q` the packed status-entry count, `F` the number of status-bearing
primitive folds, `T` the number of prepared device-telemetry sources, `R` equal
one exactly when Metal owns a private raw prefix, and `M` the Metal publication,
resident-seal and dispatch gate controls. Metal reports exactly
`2 + C + F + T + R + M`: one open, one
close, the required imports, one fold per status-bearing primitive, one ordered
telemetry accumulation per source, an optional compact-prefix reset, and the
explicit publication controls. Vulkan reports exactly `2 + C + F + T`: open,
close, canonicalization, primitive folds and telemetry. Zero-work reports zero.
Barriers and user graph dispatches are excluded from this counter. `control_ns`
covers those lifecycle commands through summary observation and may overlap
enclosing kernel, submission, or wait timing.

Metal replacement identity is `(active occurrence, local binding ordinal,
frozen raw offset)`, never native pointer identity. Primary, count, and
predicate telemetry bindings resolve only inside that occurrence's status
slice. Vulkan retains primitive-local status and telemetry slices,
canonicalizes and folds each status slice before advancing, and therefore
never leaves an earlier observation dependent on a later occurrence's raw
range.

Factor, Solve, and Spectrum keep their public per-batch U32 status Buffer
unchanged; callers that explicitly read that declared output still observe the
Kernel status enum and failing batch identity. The same semantic failure enters
the packed execution-status arena through this canonical projection:

| Program status | Canonical terminal Reason | ABI value | Error text |
| --- | --- | ---: | --- |
| `FactorStatus::Singular` | `FactorSingular` | `0x8020` | `compute_factor_singular` |
| `FactorStatus::NonSpd` | `FactorNotPositiveDefinite` | `0x8021` | `compute_factor_not_positive_definite` |
| `FactorStatus::PivotUnderflow` | `FactorPivotUnderflow` | `0x8022` | `compute_factor_pivot_underflow` |
| `FactorStatus::InvalidScaling` | `FactorScalingInvalid` | `0x8023` | `compute_factor_scaling_invalid` |
| `SolveStatus::Singular` | `SolveSingular` | `0x8024` | `compute_solve_singular` |
| `SolveStatus::NonSpd` | `SolveNotPositiveDefinite` | `0x8025` | `compute_solve_not_positive_definite` |
| `SolveStatus::PivotUnderflow` | `SolvePivotUnderflow` | `0x8026` | `compute_solve_pivot_underflow` |
| `SolveStatus::InvalidScaling` | `SolveScalingInvalid` | `0x8027` | `compute_solve_scaling_invalid` |
| `SpectrumStatus::NonConvergence` | `SpectrumNonConvergence` | `0x8028` | `compute_spectrum_non_convergence` |
| `SpectrumStatus::InvalidScaling` | `SpectrumScalingInvalid` | `0x8029` | `compute_spectrum_scaling_invalid` |

`Ok` remains zero and does not select a failure. This projection preserves both
authorities: the public status Buffer owns per-batch algorithm evidence, while
the Pipeline terminal owns one canonical failure for control flow. None of
these semantic statuses collapse to generic `BackendFailed`; that reason is
reserved for a transport or backend failure that has no validated semantic
status. The central primitive-status mapper returns `ReasonInvalid` for an
unknown status ordinal; substituting `BackendFailed` for that programmer/data
contract violation is forbidden.

Each status entry has either one logical writer or an atomic minimum over a
canonical failure key; last-writer timing is never a reason selector. The
failure key orders `(status ordinal, semantic reason priority)`. Semantic
priority is generated from the canonical Reason table and is independent of
the numeric value assigned to a public error code. Prepared Program streams
contain compute and Program-internal barriers, but do not own
status reset, host readback, or terminal publication. The enclosing Job, Batch,
or Pipeline execution owner supplies those status-lifecycle commands through
this one ABI. Pipeline may aggregate many Program slices; Job uses the same ABI
with one Program. A Pipeline-only status implementation is not admitted as a
second failure authority.

## Failure Vocabulary

The generated Compute Reason table remains the single failure authority.
Preparation and execution select the narrowest meaning below and return it
through the common Compute result vocabulary.

| Reason | Meaning |
| --- | --- |
| `PipelineInvalid` | Moved-from, unprepared, or structurally invalid owner. |
| `PipelineEmpty` | Preparation has no declared Program step. |
| `PipelineCapacity` | A fixed step, binding, resource, or command bound is exceeded. |
| `BindingDeviceMismatch` | A Program or Buffer does not belong to the builder's exact Device. |
| `ShapeMismatch` | A binding's scalar lane, element count, or complete byte extent differs from the Program slot. |
| `FixedFormatMismatch` | Fixed `(I,F)`, rounding, overflow, or approximation policy differs. |
| `BindingAliasUnsupported` | One step aliases a read and write, or aliases logical outputs that map to incompatible physical outputs. |
| `BindingDuplicate` | Distinct physical output slots in one step bind the same Buffer owner. |
| `PipelineMemoryBudget` | `PipelinePlan::peak_bytes` exceeds `MemoryBudget`. |
| `BoundedCountInvalid` | A resident device count exceeds the authored `Max`; payload work and publication remain zero. |
| `PipelineBusy` | The same prepared Pipeline already has an active request. |
| `PipelinePoisoned` | A previously submitted execution failed or was canceled after writes became possible. |
| `BufferBusy` | A dynamic read/write claim conflicts with an active execution. |
| `BufferPoisoned` | The Buffer may contain unpublished writes from a failed execution. |
| `ProfileUnavailable` | Step profiling was not enabled before Pipeline preparation. |
| `ProfileInvalid` | Step profiling was requested from an invalid or moved-from Pipeline. |
| `ProfileBusy` | Step profiling was requested while the Pipeline execution is in flight. |

`BackendFailed`, accelerator preparation reasons, `Cancelled`,
`AlreadyCompleted`, and Program-owned reasons retain their existing meanings.
The ten Factor/Solve/Spectrum execution reasons above are Program-semantic
terminal failures and retain the public status Buffer rather than replacing it.
Cycle is absent deliberately: the public type surface cannot express a
dependency edge, and the private plan admits only forward inferred edges.

## Read And Hash

`Pipeline::read(buffer, output)` is the explicit host observation boundary. It
accepts only a declared write resource from the latest successful generation
and reuses the existing typed Buffer transfer implementation.

The Pipeline output set contains each canonical resource written by any step,
once, in canonical resource order. Reading one resource records its leaf hash.
For a transactional state pair, both public Buffer names resolve to that one
canonical output and current parity selects only the physical read source. The
pair consumes the same observed bit as an ordinary output; it never creates a
second leaf and never bypasses aggregate completion. Reading initial or freshly
restored state before this Pipeline has completed a run remains valid state
observation but does not manufacture an execution output epoch.
The aggregate output hash remains zero until every output-set resource has been
observed for the same successful generation, then becomes:

```text
H_output = H(output count,
             ordered resource ordinal,
             scalar and Fixed policy,
             element count,
             ordered leaf content hash)
```

Read order cannot change the aggregate. A read's accelerator transfer submit,
bytes, and timing belong to the read evidence and do not retroactively change
the Pipeline tick's one-submit result.

## Session Integration

The private Session Compute payload is one type-erased Compute operation value
containing a retained owner and a static
operation table for reserve, submit, advance, cancel, result, evidence, and
release. Job and Pipeline both enter that owner; public Request, Submission,
Poll, Completion, and `Session::compute(...)` vocabulary remains singular.

Type erasure occurs once at the Session boundary. No indirect call is added to
element, tile, dispatch, or reduction loops. Pipeline's synchronous and Session
paths call the same reserve/execute/publication owner, so cancellation and
close cannot acquire a second state machine.

`Session::compute(Pipeline&)` returns the move-only `Request`. Call
`.submit()` or directly `co_await` the rvalue Request. `Submission::poll()`
distinguishes admission (`submitted`), actual backend start
(`backend_submitted`), terminal publication (`completed`), and the exact
Reason; an admitted nonterminal Poll has `Reason::Ok`.
`wait_for(duration)` waits only up to the caller's bound and returns the same
non-owning `Poll`; timeout does not cancel, detach, or release native work.
`wait()` owns the one terminal Completion, and `cancel()` requests cancellation
through that same operation owner.

Claims are acquired all-or-none at submit and remain held through terminal
publication. Pre-backend cancel or close releases them and returns the Pipeline
to Ready. CPU cancellation after a write can start records a known failed step
when the ordered executor can prove it. GPU cancellation after accepted native
submission waits for backend completion and names a failed step only from a
valid observed 128-byte control record. In either path, declared transactional
pending state is discarded with parity/generation unchanged; only non-state
writes are poisoned. Session close stops admission, requests cancellation,
waits one terminal Completion for each accepted Compute operation, releases its
claims, and only then resets Runtime state.

## Evidence

Pipeline reuses the common `Stats` counters. Immediately after a successful
tick and before explicit read:

```text
dispatches             = sum of physical Program dispatches, including each
                         required device View gather/scatter
command_submits         = 0 on CPU, 1 on nonempty Metal/Vulkan
pipeline_compiles       = 0
buffer_allocations      = 0
uploaded_bytes          = 0
download_events         = 0
downloaded_bytes        = 0
internal_roundtrip_bytes = 0 for whole-buffer and Map-native View bindings;
                           otherwise exact dense-port gather + publish bytes
external_roundtrip_bytes = 0
output_hash             = 0
graph_hash              = low 64 bits of the Pipeline fingerprint
```

Pipeline-specific evidence is the nested fixed-size `stats().pipeline`
projection rather than parallel top-level counters:

| Field | Exact meaning |
| --- | --- |
| `step_count` | Frozen Program step count. |
| `resource_count` | First-use canonical Buffer-resource count. |
| `barrier_count` | Exact nonzero boundaries in the compact frozen schedule: canonical resource hazards plus shared-workspace reuse; independent of `prepared_command_count`. |
| `claim_conflict_count` | Complete Pipeline admissions rejected by one Buffer claim conflict. |
| `verified_step_count` | Contiguous declaration-ordered prefix whose Program completion is known successful. |
| `failed_step_index` | First known failed step, or `PipelineStats::no_failed_step` when no failed step is known. |
| `status_entry_count` | Packed device status entries owned by all steps. |
| `control_byte_count` | Host-observed control bytes; exactly 128 iff the backend reports `control_observed`, otherwise zero, including CPU, zero-work, and control-lost failure. |
| `control_command_count` | Exact `2 + C + F + T + R + M` Metal, `2 + C + F + T` Vulkan, or zero-work command law above; separate from Program dispatches and barriers. |
| `prepared_template_count` | Compact retained route-template count. It does not count native occurrence references. |
| `prepared_command_count` | Checked authored occurrence capacity. A nested body contributes `K * (N + 2)` logical Seed/Action/Fold occurrences even when a proved Action transducer lowers them to `K * 3` physical Program occurrences. It does not imply distinct Program, Job, binding, or native-command owners. |
| `rebinding_count` | Post-prepare retained binding-identity mutations. The immutable prepared executor reports zero by construction; cold native descriptor encoding is not a mutation, and the structural owner/View snapshot is the independent proof. |
| `claim_ns` | Saturating nanoseconds spent in the resource-claim boundary; diagnostic timing, not a portable performance claim. |
| `control_ns` | Saturating nanoseconds spent resetting, reducing, and observing control state; diagnostic timing, not payload-readback time. |

`Stats::control` reports device-generated work: generated item/capacity totals,
indirect dispatch and logical work-item counts, executed/skipped bounded
iterations, deterministic conflict count, and the first overflow ordinal.
`Stats::publication` reports the durable state boundary: current generation,
successful atomic commit count, discarded pending-generation count, snapshot
bytes/hash, restore transfer bytes, and device-loss count. Commit/discard and
generation survive the per-run statistics refresh; graph work counters remain
per execution. Snapshot and restore update publication/transfer evidence only
when the complete operation succeeds.

Session telemetry copies this projection and the common Compute Stats without
changing their meaning. A moved-from or invalid Pipeline returns unavailable
common evidence and the value-initialized nested projection; it never looks
like a successful zero-work CPU tick.

### Step profile

One `PipelineStepProfile` row corresponds to one frozen declaration ordinal.
Rows are emitted in ascending ordinal order, and `index` is that canonical
ordinal. `program` equals the step Program's public fingerprint. It is an
identity correlation value, not a diagnostic Program label or a semantic name.
The consumer remains the sole owner of names such as `bounds`, `broadphase`,
`contact`, `solve`, and `publication`.

`PipelineProfileSnapshot::execution` is the complete aggregate `Stats` copied
in the same Pipeline-gate epoch as the step rows. Its
`verified_step_count` and `failed_step_index` remain the failure-correlation
authority; a row stores no second failure reason or outcome. Step work is the
latest accepted execution, not a lifetime accumulation.
`PipelineStepStats::sample_count` is zero when that execution's work evidence
is unavailable and one when it is available. A verified zero-work step
therefore reports one sample with zero work counters. A row from an earlier
generation is invalidated before another submission and cannot reappear after
an unknown submit loss or device loss.

The work projection has these exact meanings:

| Field | Exact meaning |
| --- | --- |
| `original_dispatches` | Authored Program dispatch count before fusion or physical View transfers. |
| `final_dispatches` | Actually executed physical dispatch count, including required device View gather/scatter commands. |
| `barrier_count` | Exact conflict barrier boundary immediately before this step; zero or one under declaration order. |
| `worker_count`, `participating_workers` | Configured and participating CPU workers when owned by this step. |
| `tile_count`, `tile_size` | CPU tile evidence when owned by this step. |
| `vector_chunks`, `tail_chunks` | CPU SIMD full-vector and tail evidence when owned by this step. |
| `workgroup_count`, `work_item_count` | Native accelerator workgroup and logical work-item evidence when the backend owns an exact count. |
| `control` | This step's generated work, indirect dispatch, bounded iteration, conflict, and overflow projection. |

On CPU, `original_dispatches` is the frozen CPU runtime-step count and
`final_dispatches` is the worker-backend dispatch count actually executed for
that attempt; multi-pass collectives may therefore make them differ. On Metal
and Vulkan, the authored native plan supplies `original_dispatches` and the
executed command stream supplies `final_dispatches`. A backend leaves
workgroup/work-item fields at zero when it does not retain an exact owner for
that count; it never derives them from Buffer length or another heuristic.
Metal sums checked grid and thread geometry only when every command owned by
the declared Program is direct. Vulkan does the same from retained direct Map
dispatch windows. Either backend leaves both totals at zero for the whole row
when one command is device-authored/indirect, unsupported by the exact
projection, or overflows checked arithmetic; it never publishes a misleading
partial total. Per-step `control.indirect_work_item_count` remains the separate
logical dynamic-work authority where that Program owns telemetry.

Pipeline-wide resource claim, submission, terminal control, publication,
snapshot, restore, and explicit read costs are not apportioned to a step.
They remain in the aggregate `Stats` and shared Pipeline ownership.

`StepTiming::duration_ns` is a nanosecond duration from the declared
`StepClock`; a zero sample count, `StepClock::Unavailable`, or
`StepTimingRelation::Unavailable` means timing is unavailable. A truthful
interval may have duration zero at the clock's resolution, so
`sample_count != 0` distinguishes measured zero from missing evidence. The
duration is for the latest accepted execution and sample count is zero or one;
it is not a cumulative average. `HostSteady` is a monotonic host clock and
`Device` is the selected backend's device timestamp domain projected to
nanoseconds.
`StepTimingRelation::Exclusive` means the reported step intervals are disjoint
under that clock and may be summed. `NonAdditive` means overlap or native scope
prevents such a sum. The backend may publish either relation only when its
timestamp contract proves it; otherwise the timing is unavailable. Saturating
duration or sample evidence remains raw and is exposed by `saturated()`.

Each row's `memory` is the disjoint Pipeline-owned partition attributable to
that private step: both prebuilt parity Jobs when transactional, their retained
metadata, private internal Buffers, CPU tile storage, and step-owned staging.
`shared_memory` owns the remainder: Pipeline coordinator metadata, the typed
View arena, resource and hazard plans, terminal control, native prepared
command and optional query resources, checkpoint/read staging, frame, and
transfer traffic. `memory`, every row partition, and `shared_memory` use the
selected backend and `MemoryScope::Pipeline`; the surrounding field establishes
whether the value is the complete owner or one disjoint partition. No byte is
divided proportionally among steps.

For every `MemoryCategory` counter and each of its `current`, `peak`,
`cumulative`, `reused`, and `budget` coordinates, saturating addition obeys:

```text
snapshot.memory
  = saturating_sum(snapshot.shared_memory,
                   every written-or-unwritten canonical step partition)
  = Pipeline::memory() at the captured epoch
```

Caller-owned Buffers and shared Program owners are not Pipeline-owned memory
and never enter a step or `shared_memory`. `referenced_resource_bytes` reports
the saturating sum of the full logical extent of each canonical caller Buffer
exactly once as a separate non-owning footprint. It is excluded from the
reconciliation formula. Truncating caller row storage changes neither totals
nor the complete-memory summary.

Profiling preserves the existing terminal-control ABI. The sole
`PreparedPipelineControl` observation remains exactly 128 bytes and
`control_byte_count` retains its current meaning. An enabled native profile
uses a separate preparation-bounded evidence block observed at the same
submission completion boundary. Its exact additional native commands, bytes,
and later host snapshot cost are reported separately as
`instrumentation_command_count`, `instrumentation_byte_count`, and
`observation`. The first two belong to the profiled execution. `observation`
measures only the caller's allocation-free `profile()` row and summary
collection after execution, with `HostSteady`, `Exclusive`, and one sample.
It is outside the Pipeline run interval, is not additive with step timings,
and is never subtracted from a synthetic useful-work total. Instrumentation
bytes are the exact compact backend evidence width observed for that run, not
payload traffic or the retained host projection (which remains in
`MemoryStats`). For `D` declared steps and `A` active Programs, Metal reports
`0` extra commands and `64D` evidence bytes. Vulkan reports three profile-block
reset/visibility commands and `64D` bytes without device timestamps; when
timestamps are available it additionally reports one query reset, `2A`
timestamp commands, and `16A` query-result bytes. A native all-zero-work stream
reports zero instrumentation commands and bytes.

Profiling never disables register recurrence. The profiled and unprofiled
Pipeline use the same specialized artifact, dispatch windows, carried-state
order, and one native submission. Because one physical recurrence dispatch
implements several authored occurrence rows, raw physical work and a native
timestamp cannot be divided among those rows without inventing evidence. The
first canonical occurrence row therefore owns the exact physical dispatch,
workgroup/work-item, and available device-timing evidence once; later
occurrence rows retain their authored dispatch identity and report zero
physical dispatch with timing unavailable. No proportional timing or work is
fabricated. Summing physical dispatches over the recurrence rows equals the
Pipeline's actually executed dispatch count.

With `PipelineProfile::None`, preparation retains no step query or evidence
block and warm execution performs no per-step clock read, evidence copy, query
command, resolve, profile observation, additional submission, fence, wait,
payload readback, or heap allocation. Enabling `Steps` preserves the same
algorithmic submission count and prepared dependency plan but measurement
itself is not free. Its wall overhead is measured separately against the
disabled Pipeline under the repository performance method.

`Stats::dispatches` reports physical Program work. On Metal and Vulkan,
`final_dispatches` includes device View gather/scatter dispatches and
`original_dispatches` remains the authored Program dispatch count, so the added
physical cost is visible instead of being mistaken for fusion. The nested
projection separately reports status-control dispatches and retained native
control commands; those do not inflate Program work. `command_submits` includes
the complete stream once.

Pipeline retained memory follows the existing category authority. Step and
binding owners, the hazard frontier, fingerprint storage, and typed operation
table are Host/Metadata. Caller Buffers remain caller-owned physical storage.
Program-private intermediates, the status arena, the Metal parameter arena,
and backend command resources are retained Internal storage. A Vulkan mapped
128-byte terminal block is Staging. Snapshot rows enumerate each retained owner once;
warm execution may change contents and generations but not retained capacity.

Pipeline workspace placement consumes Program chunks without narrowing the
graph storage contract. A chunk at most
`min(1 GiB, backend storage limit)` may share a physical owner with other
ordinary chunks in the same step at 256-byte-aligned offsets. A chunk larger
than that ordinary ceiling owns its complete offset-zero Buffer and excludes
every other chunk for that step. Nonconcurrent steps may deterministically
reuse that owner, whose capacity is the maximum assigned extent rather than
the sum. No later coalescing pass may merge these frozen owners. Therefore for
oversized assignments `L_s` at step `s`, one owner contributes
`max_s(L_s)` bytes, never `sum_s(L_s)`, while concurrent chunks remain
additive. The plan exposes the largest chunk bytes and its logical
step/iteration/chunk location so capacity failures can name the controlling
workspace without backend addresses.

`persistent_bytes` is the exact logical payload of caller Buffers referenced
by the Pipeline. `peak_bytes` is the exact Pipeline-owned planned payload:

```text
peak_bytes = state_bytes + transient_bytes + prepared_bytes
```

`prepared_bytes` is the sum of the dense View arena and primitive scratch arena
required by Metal or Vulkan prepared commands; CPU has no such term. Pipeline
owns those Buffer objects once. Private Jobs, recurrence banks, and
transactional alternates borrow the same arenas and cannot create another
owner. `view_bytes`, `view_step`, `view_iteration`, and `view_binding` identify
the largest View requirement; `scratch_bytes` and `scratch_count` identify the
scratch backing and page count; `largest_*` continues to identify Program
workspace. `total_bytes = persistent_bytes + peak_bytes` is checked without
saturation and is the complete planned Buffer observation. `MemoryBudget`
admits `peak_bytes`; a budget failure therefore occurs before state, workspace,
View, or scratch Buffer materialization.

Let the simultaneous dense View requirements of graph node `n`, sorted by
decreasing bytes with binding ordinal as the tie break, be
`s(n,0), s(n,1), ...`. Graph nodes and Pipeline steps execute in canonical
sequence, while Views used by one node may be live together. The backing arena
is raw word storage, so non-overlapping uses may share a slot even when their
scalar types differ. Therefore the minimum safe logical slot payload is:

```text
view_payload = sum(j) max(n) s(n,j)
```

Rank `j` is one physical slot. The lower bound follows because the node
requirements and every feasible slot-capacity vector can both be sorted
decreasingly, after which feasibility requires the slot vector to dominate
every node's requirement vector componentwise. Thus slot `j` must be at least
`max(n) s(n,j)`. The ranked construction attains every bound and is therefore
optimal under the frozen View-transfer lifetime. The placement never depends
on pointers, allocator order, worker count, or runtime payload.

That lifetime has an explicit backend happens-before edge. Input gather is
visible before the authored primitive, and the primitive or output scatter is
visible before the next graph node may reuse the same slot. An input-only node
therefore closes with a compute-to-compute Buffer barrier as well. Slot reuse
never relies on queue submission order without a memory dependency.

Recurrence preparation analyzes only the seed occurrence and the two frozen
bank phases. From iteration three onward `same_recurrence_phase` proves the
same Job, binding layout, and View requirement as iteration `i - 2`; planning
records reuse without walking the Program graph again. For iteration bound `K`
and Program graph size `G`, an exact two-bank recurrence is therefore
`O(3G + K)`, not `O(KG)`. An occurrence whose authored View changes with its
iteration, such as a resident-window ordinal, is not the same phase and is
analyzed independently. That path remains `O(KG)` because eliding its graph
walk would discard information needed to prove the distinct byte range; it
does not select a runtime strategy or create another placement authority.

Each slot also retains the strongest natural scalar alignment among its uses.
The slots are then suballocated into U32 backing owners at the maximum of that
natural alignment and the selected backend's published storage-offset
alignment. At most `min(1 GiB, backend storage limit)` of ordinary slots shares
one owner; a larger admitted slot remains one dedicated owner and is not split.
Define
`view_storage = prepared_bytes - scratch_bytes`; this is the exact View
backing-owner payload including alignment holes. For `m` slots and
`A = max(storage alignment, largest scalar width)`,

```text
view_payload <= view_storage < view_payload + m*A
```

The owner and offset of every slot are frozen before allocation. Metal binds
the byte offset directly; Vulkan consumes the same offset in its storage
descriptor. Both the allocation count and prepared bytes therefore describe
the physical suballocation plan actually consumed by prepare.

Primitive scratch uses the same immutable planning boundary. Let `P` be the
selected backend's largest storage binding and `A` its offset alignment. Each
admitted Kernel operation publishes its simultaneous temporary byte requests
in canonical request order. Request `r` is placed in the first page whose
aligned remaining range admits it; otherwise one page is appended. The
operation boundary resets placement after sealing a barrier before the next
scratch user. If the largest operation envelope has `q > 0` pages and, among
operations with that count, maximum aligned terminal extent `L`, the Program
needs exactly

```text
(q - 1) * P + L
```

bytes. Because operations, Pipeline Programs, and recurrence occurrences
execute serially, the shared arena is the maximum page envelope, not the sum of
operation or Program arenas. Metal and Vulkan consume those exact page owners
and byte offsets. Prepared primitive objects own only lightweight borrowed
ranges; destruction and pool release remain the Pipeline owner's
responsibility. The added barriers express only the memory visibility required
by overlapping physical scratch; arithmetic dispatch order, reduction,
overflow, and publication order remain unchanged and therefore result bits do
not change.

Logical Buffer payload, backend allocation granularity, host container
capacity, driver metadata, and transfer-pool high-water marks are different
dimensions. The cold plan includes its own typed View Buffer payload but does
not combine opaque dimensions into a false scalar device bound. After
successful prepare, `memory()` and `memory_snapshot()` enumerate View and
scratch Buffers once and measure retained host, device, resident, and staging
categories from their actual owners. Snapshot rows classify scratch as
`MemoryUse::Scratch`; Resident rows report logical payload and Device rows
report actual physical allocation. Backend allocation or native preparation
failure retains the owning typed reason.

Host-width binding, View, snapshot, and recurrence products consume the one
private `compute/size.hpp` law; U64 plan totals consume the Kernel checked law.
Prepared window-copy cardinality is derived and checked once before reserve.
The emission pass consumes that frozen count and immutable Job output shapes;
it does not repeat the same overflow predicate. Publication byte offsets and
strides are likewise derived once and copied into the backend records.

Let `V(x) = capacity(x) * sizeof(x::value_type)` for a retained vector and let
all additions saturate at `UINT64_MAX`. For a private step Job, define:

```text
M(Job) = sizeof(JobState)
       + V(inputs) + V(input views)
       + V(write inputs)
       + V(graph buffers)
       + V(outputs) + V(output views)
       + V(CPU View input transfers)
       + V(CPU View output transfers)
       + V(CPU View dense Buffer owners)
       + CPU retained-run host metadata

M(JobWorkspace) = sizeof(JobWorkspace)
                + V(Program-internal Buffer routes)
```

The private form leaves `JobState::terminal` null and therefore owns no
`JobTerminalState`. Public convenience and resident Jobs allocate exactly one
such terminal owner at their cold construction boundary; it is included in
their Job memory observation, not in `M(Job)` above.

The exact common Pipeline coordinator metadata extent is:

```text
HostMetadata(Pipeline) = sizeof(PipelineState)
                       + V(steps)
                       + sum_step(M(primary private Job)
                                  + M(optional alternate private Job))
                       + sum_unique(M(JobWorkspace))
                       + V(shared Buffer owners)
                       + V(window prefix rank)
                       + V(resources)
                       + V(claims)
                       + V(alternate claims)
                       + V(state pairs)
                       + V(output state)
                       + V(output lookup permutation)
                       + V(range dependencies)
                       + V(barrier boundaries)
```

`V(steps)` already contains each inline `PipelineStep`; it is never added a
second time. The ready step owns no binding, View, or resource-ordinal vector;
those values have exactly one execution authority in its private Job. On the
64-bit Apple ABI the checked `PipelineStep` extent is 56 bytes.
`PipelineBinding` is 72 bytes and an ordinal is four bytes on that ABI.
`BufferClaim` remains 16 bytes after adding transactional-state membership.
Nontransactional Pipelines allocate no alternate-claim vector.
Output generation, leaf hash, resource ordinal, and observed state share one
exact output-state vector; a pointer-sorted ordinal permutation is the only
additional read lookup. The cached status-entry count and remaining unobserved
count are scalars already included in `sizeof(PipelineState)`.
Each distinct private Job and its vectors are included once in the coordinator
metadata group. A transactional route owns two Jobs because native bindings
are frozen; both are counted, while a nontransactional route retains only the
primary. Every Program with internal memory receives one `JobWorkspace` route.
All occurrences of one recurrence reuse that route, and primary/alternate
transactional Jobs also reference it rather than allocating another internal
payload.

Nonzero Program arenas are shared across distinct Pipeline steps by one cold
canonical plan. One raw-U32 Pipeline arena owns the physical storage; typed
width and Fixed policy remain exclusively in the value route. Within each Program,
the Program compiler seals one immutable chunk permutation ordered by
descending U32 count and then local ordinal. Pipeline planning and workspace
materialization consume that same permutation for every occurrence and backend;
neither pass allocates or sorts a second chunk order.

The Pipeline planner visits steps by descending total chunk words, with logical
step, iteration, and physical step index as the canonical tie-break. Within a
step it consumes the Program permutation once. For each ordinary chunk, every
existing owner is tested at the next 256-byte-aligned offset without crossing
the ordinary `min(1 GiB, backend storage limit)` ceiling; for a larger admitted
chunk, only an unused offset-zero owner is eligible. The selected candidate
minimizes the checked tuple

```text
(owner growth, unused slack, owner ID)
```

If none fits, a new owner is appended. This single placement authority can pack
different chunk ranks into one owner when their steps are nonconcurrent; it is
not a parallel rank-envelope implementation, and there is no post-placement
coalescer. After the mapping is frozen,
`peak_step` and `peak_iteration` are recomputed from the aligned extents used by
each physical step. Arena creation is full-overwrite: every Program reset route
resets its exact assigned interval before that Program's node zero, while a
full-write chunk overwrites its assigned interval before any read. Distinct
reset routes remain distinct when they share the same Buffer owner, because
their half-open offsets are the semantic identity. No whole-arena
initialization pass is emitted and alignment padding is never counted as reset
traffic.
Consequently reset and full-write chunks share the same envelope without
carrying prior-step bytes into semantics. A visibility barrier precedes every
later step that can reuse a shared chunk. CPU
executes the same declaration frontier, Metal emits a buffer-scope barrier, and
Vulkan emits the matching compute visibility barrier. The plan depends only on
frozen chunk/rank order, never element count thresholds, timing, backend
model, or available memory.

The local scale contract declares eight simultaneous maximum ordinary owners
and evaluates this exact production planner without materializing their
payload. Small CPU, Metal, and Vulkan executions independently prove owner and
offset consumption. Because both checks call the same placement code and no
runtime scale selector exists, large-layout proof does not require an
eight-gigabyte local allocation.

The shared Buffer-owner vector and its physical payload are charged to
`shared_memory` exactly once. JobWorkspace routes remain in their owning step
metadata, but shared payload is excluded from every private Job partition.
CPU tile storage remains with the step. The native
prepared owner, including command and parameter/status storage, is measured
once using the common backend memory API. Caller Buffer
physical storage and shared Program owners are excluded because Pipeline
retains their lifetimes but does not own a second copy of their payload or
compiled artifact. A register-recurrence specialization is part of that native
owner: its dispatch-window, resident-binding, and descriptor-handle capacities
are Host metadata; Vulkan's retained parameter buffer is Staging and Metal's
retained parameter buffer follows its shared-memory Device category. Creation,
reuse, and destruction follow the backend's existing resource authority;
Vulkan performs the added descriptor-pool and parameter-buffer lifetime work
under its adapter lock, so the specialization cannot escape either memory
accounting or pool lifetime authority.

`memory_snapshot()` emits one Pipeline metadata group, one aggregate internal
group for private step storage when nonempty, one native prepared-owner group,
one 128-byte control staging group when that retained mapping exists, and the
existing explicit-read/checkpoint staging and traffic meters. Vulkan
checkpoint staging records cumulative `B` snapshot bytes or `2B` restore bytes
while each live chunk is bounded by `max(L, max(b_i))`; small operations that
fit `L` still use one contiguous range. Snapshot and restore both contribute
their exact allocation, reuse, reused-byte, command-submit, and semantic
transfer counters. Summary and rows use the same owner walk.
Observation allocates nothing; insufficient caller row capacity
sets `truncated()` without changing totals or retained state. Across successful
warm ticks every retained `current`, `peak`, and `budget` value remains stable
except documented frame, traffic, and generation counters; no caller Buffer or
Program owner appears in two rows.

Actionable telemetry reports cause and action for at least:

- resource claim conflict: name the canonical resource ordinal and suggest
  separating ownership or sequencing the authoritative tick;
- low submission amortization: report dispatches per native submit;
- unexpected warm allocation, compile, transfer, or payload readback: name the
  nonzero counter and the cold boundary that must own it;
- barrier pressure: report conflicting step boundaries, not an inferred domain
  label;
- poisoned output: require checkpoint restore or owner replacement and never
  suggest retrying on CPU.

## Performance Model

For `N` Programs, total useful device work `K`, materialized intermediate
traffic `M`, one native submit-and-completion cost `S`, per-Program separate
host encode cost `E_i`, retained Pipeline execution cost `E_p`, resource-claim
cost `C_r`, cross-step barrier cost `B`, and device status reset/reduction plus
ordered telemetry accumulation plus 128-byte observation cost `C_s`:

```text
T_separate = N*S + sum(E_i) + K + M
T_pipeline = S + E_p + C_r + B + C_s + K + M

gain = (N - 1)*S + sum(E_i) - E_p - C_r - B - C_s
```

Pipeline is structurally faster exactly when `gain > 0`. The upper speedup is
`N` only when submission dominates and useful device work tends to zero. It
does not claim to remove `M`; only cross-Program fusion could remove a
materialized intermediate, and Pipeline deliberately does not create that
second compiler authority.

`stats().pipeline.claim_ns` observes `C_r`, and `control_ns` observes the host
and backend control portion of `C_s`; they are saturating diagnostic clocks and
may overlap backend timing scopes, so their sum is not a wall-time authority.
The Release measurement compares total wall medians and reports these
components without subtracting them into a synthetic total.

Warm runD command construction is independent of dispatch count on the target
GPU paths: Metal executes one full retained ICB range, and Vulkan submits one
retained primary command buffer. Metal makes one bulk API call over its `R`
frozen unique resources; the driver may process `Theta(R)` residency state,
but runD visits zero command, binding, range, indirect-grid, and recurrence
state rows. Vulkan's equivalent commands are already recorded. Metal pays
device issue and uniform-guard cost for its frozen physical commands, including
inactive ones; performance measurements retain that cost. If the status arena
contains `Q` U32 entries,
canonicalization and deterministic reduction perform `Theta(Q)` device work.
Cold dependency projection is `Theta(B + D)` after exact hazard analysis rather
than a dependency-by-barrier nested search, and transactional projection is
`Theta(A + R + P)` rather than pair-by-binding/resource searches. The cold plan
also retains the exact active-step count. GPU submission and
completion consume that scalar and aggregate control evidence directly; they
do not rescan `N` private Jobs, reacquire `N` Job gates, or republish invisible
per-Job terminal state. CPU execution advances the same private owners under
the Pipeline phase authority without a second Job lifecycle.
Metal resets only its compact private raw prefix and imports each public raw
range once; it owns no canonical arena or raw-to-canonical copy. Host
observation remains exactly 128 bytes. The hazard plan and fingerprint are
cold-only. Metal's cold status layout uses one binding pass, one private/public
raw-offset partition pass, and one metadata rewrite pass, `Theta(B + Q)` for
`B` bindings and `Q` observed entries. Measurements must separate wall,
submit/completion, kernel, claim, status-control, and explicit read time.
Throughput claims require identical Programs, Buffer contents, step order,
numeric policy, and source manifest in serial/Pipeline ABBA pairs.

Vulkan device admission snapshots both 2D compute limits once. View,
publication, and bounded-window preparation share the exact `ceil(N / W)` to
`(x, y)` projection documented by the accelerator resource contract. This
removes two repeated physical-device property queries and moves View grid
division and limit validation out of warm encode. The retained grid and the
shader's row-major workgroup linearization preserve logical index order; this
is a host-overhead and capacity improvement, not an algorithm or
workload-dependent execution branch.

## Rejected Alternatives

- One monolithic Flow: valid for compiler-visible fusion, but it removes the
  requested modular Program boundary and is not a Pipeline implementation.
- Dependent Batch mode: breaks Batch's disjoint-storage and independent-result
  contract and creates two Batch meanings.
- Explicit DAG: creates another order authority and requires stable
  topological scheduling that the use case does not need.
- Buffer-address fingerprint: makes equivalent reconstruction and replay
  identity depend on allocation.
- Per-step submit: preserves modularity but leaves the structural bottleneck.
- Metal direct-encode fallback: retains two warm execution authorities and
  makes capability determine performance meaning.
- Hidden host pack/unpack: violates resident state and payload round-trip law.
- Implicit CPU retry: changes backend, timing, failure, and numeric evidence.
- Result memoization: Buffer contents and external state are not proven by the
  Pipeline fingerprint.

## Physical Ownership Map

The implementation follows repository naming rules and keeps one-word leaves.
The active ownership map is:

```text
node/include/rund/compute/pipeline.hpp
node/include/rund/compute/pipeline/bind.hpp
node/include/rund/compute/pipeline/profile.hpp

node/src/compute/pipeline/state.hpp
node/src/compute/pipeline/build.cpp
node/src/compute/pipeline/plan.cpp
node/src/compute/pipeline/claim.cpp
node/src/compute/pipeline/run.cpp
node/src/compute/pipeline/read.cpp
node/src/compute/pipeline/async.cpp

node/src/accel/kernel/status.hpp
node/src/accel/kernel/prepared.hpp
node/src/accel/kernel/prepared/model.hpp
node/src/accel/kernel/prepared/run.cpp
node/src/accel/kernel/prepared/batch.cpp
node/src/accel/kernel/prepared/pipeline.cpp
node/src/accel/kernel/prepared/completion.cpp
node/src/accel/kernel/prepared/evidence.cpp
node/src/accel/kernel/recurrence.hpp
node/src/accel/kernel/recurrence/
node/src/accel/metal/kernel/pipeline/
node/src/accel/vulkan/kernel/pipeline/

node/tests/contract/compute/pipeline.cpp
node/tests/contract/compute/pipeline/
node/tests/contract/runtime/product/compute/pipeline.cpp
node/tests/contract/runtime/product/compute/pipeline/
package/tests/consumer/example/pipeline.cpp
tools/measure/compute/pipeline/profile.cpp
tools/measure/compute/pipeline/recurrence.cpp
tools/measure/compute/pipeline/run.cpp
```

The common prepared-entry and status owners are shared by Pipeline backend
execution rather than mirrored below Metal and Vulkan. Pipeline may not call
the public Batch executor, use the Program convenience cache, or retain adapter
mirrors.

## Verification

Use the narrowest exact owner first, then widen without changing source bytes:

```sh
tools/test/run compute.pipeline --backend cpu
tools/test/run compute.pipeline --backend metal
tools/test/run compute.pipeline --backend vulkan
tools/test/run runtime.compute-pipeline
tools/test/run runtime.compute-pipeline-accel
tools/measure/compute/run --pipeline metal
tools/measure/compute/run --pipeline vulkan
tools/measure/compute/run --recurrence metal
tools/measure/compute/run --recurrence vulkan
tools/measure/compute/run
tools/measure/build/run
tools/check/run
tools/sanitize/run address
tools/sanitize/run thread
tools/release/run
tools/evidence/status
```

The canonical `pipeline.cpp` test owner retains only case registration and
backend parity accumulation. One-word leaves below `pipeline/` own one semantic
contract each; `local.hpp` owns shared declarations and template adapters, and
`oracle.cpp` owns the compiled memory/profile oracle. No leaf registers a case,
depends on another leaf's execution, or owns a second backend loop. The case
name and result offsets remain stable while a leaf edit invalidates only its
semantic contract object.

The CPU Session integration case keeps its one registry symbol in
`runtime/product/compute/pipeline.cpp`. The compiled `pipeline/local/model`
owner alone constructs the CPU Device and Session, owns the common exact-read
oracle, preserves diagnostic labels and result offsets, and runs the semantic
leaves in canonical order. `lifecycle`, `claim`, `cancel`, `status`, `view`,
`state`, and `nested` create all operation-local Programs, Pipelines, Buffers,
and result storage themselves; no result depends on translation-unit
initialization or an earlier leaf's mutable fixture. Editing one semantic leaf
therefore compiles that leaf and relinks the existing
`runtime.compute-pipeline` runner instead of compiling the complete integration
contract.

The three selected `compute.pipeline` routes own standalone type, binding,
order, hazard, fingerprint, claim, poison, read/hash, memory, status, counter,
CPU, Metal, and Vulkan semantics without opening an unselected accelerator.
`runtime.compute-pipeline` owns CPU Request/Submission/Poll/await,
cancel, wait, and close. `runtime.compute-pipeline-accel` owns the same Session
lifecycle over the selected native GPU paths. The release route installs the
focused header and executes the black-box Pipeline consumer through
`runD::sdk`. The Compute measurement owns identical serial/Pipeline ABBA work;
the build measurement owns focused include bytes, cold/warm syntax time, and
unrelated-consumer fan-out.

One frozen source manifest must prove every requirement below before a release
can claim the Pipeline contract:

1. installed SDK compilation for multi-input, Bounded, Record, multi-output,
   and repeated logical-output aliases;
2. a two-stage, at least three-input `Fixed<I,F>` 64-bit Pipeline whose second
   Program observes the first Program's resident write;
3. same-width different `(I,F)`, rounding, overflow, and approximation
   rejection before backend preparation;
4. exact policy, shape, Device, logical projection, same-step alias, capacity,
   moved-owner, and empty-owner typed reasons;
5. cross-step RAW, WAR, WAW, and read/read laws with adversarial stale-cache
   fixtures;
6. canonical fingerprint equality across allocation addresses and creation
   order, plus inequality for changed order, alias topology, Program, extent,
   role, and numeric policy;
7. all-or-none resource claims, shared-read admission, writer conflict,
   concurrent independent Pipelines, cancellation, and Session close;
8. pre-submit failure with zero writes/submits; post-submit ordinary-write
   poison; and transactional cancellation, semantic failure, backend failure,
   and device loss with unchanged parity/generation and one pending discard;
   live-Device failures retain readable/snapshot-eligible published state,
   while DeviceLost rejects resident I/O and a pre-loss snapshot restores on a
   newly opened Device;
9. CPU exact verified prefix and GPU known/unknown failure-step evidence without
   a success-shaped output;
10. repeated warm ticks with zero SDK allocation growth, compile, descriptor
    growth, upload, payload download, implicit payload readback, and round trip;
11. GPU terminal observation is exactly 128 bytes, follows canonical Reason and
    failing-ordinal selection, appends eight device-generated bounded-control
    U64 fields, three nested failure coordinates, and four nested work-total
    U64 fields, and is reported only as control evidence;
12. on Metal/Vulkan, `dispatches == final_dispatches == authored final
    dispatches + required device View gather/scatter dispatches` and
    `original_dispatches` remains the authored plan; control commands are
    separately counted, CPU submits zero, nonempty Metal commits one, and
    nonempty Vulkan calls the queue submit API once before any explicit read;
13. a resident-window Pipeline with device-resident `C = 0` submits only its
    resident preflight and publishes empty output;
14. CPU, Metal, native Vulkan, and MoltenVK translated-path value, reason,
    fingerprint, verified-prefix, and explicit output-hash parity;
15. Metal reusable indirect-command execution and Vulkan immutable-primary
    resubmission are measured against identical separate Program runs using the
    repository performance method;
16. Program cache identity, serialized convenience execution, resident Job,
    independent Batch, and their existing counters remain unchanged;
17. Standalone, Session await/poll/wait/cancel/close, installed package, Release,
    ASan/UBSan, TSan, and source-manifest closure all pass.
18. Pipeline retained memory snapshots are exact, warm-stable, allocation-free,
    and do not count caller Buffers or Program owners twice;
19. focused-header closure measures local include count, bytes, warm and cold
    syntax time, and unrelated Compute consumer rebuild fan-out.
20. state success alone flips every pair once, increments one generation,
    reuses the prebuilt primary/alternate projections without warm allocation,
    and publishes no partial pair set; four consecutive native ticks prove the
    stride-two primary/alternate generation sequence, while a parity-one
    semantic failure followed by a valid retry proves exact rebase; state-only
    and state-plus-ordinary-output reads close one canonical aggregate hash in
    either parity and match CPU, Metal, and Vulkan bit for bit;
21. a copyable snapshot restores both parity Buffers in a replacement Pipeline
    on a newly opened Device, rejects fingerprint/schema mismatch before any
    publication, reports exact snapshot/restore bytes and hash, and continues
    correctly from both odd and even saved generations;
22. contiguous and strided Views execute from their authored offsets/strides,
    disjoint interleaves emit no false dependency/barrier, true overlap is
    ordered identically on CPU, Metal, and Vulkan, dense-only Reduce/Sort/Scan
    and multi-input matrix paths retain no host fallback, warm memory is stable,
    and a reused Buffer's untouched partial-write lanes remain zero.
23. opt-in step profiling preserves frozen ordinals, Program fingerprints,
    outputs, state/snapshot hashes, generation, backend, one native submission,
    and success/failure identity; uses caller-provided bounded storage without
    warm allocation; exposes truthful CPU/Metal/Vulkan timing availability and
    exact direct-work evidence; and reconciles disjoint step plus shared memory
    with the complete Pipeline owner. Every accepted attempt first invalidates
    rows from an earlier accepted attempt. Known failure keeps only
    generation-bound rows, cancellation and
    unknown/device-loss completion expose only the truthful prefix, and a
    rejected poisoned rerun preserves the latest accepted attempt rather than
    manufacturing another one. The direct installed-SDK consumer proves the
    public success, known-failure, identity, topology, memory, and backend
    surface; backend fault contracts inject completion loss and device loss.
24. balanced ABBA measurement reports enabled instrumentation and later
    `profile()` observation cost separately from the disabled Pipeline wall
    time without subtracting either into synthetic useful work.
25. two distinct multi-pass Programs bind disjoint caller Buffers while their
    nonzero internal arenas share one maximum-envelope Buffer; the injected
    visibility frontier is fingerprinted, CPU/Metal/Vulkan outputs are exact,
    reset and full-write chunks share that envelope, and step plus shared
    memory reconciles to the complete Pipeline owner. Program compilation
    retains exactly one canonical chunk-rank permutation; a 1,024-occurrence
    recurrence reuses that permutation and one workspace without an
    occurrence-local order allocation or sort.
26. explicit `windows<Max, Tile>` freezes `K = ceil(Max / Tile)` ordered
    occurrences; counts `0`, `Tile - 1`, `Tile`, `Max`, and `Max + 1` prove
    ordered coverage and structured overflow, while CPU/Metal/Vulkan bits
    match, every occurrence reuses one Program workspace, native GPU work
    remains one submission, and warm execution performs no count readback or
    allocation. `PipelineBuilder::plan()` and the prepared Pipeline report the
    same `PipelinePlan`, and a budget below `peak_bytes` fails before Pipeline
    allocation.
27. `tile_repeat<N>` proves its Seed/Action/Fold type laws and canonical
    outer-window/inner-iteration order for `N = 1`, even and odd `N`, and more
    than one window. Counts `0`, `1`, `Tile - 1`, `Tile`, `Tile + 1`, `Max`,
    and `Max + 1`, high sparse ordinals, the last ordinal alone, and canonical
    producer conflicts exercise empty, tail, full, and overflow behavior.
    Initial and Fold-produced terminal values prove early stop. Injected Seed,
    first/middle/last Action, and Fold failures prove canonical first-failure
    attribution; a later higher-priority failure after an otherwise
    publishable Fold result still suppresses publication. A telemetry-bearing
    Action compares one active tile in a three-window terminal schedule with
    the same tile in a one-window schedule, proving stopped occurrences cannot
    re-accumulate stale route counters. Two nested logical steps with different
    `K` and `N`, zero and partial counts prove that outer/inner work totals are
    attempt-wide saturating sums rather than the final step's values on
    standalone CPU/GPU and CPU Session execution. CPU, Metal, Vulkan, and
    native Windows Vulkan where available produce matching results, failure
    coordinates, and outer/inner work totals against a serial oracle;
    each GPU attempt submits once, warm counters report zero compilation,
    allocation, descriptor growth, binding mutation, count readback, and
    fallback, and an independent frozen-owner/View snapshot proves that the
    zero binding-mutation count is structural rather than an unwritten
    counter. `N = 1` and `N = 64` retain the same Action scratch allocation
    count and one Action fingerprint rather than graph expansion. A plan-only
    large `Max` proves route-template and retained-memory growth is `O(K + N)`,
    not `O(K * N)`, while the tile invariant bank, two carried banks, and
    maximum shared workspace/View/scratch envelope remain independent of `K`;
    a one-byte-short budget rejects before backend allocation. Flat-only,
    pure nested, and nested-plus-ordinary plans prove that `barrier_count`
    equals the prepared compact boundary vector and does not grow as
    `prepared_command_count - 1`.
28. one prepared Pipeline composes stable Compact output directly into
    `windows<Max, Tile>(tile_repeat<N>)`: zero, high-sparse, partial-tail, full,
    repeated warm, and overflow attempts prove exact ordinal/count lineage,
    one accelerator submission, zero intermediate readback or setup growth,
    and transactional suppression of queue/count/state publication on
    overflow across CPU, Metal, and Vulkan.
29. Metal Pipeline preparation records every control and payload command in
    one guarded ICB, resets recurrence selectors on-device, preserves the
    logical/failure guards of every formerly indirect private primitive, and
    warm execution performs one full-range ICB call with no command, range,
    binding, indirect-grid, or recurrence-state traversal. Focused semantic
    tests and an immediate-pre-edit/after ABBA measurement are both required;
    a one-submit counter alone is not hard-cut evidence.

An unavailable backend capability, typed rejection, partial implementation,
direct-encode fallback, skipped native Vulkan run, or documentation-only API is
not a passed requirement. Unrun verification is named as unrun; it is never
inferred from another backend or a passing build.

## Primary Platform Evidence

- Apple documents compute commands in reusable indirect command buffers,
  including retained pipeline state, Buffer arguments, dispatch, and command
  barriers: <https://developer.apple.com/documentation/metal/mtlindirectcomputecommand>
- Apple documents indirect command buffers as the reusable command store for
  avoiding repeated command encoding: <https://developer.apple.com/documentation/metal/indirect_command_encoding>
- Apple's Metal feature tables define the per-kernel Buffer argument-entry
  limit used by the reserved Pipeline guard ABI:
  <https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf>
- The Vulkan specification states that primary/secondary command-buffer
  boundaries add no synchronization and require explicit synchronization for
  memory visibility: <https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html>

## Update Rule

Implementation may change a contract choice only by updating its invariant,
cost model, failure law, and verification evidence together.
