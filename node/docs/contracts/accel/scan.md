# Accel Scan Contract

## Scope

Kernel owns scan identity, logical-count meaning, inclusive/exclusive sum,
stored width, numeric domain, and stable failure reasons. Node owns the native
CPU, Metal, and Vulkan execution of that one meaning. There is no backend
fallback and no alternate replay or portability algorithm.

The observable result contains every active prefix. For active count `A`, the
inclusive and exclusive results are respectively

```text
inclusive[i] = sum(input[j], 0 <= j <= i)
exclusive[i] = sum(input[j], 0 <= j < i)
```

for every `0 <= i < A`. The admitted capacity may exceed `A`, but its inactive
tail is not semantic output. Because all `A` prefix values are observable,
generic scan must materialize them; an implementation that retains only a
selected subset is not this primitive.

Resident and scratch admission use the single Kernel checked-arithmetic
authority before multiplying a count by an element width. Metal and Vulkan
therefore reject the same overflowing shape before allocation or address
calculation; a backend-local copy of that arithmetic is not an execution law.

## Vulkan hierarchy

Vulkan uses one fixed 128-lane contiguous-chunk hierarchy on every driver. The
width is the largest workgroup width guaranteed by the Vulkan Core device
limits, so shader identity and arithmetic never depend on a vendor, device,
subgroup width, or driver string. One workgroup owns one logical block, each
lane owns one contiguous chunk, and a seven-level shared tree prefixes the 128
lane totals. Every lane then emits its chunk from the frozen exclusive lane
offset.

For more than one logical block, one prepared command stream contains:

1. `block`: materialize every block-local prefix and one block total;
2. `prefix`: compute exclusive offsets over block totals;
3. `offset`: add the owning block offset to every materialized prefix.

For one logical block, only `block` is needed. No invocation loops over all `A`
outputs or all `B` block totals, and no workgroup waits for another workgroup
to be scheduled. Progress depends only on dispatch order and explicit Vulkan
barriers between stages.

For block size `N`, the lane span is `C = ceil(N / 128)`. Its dependent
addition depth is bounded by `2C + log2(128)` when block totals and output
prefixes are both materialized. At the product block size `N = 256`, the bound
is `11` dependent additions. The in-place block tree uses one publication
barrier and two barriers for each of its seven frozen read/write levels, for a
fixed total of 15. Modulo-width addition is associative, and contiguous lane
ownership fixes the tree leaves, so this parallelization changes neither
output bits nor order.

The block-total prefix uses the same partition once. For `B` totals and
`C_B = ceil(B / 128)`, lane `l` owns the contiguous half-open range
`[l C_B, min((l + 1) C_B, B))`. It first emits that range's local exclusive
prefix and one chunk total. A two-bank seven-level tree computes the 128 frozen
chunk offsets, then each lane adds its one offset to its range. Per-lane work
is at most `2 C_B + 7` additions and the longest value dependency is at most
`C_B + 8`; no invocation owns all `B` values. The workgroup executes exactly
eight shared-memory barriers, independent of `B`: one after publishing chunk
totals and one after each tree level. Empty chunks contribute zero and do not
create a second tail algorithm.

Block and offset workgroups are split at the selected device's immutable
`maxComputeWorkGroupCount[0]`. Each chunk carries one 64-bit base block in a
declared compute push-constant range; the shader keeps address arithmetic at
64 bits and converts only the buffer index already proved to fit the admitted
32-bit storage ABI. Both modes own one logical block per workgroup, so
workgroup count is `G = B` and command chunk count is
`ceil(G / maxComputeWorkGroupCount[0])`, without truncation, retry, or a second
execution path.

Generic and segmented scan, and radix sort, obtain this chunk count from the
single internal Vulkan collective authority in `collective/chunk.hpp`. For a
nonzero workgroup count `G` and device limit `L`, its integer form
`1 + (G - 1) / L` avoids addition overflow and rejects any derived
physical-dispatch count that would overflow 64-bit telemetry.

Let `b(i)` be the logical block containing `i`, `L(i)` the modulo-width local
prefix, and `T(k)` the modulo-width total of block `k`. The final value is

```text
prefix(i) = sum(T(k), 0 <= k < b(i)) + L(i)  (mod 2^W)
```

where `W` is 32 or 64. Integer addition modulo `2^W` is associative, so the
fixed tree produces the same stored bits as canonical scalar evaluation.
Schedule, workgroup arrival, driver identity, and subgroup width cannot change
the result.

Overflow remains fail-closed and follows canonical ascending transitions, not
block-local translations. A one-block scan checks transitions in `block`. A
multi-block scan uses `block` and `prefix` only to construct modulo prefixes;
`offset` then combines the final global prefix with the original input and
atomically ORs each chunk's actual canonical transitions into one reset status
word. This distinction is required for signed arithmetic because
overflow is not translation invariant: a carry may cancel a local overflow or
create one that local evaluation from zero cannot see. Status OR is
associative and commutative; status is diagnostic evidence and never a second
result accumulator. The success path performs no status atomic; only a
detected error contributes to the OR. Device scratch, reset traffic, mapped
readback, and host completion work are therefore all constant: `4 bytes`,
`4 bytes`, one word, and `O(1)` respectively, rather than `4 B bytes` and an
`O(B)` host fold. Signed domains use the sign-bit transition law and
unsigned domains use `sum < lhs`. `I32`, `U32`, `I64`, `U64`, and both Fixed
storage widths therefore retain their exact domain-specific reason.

Inclusive/exclusive mode is one frozen parameter in the 32-byte Vulkan scan
parameter ABI. Block output remains operation-specialized, while prefix and
offset each have one source and one pipeline identity. There is no runtime or
driver-selected algorithm choice.

Metal uses the same fixed 128-lane contiguous-chunk hierarchy and three-stage
proof for every 32- and 64-bit domain. One workgroup materializes each logical
block, one workgroup prefixes block totals, and `offset` runs blocks in
parallel. Values combine modulo `2^W`, so reassociation changes only the work
partition. A one-block scan checks each canonical `(previous, value, next)` in
`block`. A multi-block scan gives neither `block` nor `prefix` overflow
authority; `offset` is the only canonical checker. Only block zero contributes
an invalid logical-count bit. There is no element-count or device-selected
width, serial lane-zero domain branch, status binding in `prefix`, retry, or
fallback.

The 32-bit block and block-total prefix use a two-bank Kogge-Stone tree. For a
product power-of-two width `W`, it has fixed depth `log2(W)`, exactly
`W log2(W) - W + 1` modulo additions, and `1 + log2(W)` threadgroup barriers.
Every stage reads one frozen bank and writes the other, so workgroup scheduling
cannot expose a partially updated level. The extra bank is `4W` bytes. The
64-bit path uses its fixed Blelloch tree. Comparative latency belongs to the
installed Release measurement route, not to source topology.

For a multi-block Metal scan, `block` materializes local-inclusive values
`L(i)`, including for a requested exclusive result. `offset` reads the original
value once and reconstructs

```text
next(i)     = block_offset + L(i)  (mod 2^W)
previous(i) = next(i) - value(i)   (mod 2^W)
```

where `W` is the stored width. Vulkan's block output is already in the requested
local inclusive or exclusive form; its offset uses that form and the same
original value to construct the identical `(previous, value, next)` triple.
The extra input read is contiguous and requires no workgroup barrier, shared
array, subgroup assumption, or neighbor-output read. In particular, no lane can
observe a prefix that another lane has already overwritten, so scheduling does
not participate in overflow authority.

## Traffic model

Let `A` be the active element count, `B` the block count, and `E` the stored
element width in bytes. Counts below are logical shader loads and stores, not
claims about cache-line transactions.

For a multi-block Metal or Vulkan scan of either stored width, the dominant
main-array traffic is

```text
block:  input read A E + local-output write A E
offset: local-output read A E + input read A E + final-output write A E
total:  5 A E
```

At `A = 262144` and `E = 4`, this bound is 5 MiB. One-block execution skips
prefix and offset and uses `2 A E`. Scratch, parameters, and block-total traffic
add `O(B E)` bytes; the single status word adds `O(1)` bytes. These terms are
stated separately rather than hidden in the dominant term. Reducing the main
array term would require carrying canonical values across stages without a
portable global barrier or fusing the scan consumer; it is not obtained merely
by replacing the input read with schedule-sensitive adjacent-output reads.

Vulkan's block-total prefix reads and writes each of its `B` totals once while
forming chunk-local prefixes, then reads and writes them once while adding the
frozen chunk offset: `4 B E` logical bytes plus `2 * 32 * E` shared storage.
It changes no `A`-sized input or output traffic. Its benefit is the bounded
eight-barrier hierarchy proved above, which must be judged by installed-Release
measurement rather than inferred from source shape.

Any full scan has the information-theoretic lower bound `2 A E`: every input
must be observed and every public prefix must be materialized. For a sustained
device-memory bandwidth `M`, even an otherwise free implementation therefore
obeys `time >= 2 A E / M`. Dispatch, synchronization, arithmetic, scratch, and
status can only increase that lower bound. A requested multiplier whose time
target is below it is physically impossible for the same materializing scan;
achieving it requires changing the workload boundary, for example by fusing a
consumer so public prefixes remain inside one execution boundary.

## Segmented carry hierarchy

Segmented scan uses the same canonical transition law with reset points. CPU
evaluates the segment directly. Metal and Vulkan keep three deterministic
stages without a second result authority:

1. `block` emits modulo-width local prefixes and checks only subsegments that
   begin at a head inside the block;
2. `prefix` freezes the carry entering every block;
3. `offset` starts from that carry, revisits the leading subsegment in ascending
   order, emits the final inclusive or exclusive values, and atomically raises
   the shared status for any actual transition overflow.

The leading pass is necessary for signed and Fixed domains. Overflow is not
translation invariant: evaluating a later block from zero can falsely reject
when a negative carry cancels its local sum, and checking only the final block
total can falsely accept when `MAX + 1 - 1` crosses the bound and returns.
Heads partition the proof: every transition before the first local head belongs
to `offset`; every transition at or after that head belongs to `block`.
Therefore each canonical transition has one checker and all backends preserve
`compute_segmented_scan_sum_overflow` for both inclusive and exclusive modes.

For `B = ceil(capacity / logical_block_size)` and element width `E`, Vulkan's
native internal storage is `B * E` bytes of totals plus `B * 4` bytes of
status, apart from fixed parameters and descriptor state. The user-visible
output remains `capacity * E` bytes because the public result requires it.
Kernel `ScanPlan::temp_bytes` remains portable admission evidence; backend
physical scratch and dispatch count are runtime evidence, not graph identity.

The adapter freezes `L = maxComputeWorkGroupCount[0]` without clamping. Device
admission rejects an impossible zero limit. Generic and segmented block and
offset stages use `G = B` workgroups and `C = ceil(G / L)` command chunks with
a frozen base-block push constant. The physical dispatch count is `C` for one
pass and `2C + 1` for three stages. Vulkan runtime telemetry records that
physical count rather than the device-neutral kernel stage count.

## Authority and verification

- `/node/src/accel/vulkan/scan`
- `/node/src/accel/metal/scan`
- `/node/src/accel/cpu/scan.cpp`
- `/node/tests/contract/accel/kernel/scan`
- `/node/tests/contract/compute/collective/modes.cpp`
- `/node/tests/contract/compute/bounded.cpp`

`accel.kernel-core` verifies exact direct output and the three-stage Vulkan
pipeline-cache shape. `compute.collective-modes` verifies inclusive/exclusive
output and overflow reasons over all six stored domains on CPU, Metal, and
Vulkan. `compute.bounded-parity` verifies resident logical counts, stable
filter-to-scan composition, reuse, and cross-backend hashes. Performance claims
require an installed-Release measurement under the repository performance
contract; source structure alone is not a speedup claim.
