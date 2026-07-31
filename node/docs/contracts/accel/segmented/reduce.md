# Accel Segmented Reduce Contract

## Scope

Node executes kernel-planned segmented reduction on authenticated resident
buffers. CPU, Metal, and Vulkan implement the same `Sum`, `CountNonzero`,
`Min`, and `Max` meaning for 32-bit and 64-bit stored domains.

## Admission

The graph step has exactly three roles: read values, read `u32` segment heads,
and write output. Value and output widths match the declared element type; head
and value counts match; output capacity is at least the input count. The first
head is `1`, every later head is `0` or `1`, and a required unavailable backend
returns its precise adapter reason. No backend fallback exists.

## Deterministic Law

Segments are discovered in ascending input order and output slots follow that
order. CPU uses the kernel reference. Metal and Vulkan build a stable head-index
list once. One backend-neutral physical model owns the 256-element index block,
65,535-group grid bound, status bits, and structural-before-numeric reason
priority. Backend code consumes that model and owns only its API resources and
encoding.

Metal assigns every segment to one SIMD group inside a fixed 256-thread
workgroup. For hardware SIMD width `W`, each lane visits the ascending
`begin + lane + kW` subsequence and a SIMD-scoped binary tree combines the exact
partials. The pipeline's immutable execution width fixes `Q = 256 / W`
segments per workgroup during preparation. Vulkan uses one fixed 256-thread
workgroup containing four disjoint 64-lane teams. Team `q` owns segment
`group * 4 + q`, visits `begin + lane + 64k`, and combines the 64 exact
partials through the same six fixed binary-tree levels for every segment
length. Excess groups continue through one canonical grid stride.

Neither backend has a short/long classifier, serial-lane alternative,
length-selected kernel, or second reduction dispatch. Work partition cannot
change result bits because exact addition, Min, and Max are associative. Head
indexing uses fixed 256-element blocks: parallel classification writes one
count per block, one 256-lane fixed tree scans those block counts, and parallel
stable scatter writes each head index at

```text
head_slot(i) = block_offset(block(i)) + local_head_prefix(i).
```

Both prefix terms count only heads at smaller input indices, so the result is
independent of workgroup completion order. The block-count scan assigns each
lane one contiguous quotient/remainder partition, then combines the 256 exact
lane totals with fixed Blelloch tree edges. Its longest serial partition is
`ceil(ceil(n / 256) / 256)` block counts rather than `n` head flags. Index
scratch is two `ceil(n / 256)` count arrays plus one stable head-index array
with capacity `n`; no input-sized per-element offset array is materialized.
Physical work is `O(n + S log W)` on Metal and `O(n + S log 64)` on Vulkan for
`n` values and `S` segments; no output slot rescans an earlier prefix. For a
segment of length `m`, Metal's longest lane chain is
`ceil(m / W) + log2(W)`, while Vulkan's is `ceil(m / 64) + 6`. Thus the
contract rows of length 128 and 512 have Vulkan chains of 8 and 14 exact
combines without selecting different programs. Metal's tree uses disjoint
SIMD-group slices and SIMD-scoped barriers. Vulkan's four fixed teams share one
workgroup array and cross the same six workgroup barriers. Release measurement,
not this cost model, owns observed wall time. The fixed team deliberately
rejects a packed tiny-segment path: it optimizes bulk segment work and removes a
second execution authority, but very short segments can leave lanes idle.
Performance evidence must therefore publish the segment-length distribution
and may not generalize a bulk result to a tiny-segment workload.

Sum sign- or zero-extends each stored value into an exact 128-bit carrier.
CountNonzero first maps each value to exactly `0` or `1`. For admitted
`n <= 2^64 - 1`, unsigned totals are below `2^128`, signed subset magnitudes
are below `2^127`, and counts are below `2^64`. Every lane accumulation and
fixed-tree node is therefore exact, so grouping and completion order cannot
change the result. The final fit check publishes the canonical overflow
reason. Min and Max use their typed associative identities.

The carrier operator is two-word addition modulo `M = 2^128`. For all carrier
values `a`, `b`, and `c`,

```text
((a + b) mod M + c) mod M = (a + (b + c) mod M) mod M.
```

The admitted bounds prove that the final bit pattern is also the exact signed
or unsigned mathematical total, rather than an ambiguous wrapped value. SIMD
width, lane grouping, and command completion order therefore cannot become a
numeric authority.

Structural validity has higher status severity than sum or count overflow.
CPU uses the kernel-owned segment-head predicate and scans an unread head
suffix only after a numeric failure candidate; successful execution remains
single-pass. GPU status accumulation is order-independent and reason decoding
checks structural invalidity first. Mixed malformed-head and earlier-overflow
input therefore returns `compute_segmented_reduce_segment_invalid` on every
backend and stored domain.

## Resources and Evidence

Metal and Vulkan preparation owns head indices, count, status, parameters,
one indirect dispatch triple, and backend scratch. Each backend owns exactly
one reduction pipeline and one reduction descriptor for all segment lengths.
Metal's indirect workgroup count is `ceil(S / Q)`; Vulkan's is `ceil(S / 4)`.
Both are capped by the fixed command-grid limit, and excess segments continue
by the same group-count stride. Vulkan uses one compiled pipeline, one
descriptor set, one physical dispatch, and one compact indirect argument
payload; it adds no allocation, copy, readback, or materialized workspace.
Warm execution reuses prepared storage, resident execution performs no
implicit host readback, and original/final kernel-plan evidence remains
separate from physical backend dispatch and submit counts.

Private declarations are split by backend under
`node/src/accel/segmented/reduce/{metal,vulkan}.hpp`. There is no combined GPU
header in the CPU execution closure: a backend implementation
includes only its own declaration owner, while CPU consumes the kernel
reference directly.

## Verification

Accel contracts feed negative stored values through the declared Fixed domain
to prove signed reference parity on CPU, Metal, and Vulkan. Metal and Vulkan
each assert one submit with four physical dispatches. Standalone Compute
contracts compare CPU, Metal, and Vulkan across
signed, unsigned, and fixed storage widths, including empty, cancellation, and
overflow rows. One 512-element compiled program exercises 128-element segments
and a 512-element segment through the same fixed tree. Mixed malformed-head
rows prove structural priority in both native GPU execution paths and parity
on every backend without paying for a second graph compile.
