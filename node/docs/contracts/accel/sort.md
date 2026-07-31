# Accel Sort Contract

## Scope

Kernel owns the stable ordering law, radix shape, stored numeric domain, and
reference result. Node lowers that one meaning independently to CPU, Metal, and
Vulkan. There is no portability backend, adapter, runtime algorithm
selector, or CPU fallback.

Metal and Vulkan derive radix-table byte extents through the single
Kernel checked-arithmetic authority. A block-count or byte multiplication
that cannot fit the admitted 64-bit storage extent is rejected before resource
allocation; backend-local overflow formulas are not separate sort semantics.

## Native radix law

Metal and Vulkan use backend-owned radix-8 pipelines. For pass `d`, the input
to that pass is the output order of pass `d - 1`; a histogram from the original
block partition is therefore never reused after elements move. Each pass has
one current-order histogram, one deterministic prefix over block counts, one
256-bucket base prefix, and one stable scatter.

For key `x` in block `t` with digit `b`, its destination is

```text
G(b) + P(b, t) + R(x)
```

where `G(b)` is the number of keys in lower buckets, `P(b, t)` is the number
of bucket-`b` keys in earlier current-order blocks, and `R(x)` is the number of
earlier lanes in the same block with digit `b`. The three terms are disjoint
and their sum is below the admitted logical count, so destinations are unique
and in range. `P` and `R` preserve source order for equal digits; induction over
least-significant-digit passes proves stable full-key order.

An upfront original-block histogram for every digit is forbidden. After the
first pass, block membership can change, so such a table cannot represent
`P(b, t)` for a later pass. The cross-block stability contract contains this
counterexample explicitly and compares both keys and original ordinal values
against the CPU stable reference.

## Architecture

Metal uses 2,048-element blocks: 256 threads process eight fixed rounds, and a
SIMD/threadgroup prefix handles each bucket. One block table and one histogram
publication serve all eight coalesced rounds without changing the per-element
radix law. The power-of-two block shape also keeps block division exact at the
device boundary.
Vulkan uses 256-thread workgroups that process 512-element tiles in two
coalesced rounds. Classify uses one shared 256-bucket histogram. Scatter divides
the tile into 64 fixed 8-element rank groups. For local position `i` and
bucket `b`, it computes

```text
R(i, b) = sum(count(b, group), group < floor(i / 8))
        + count(j < i and floor(j / 8) = floor(i / 8) and bucket(j) = b)
```

Each group count occupies one nibble and eight adjacent groups share one 32-bit
atomic word. A group contributes at most 8, so no nibble can carry into its
neighbor; atomic integer addition therefore represents eight independent exact
counts regardless of invocation order. Each item compares at most 7 earlier
digits and folds at most eight packed words. The maximum local comparison work
per 512-element tile is `64 * sum(0..7) = 1,792`: 53.33% below the preceding
16-element grouping and 88.89% below the original 64-element grouping. The
number of atomic additions, packed words, barriers, and the address range owned
by each packed word stay unchanged.

The 256 lanes own their rank independently, so no lane serializes the whole
tile. Every prefix owner, nibble field, and rank group is fixed, making the
destination independent of invocation schedule and subgroup width. The scatter
shader uses 2 KiB for tile buckets and 8 KiB for packed counts. The latter
storage is reused for bucket and block bases after ranking, so the fixed
shared-memory requirement remains 10 KiB, below Vulkan's 16 KiB minimum.

The Vulkan adapter admits devices that support at least 256 invocations in the
first workgroup dimension. This is a native backend requirement shared by the
radix and collective kernels, not a runtime algorithm selector.

Both backends reuse one block-count table, one block-offset table, and one
bucket table across passes. For Vulkan, `B = ceil(N / 512)` and the two radix
tables occupy `(2 * B * 256 + 256) * 4` bytes. Scratch is therefore proportional
to one pass rather than multiplied by the number of key bytes.

The implementation deliberately does not use cross-workgroup spin waiting.
Neither Metal nor core Vulkan promises cross-workgroup forward progress, so a
lookback algorithm that requires an earlier workgroup to run could deadlock.
The native staged form retains the memory-traffic and stable-prefix structure
of high-throughput radix sorting while making progress depend only on command
order and explicit backend barriers.

[Onesweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus)
is the relevant lower-traffic reference: it couples digit binning, prefix, and
reorder through decoupled look-back. The portable rejection above is an
inference from that dependency and the absence of a cross-workgroup scheduling
progress guarantee in the
[core Vulkan execution model](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html).
runD does not hide a staged fallback behind device or workload detection. Its
portable implementation makes progress depend only on command order and
explicit backend barriers on both Metal and core Vulkan.

For U32 keys with U32 values, a materializing radix-8 pass must at least read
and write both arrays. Four digits therefore impose the payload lower bound

```text
bytes >= 4 passes * (4 + 4 byte read + 4 + 4 byte write) * N = 64N
time  >= 64N / sustainable_device_bandwidth
```

Histogram, prefix, barriers, submission, and host materialization add traffic
or latency beyond that bound. A claimed 100x improvement of the same operation
must satisfy this inequality; otherwise it describes a different boundary,
such as keeping data resident or amortizing many independent jobs, and must be
measured under that explicit boundary.

Warm execution submits one prepared command stream. Pipeline compilation,
buffer allocation, upload, and download are zero inside the warm loop. Physical
dispatch counts remain backend evidence and never enter graph identity.

Vulkan freezes `L = maxComputeWorkGroupCount[0]` at adapter admission. For
capacity block count `B`, preparation owns `C = ceil(B / L)` indirect command
triples. A one-invocation setup kernel derives every chunk's active group count
from the resident logical count, and classify/scatter encode all `C` chunks
with a push-constant base block. No indirect command exceeds the device limit;
bounded execution still reads the logical count only on device. Physical
dispatch telemetry counts every encoded chunk, prefix, and bucket-base command.
The chunk and physical-dispatch formulas are the same overflow-checked
Vulkan collective authority used by scan; sort does not mirror that arithmetic
in its resource allocator.

## Authority

- `/node/src/accel/metal/sort`
- `/node/src/accel/vulkan/sort`
- `/node/src/accel/cpu/sort.cpp`
- `/node/tests/contract/accel/kernel/collective/sort`
- `/tools/measure/compute`

`accel.kernel-core` owns small-domain, bounded-count, signed-order, U32/U64,
identity-value, and cross-block stability contracts. The installed Release
Compute measurement owns dense and bounded sparse throughput, output-hash
parity, one-submit evidence, and zero warm setup work.
