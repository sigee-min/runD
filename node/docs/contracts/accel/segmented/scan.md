# Accel Segmented Scan Contract

## Scope

Node lowers the kernel-owned inclusive or exclusive segmented prefix law to
CPU, Metal, and Vulkan over authenticated resident values, `u32` segment
heads, and output. The first head is `1`; every later head is `0` or `1`. There
is no backend fallback or alternate result law.

## Block law

Metal and Vulkan partition the input into fixed logical blocks. Each block
is executed by one fixed 256-lane workgroup. Lanes consume consecutive tiles
and apply the associative summary operator

```text
(a, h) combine (b, k) = (k ? b : a + b mod 2^W, h or k)
```

with a schedule-independent fixed tree. This computes stored-width modular
prefixes, the final tail, and the first segment-head position in
`O(ceil(block_size / 256) * log2(256))` workgroup depth instead of one serial
loop.
The tree changes grouping only for addition modulo `2^W`, which is associative;
it does not move overflow observation. Every lane reconstructs its original
immediately preceding prefix and checks the same canonical input transition.
All lanes freeze the current tile's `segment_carry` and `segment_seen` reads
before lane zero publishes the next tile state. A workgroup barrier separates
those reads from the publication and another separates publication from the
next tile. Without both edges, a legal backend schedule can evaluate some
lanes with the next tile's carry, producing a schedule-dependent extra block
sum even though the tree itself is fixed.

A segment that starts inside a block has zero carry, so its overflow can be
decided locally. A leading segment continued from an earlier block cannot be
judged against local zero: a negative incoming carry can make every canonical
signed prefix representable even when the zero-based local sum crosses the
signed limit.

The prefix stage therefore writes only the incoming stored-width carry for each
block. Its 256 lanes each own one contiguous block range, build a summary, scan
those summaries with the same fixed tree, and then replay only their own range.
For a block with a head, the next carry is its post-head tail; otherwise it is
the modular sum of incoming carry and block tail. The offset stage assigns the
leading elements to the same fixed lanes. It reconstructs the frozen local
predecessor from the block output, adds the incoming carry, writes the final
inclusive or exclusive prefix, and checks that exact canonical transition.
No lane consumes another lane's rewritten output. This catches cross-block
overflow, including exclusive final-total overflow, without rejecting signed
cancellation that remains representable.

For storage width `W`, block `b`, incoming carry `C(b)`, and leading values
`x(i)`, the offset recurrence is

```text
p(begin(b)) = C(b)
p(i + 1) = p(i) + x(i)  (mod 2^W)
```

and overflow is evaluated at every edge `(p(i), x(i), p(i + 1))`. Signed
domains use the sign-transition law; unsigned domains use `sum < lhs`. Status
combination uses maximum severity, so malformed heads dominate numeric
overflow and completion order cannot change the reason.

CPU uses the same kernel-owned segment-head predicate. Its successful path
remains one pass; only a numeric failure candidate scans the unread head suffix
before publishing a reason. This produces the same severity ordering without
charging valid CPU scans for an unconditional validation pass.

An aggregate `leading` scratch is not part of the algorithm: an aggregate
cannot prove every observable prefix and is therefore neither storage nor
authority. Preparation owns only one stored-width carry per block plus
first-head and status words. Parallelism adds no resident allocation and no
host or device copy; the fixed trees use workgroup-local storage only.
Status uses maximum severity. Successful lanes perform no status atomic;
only a lane that observes malformed input or overflow participates in the
workgroup or final-device maximum.

Private declarations are split by backend under
`node/src/accel/segmented/{metal,vulkan}.hpp`. There is no combined GPU header
in the CPU execution closure; CPU consumes the kernel reference directly.
`node/src/accel/segmented/model.hpp` alone owns the host parameter ABI consumed
by both native backends. Backend local headers include that owner directly and
do not redeclare or re-export it.

## Vulkan dispatch bound

The selected Vulkan adapter freezes `maxComputeWorkGroupCount[0]`. Block and
offset work are encoded as `ceil(B / L)` chunks for `B` logical blocks and
device limit `L`, with a push-constant base block. Prefix remains one dispatch.
The physical count is `C` for a one-pass plan and `2C + 1` for a three-stage
plan, where `C = ceil(B / L)`. Runtime telemetry records that physical count;
kernel plan identity remains device-neutral. This calculation is owned by the
same overflow-checked Vulkan collective chunk helper as generic scan and sort,
not a segmented mirror.

## Verification

`compute.collective-modes` compares CPU, Metal, and Vulkan for all six
stored domains. Its 512-element signed cancellation fixture places the
canceling terms on opposite sides of the 256-element block boundary, and its
cross-block overflow fixture keeps both local blocks representable while
requiring the offset transition to publish the canonical overflow reason.
The 64-bit `Fixed<I,F>` Vulkan row repeatedly rebuilds the resident job on one
cached Program so descriptor, scratch-buffer, and workgroup scheduling churn
cannot change any published bit pattern.
Inclusive and exclusive mixed-error rows place malformed input after an
already overflowing prefix and require structural rejection on every backend.
