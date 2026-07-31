# Accel Compact Contract

## Scope

Compact consumes U32 flags and emits the original U32 index of every nonzero
flag in ascending input order. Kernel owns descriptor identity, capacity,
stable result meaning, and rejection vocabulary. Node owns native execution.
There is no truncation, fallback, or partially published
result when the selected count exceeds output capacity.

## Metal and Vulkan rank law

Vulkan uses fixed 256-element blocks and three stages in one prepared command
stream: classify each block, prefix the block counts, and scatter selected
indices. For selected index `i`, its unique output rank is

```text
rank(i) = sum(block_count[k], k < block(i))
        + sum(selected[j], block_begin(i) <= j < i)
```

The first term has one deterministic block-prefix owner. The second is a fixed
workgroup prefix over source order. Their ranges are disjoint, so selected
indices in earlier blocks and earlier lanes always receive lower ranks. This
proves stable ascending output independently of invocation schedule.

Metal and Vulkan Compact do not invoke the public generic Scan graph.
Generic Scan exposes every prefix, while Compact exposes only selected indices
and the final selected count. These backends use the rank equation directly;
they do not create a second public prefix result or graph node.

For `B = ceil(N / 256)`, Vulkan scratch is exactly

```text
counts:  4 * (B + 1) bytes
offsets: 4 * B bytes
total:   8 * B + 4 bytes
```

`counts[B]` is the sole selected-count and capacity-status source. The finish
boundary compares it with the admitted capacity; a larger value returns
`compute_compact_capacity_insufficient`. Every run overwrites the block counts,
offsets, and tail total, so overflow, successful reuse, and later overflow
cannot observe stale status. The output is readable only after successful
finish.

The Kernel compact plan retains its backend-neutral scan/status byte admission
facts. Vulkan's smaller native scratch is execution evidence; it does not alter
descriptor hash, semantic output, or public capacity.

## Authority and verification

- `/node/src/accel/vulkan/compact`
- `/node/src/accel/metal/compact`
- `/node/src/accel/cpu/compact.cpp`
- `/node/tests/contract/accel/kernel/compact`
- `/node/tests/contract/compute/bounded.cpp`

`compute.bounded-parity` verifies stable output, exact/underfilled/default
capacity, empty output, downstream bounded composition, overflow rejection,
failed-read rejection, and successful reuse on CPU, Metal, and Vulkan.
Performance claims require an installed-Release measurement under the
repository performance contract.
