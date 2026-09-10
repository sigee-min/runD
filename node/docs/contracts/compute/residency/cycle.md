# Residency Cycle

This page owns rolling bank order, prefetch concurrency, directional overlap
meaning, and the structural temporal-cycle model. It does not own backend
terminal implementation or performance results.

The production rolling-flight behavior is implemented by the stateless
`registry/cycle_owner.{hpp,cpp}` `CycleOwner`. It borrows Authority's one gate,
frame table, epoch slots, `CycleSlot`, and retired token; it owns no journal or
replacement state. `Authority::cycles()` is the only exposure of that owner.
The structural `cycle::Plan` remains separately owned by `cycle/plan.cpp`.

## Production Rolling Order

Direct non-Scan execution assigns bank `e mod 2`. CPU may prepare `e+1` in the
alternate authoritative Host bank while `e` computes. Accelerators submit the
alternate bank before the Host waits for the current native terminal; the two
bank terminals remain independent. Compute `e+1`, page-out `e`, and page-ready `e+2`
may overlap subject to Authority drain edges. Scan remains serialized by its
page-ordered carry and reruns the prepared Scan after prefix injection. CPU
does not bind a compute-flight journal and keeps its prior rolling order. The
public overlap entry chooses once between compile-time `UseCycle=true` and
`UseCycle=false` implementations; the CPU specialization contains no Flight
construction or bind/advance/complete/close branch.

For every accelerator non-reduction epoch, production builds one
`cycle::Flight` from the live execution lease: ordinal, parity bank, Authority
token, exact Input/Output frame ranges, active-frame masks, and write intent.
`bind_cycle` and `advance_cycle` authenticate those facts against the mutable
Authority journal. At most the two rolling banks are live. Reusing a bank at
`e+2` requires that bank's `e` flight to have reached its Authority terminal.
Once bound, ordinary `complete` is illegal: the epoch can terminate only
through `complete_cycle`. Native failure and retention failure share that one
terminal path; failure closes the journal and invalidates the named work. This
is production ordering authority, not a second frame-state owner.

On Metal the execution ranges are HostVisible Shared-buffer aliases. Output is
consumed through that coherent alias only after its Known native terminal, so
Direct coherent page-out has no D2H copy. For Direct non-Scan, non-reduction
work with equal frozen Host/Device frame capacities, the backing worker writes
the exact authenticated Shared Input view under the execution transform token;
there is no separate H2D copy. A denied view, H>K, Scan, reduction, and Graph
retain the ordinary Host-supply/private-upload path. Neither path is GPU DMA or
a native Upload node.

One Persistent backing may admit two parallel reads. The Pool's two
metadata-only prefetchers target the exact Authority Host input frames and may
prepare `e+1` and `e+2`; writes and terminal publication remain ordered.
An exact authenticated HostVisible Device-bank view can satisfy only the
bootstrap `e+1` supply before that alternate bank is submitted. A speculative
rolling `e+2` supply remains in the Host tier until its prior same-bank
dispatch terminates. Metal then promotes the exact Host receipt through its
authenticated Shared view without manufacturing an H2D submission. Vulkan
DeviceLocal owners expose no such view and therefore retain the physical
upload path. The common scheduler proves the capability from the concrete
prepared owner; the mere presence of a backend-wrapper function pointer is not
capability evidence. This prevents an `e/e+2` overwrite while preserving the
frozen prefetch distance and two-worker backing concurrency. A relative
prefetch distance is never treated as a bank-release receipt.

`h2d_overlap_ns` and `d2h_overlap_ns` are intersections of actual transfer
wall intervals with the exact native-submission-in-flight receipt.
`overlap_ns` is their saturating sum. A coherent input fill is backing/CPU work,
not H2D overlap. Backing callbacks are not transfer
overlap. Only unhidden transfer/backing time contributes to `stall_ns`; initial
page-ready and terminal page-out are fully stalled. These counters do not claim
exact GPU-kernel timestamps or speedup.

The bounded `cycle::Plan` remains a structural model and does not own product
recurrence or final publication. The production recurrent transaction is owned
by [Execution](./execution/README.md); there is no second product Plan in this folder.
