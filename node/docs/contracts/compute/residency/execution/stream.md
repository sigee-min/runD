# Recurrent Window Stream

This page owns the backend-neutral arbitrary-Q Direct pointwise and admitted
Window controller.
The immutable execution Plan remains the sole page, bank, predecessor, and
mutation formula. Fixed native command construction belongs to the backend
contract; this controller owns only chunk sequencing, Host-service joins, and
the one run Final.

Status: implemented as the coherent Direct pointwise and
serialized-Persistent Direct Window Q>4 fallback on Metal and Vulkan when
whole-Schedule cold preparation is unavailable or over budget.
The same Authority/Host-service owner is also reused by the preferred native
Schedule route. Product contracts execute Q5 and Q9; Q1..4 retain their
existing single-execution or bounded-window coordinators.

## Control

One public `run()` admits one `Q>=2` Plan and one Authority lease. The main run
thread authenticates the two initially live Input banks, hands the run once to
the controller, submits only the first native chunk, and waits on one run
Final. It neither loops epochs nor submits later chunks.

Native Releases execute exact Host Output(e) and Input(e+2) service on the
backend-owned cold residency callback lane; arbitrary-Q admission requires
that immutable backend capability and rejects an inline-callback adapter.
After all Releases and Host terminals in the current chunk are authenticated,
its Final enqueues the next fixed chunk on the Pool's accelerator-only cold
serial continuation lane. Thus even a synchronously delivered internal Final
cannot move later chunk submission onto the public run caller. Each
chunk has `first_epoch` and
`count=min(4, Q-first_epoch)`. A fixed slot is chunk-local, while every receipt
epoch and backend sequence is run-global. Re-numbering a later chunk to epoch
zero or submitting a short non-tail chunk is invalid.

Before the first chunk, one source-private stream sentinel claims and strongly
retains every distinct prepared owner reachable by both banks and both parity
generations, up to four. An intermediate raw Final clears only its bounded
`window` claim; it cannot release the stream sentinel. A peer submit and a
sentinel release while a raw window is active are Busy. The sole Known stream
Final detaches all owners for immediate reclaim. Unknown marks the sentinel
and every retained prepared submission quarantined before detaching the
caller-owned control. Release and raw submission serialize on the same sorted
prepared-submission mutex set, so neither can cross an ownerless launch gap.

The controller stores no Q-sized receipt, ticket, Pipeline, command, or page
array. It owns one two-bank Host-service journal, one four-entry circular
Release journal, and one current raw chunk control. Authority validates every
projected node against the existing Plan.

## Evidence

One public run produces `window_handoff_count=1`: it names the single accepted
controller Final, not an internal raw chunk. `window_batch_count=Q`. In chunk
fallback Metal reports Q queue calls and Vulkan reports `ceil(Q/4)`; an
admitted whole Schedule instead reports Q and one. These are backend-produced
facts, not values reconstructed from Q. Source-private Stream evidence records
the exact accepted lowering and, for fallback, `chunk_count=ceil(Q/4)` and
each raw chunk Final. Only the controller aggregates those internal facts into
public run evidence.

Known failure drains only the already accepted current chunk through its
no-write gates and submits no later chunk. Future unsent epochs are not
reported dispatched and receive no fabricated Release. Unknown terminal loss
quarantines the accepted chunk, both recurrent banks, and the whole Authority
lease and likewise submits no future chunk. Successful intermediate chunk
Finals are bank-release facts, never run publication. Only the controller
Final may close Authority and publish backing.

## Boundary

The chunk fallback removes main-thread epoch orchestration and keeps retained
Host state constant in Q. It is not a persistent GPU scheduler: Host-service
callbacks still perform backing work and issue one native handoff per chunk.
It places a rigid Final barrier between four-epoch chunks and does not submit a
smaller sliding successor as soon as one bank becomes safe. The preferred
Schedule removes those inter-chunk submissions, but Metal pays explicit O(Q)
native command storage and Q queue operations while Vulkan pays O(Q) transient
submit records. Graph, Scan, Reduce, noncoherent transfer service, persistent
device command generation, a sliding recurrent fallback, and a distinct
transfer-queue timeline DAG remain partial.

The exact native whole-run target and the different Vulkan/Metal lowering
constraints are owned by [Scheduler](./scheduler.md).

## Source ownership

`execution/stream/internal.hpp` defines the sole source-private `StreamWait`
controller and its coordinator/gate contract. `stream/service.cpp` owns Host
Input/Output service plus residency accounting and output hashing;
`stream/native.cpp` owns native signal/abort and completion publication;
`stream/pump.cpp` owns recurrent chunk continuation; and `stream/execute.cpp`
owns bootstrap, waiting, lease release, and final backing publication.
`execution/stream/execute.cpp` directly owns the public entry and the complete
bootstrap/wait/release/final-publication orchestration; no forwarding source
or second controller exists.
