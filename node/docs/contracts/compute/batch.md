# Compute Batch Contract

## Scope

`rund::compute::Batch` amortizes native accelerator submission for small,
already prepared resident Jobs. It is a bounded execution terminal, not a
second graph builder, a scheduler, or a timing-based coalescer. Flow and
Program remain the only graph and compilation authorities.

## Admission

A Batch stores at most 64 Job owners in fixed inline storage. `add(job)`
accepts heterogeneous signatures, graphs, programs, and shapes only when every
Job belongs to the exact same opened Device. Backend equality without Device
identity is insufficient. CPU Jobs are unsupported because CPU execution has
no native command submission to amortize. The same Job state cannot appear
twice.

Admission has no fallback, spill, heap growth, implicit delay, or background
flush. Empty, full, duplicate, cross-Device, CPU, moved-from, and unprepared
inputs retain distinct typed reasons:

| Reason | Meaning |
| --- | --- |
| `BatchEmpty` | `run()` has no admitted Job. |
| `BatchCapacity` | The fixed 64-Job envelope is full. |
| `BatchDuplicate` | One Job state was added more than once. |
| `BatchDeviceMismatch` | A Job belongs to another Device owner. |
| `BatchCpuUnsupported` | A CPU Job was supplied. |
| `BatchBusy` | At least one Job or prepared submission is active. |
| `BatchPreparedInvalid` | A Job or its prepared accelerator state is invalid. |

`Status::code()`, `error()`, and `exit_code()` are derived from that one
Reason; no backend text becomes a parallel public failure authority.

## Execution

`add()` is the sole membership authority: it validates prepared state, CPU
exclusion, exact Device identity, capacity, and duplicate identity once. Batch
owns each admitted Job state, and the public surface has no erase, replace, or
mutable membership view. Moving a Job handle or the Batch therefore cannot
change the admitted identity, Program, or Device. `run()` does not repeat that
sealed work. It only holds every Job-local gate, checks the dynamic phase, and
then claims every prepared submission. A fixed function pointer and caller
stack context commit Job phases and release the Job gates only after both claim
sets succeed. There is no closure allocation. No Job phase, result, evidence,
or output is changed before that callback. If either claim set fails, RAII
releases the held gates, the callback is not invoked, and every Job preserves
its preceding state. This is the all-claim-or-execute-none boundary. It also
covers an internal asynchronous prepared submission that is active while its
product Job phase remains idle.

After admission, Jobs are encoded in `add` order. Metal uses one command
buffer, one compute encoder, one commit, and one wait for the complete Batch.
Vulkan uses one existing bounded command slot, one primary command buffer, and
one `vkQueueSubmit`. Each Vulkan Job's complete step stream was recorded once
as a secondary command during preparation; Batch executes those immutable
secondaries in admission order rather than re-encoding their dispatches and
barriers. Backend validation and primary encoding complete before commit; an
encode failure therefore executes none of the commands. Once the single command
is submitted, each Job finishes in input order and retains its own result,
overflow status, graph identity, and output identity.

Resident Jobs own disjoint input, output, status, and scratch buffers. Batch
therefore needs barriers only between dependent steps inside a Job, not between
independent Jobs. Admitting borrowed or aliased Job storage would require a
separate range-hazard owner and is outside this contract.

Metal derives packed-Map views and proves their alias-free layout once when a
sealed Batch first runs. The retained Batch workspace keys that plan by the
immutable Kernel step/artifact identity and shape fields. A warm run performs
one linear entry sweep that captures both the immutable key and the prepared
Map resource, then reuses the views, grouping, offsets, and alias proof. Pack,
encode, and unpack consume that captured fixed-array projection; they do not
reopen the same prepared resource for every binding. The
writable Job's active and pending prepared states reference the same immutable
Kernel steps and exact shape, while their owned buffers remain disjoint, so an
explicit `write()` preserves the key. Adding a Job resets the workspace before
the next run. A different internal execution identity fails the key check and
rebuilds the plan before encoding; if that internal-only transition requires a
larger packed extent, the workspace replaces undersized Metal buffers before
the plan can execute.

Heterogeneous Jobs share only submission. Batch does not fuse graphs,
dispatches, pipelines, resources, or results. Reusing the same Batch executes
the same retained Job set again; changing input remains an explicit
`Job::write()` boundary.

## Evidence

`Batch::stats()` owns the one shared execution submission, queue capacity and
pressure, aggregate dispatch count, and backend timing. Immediately after
`Batch::run()` and before any explicit read, each `Job::stats()` owns only that
Job's graph, dispatch, fusion, round-trip, and result evidence. Shared
execution submission, queue, compile, allocation, transfer, and timing counters
are zero in that snapshot so summing Jobs cannot count one native submission 64
times. A later explicit accelerator read may record its own transfer command;
that read submission is not Batch execution. Batch graph and output hashes are
zero because there is no synthetic aggregate graph or result.

Output hashing remains an explicit read boundary. `Job::run()` and
`Batch::run()` publish graph and execution evidence; `Job::read()` or
`read_all()` publishes that Job's output hash exactly as in serial execution.

## Performance Model

For `N` independent Jobs, let `S` be fixed native submit-and-wait cost and `C`
be per-Job warm wrapper and device work. Serial and Batch costs are modeled by

```text
T_serial = N(S + C)
T_batch  = S + NC
speedup  = N(S + C) / (S + NC) <= N.
```

This is a throughput law, not a claim that one Job becomes ten times faster.
For the fixed `N = 64` envelope, a 30x result requires
`S / C >= 1856 / 34`, approximately `54.59`. This is a bound, not a product
claim. The official Compute measurement times the same prepared 64-Job set and
input in four serial/Batch and four Batch/serial pairs after pre-warming both
paths. It checks one Batch submit and per-Job graph/output parity and records
independent end-to-end medians. All oracle reads happen after timing. Driver submit-wait, kernel,
and `max(wall - submit_wait, 0)` medians are diagnostics only. Vulkan already
reduces a prepared Job's `D` dispatch/barrier records from `Theta(D)` warm host
construction to one secondary-command execute wrapper. If host residual
dominates after one-submit batching, the next encoding owner is Metal; a Vulkan
residual points above native step encoding. If submit-wait dominates,
submission amortization is the supported structural remedy.

The sealed admission invariant owns Program and Device membership. The run-side
work is `N` nonblocking Job gate acquisitions plus `N` prepared-submission
acquisitions: linear work with no membership scan. Internal prepared execution
retains null and backend function-pointer checks before indirect calls; those
are memory-safety boundaries, not membership authorities.

For a packed group with `g` Jobs, `R` inputs, and `W` outputs, the alias proof
collects IDs into two fixed contiguous arrays, sorts them, rejects adjacent
output duplicates, and intersects input/output IDs linearly. Its bound is
`O(gR log(gR) + gW log(gW))`, with no allocation and at most 1,024 IDs in
either array. The retained Metal plan then moves even that reduced proof out of
the warm path. Warm plan admission is `Theta(N)` scalar identity checks,
followed by the payload pack/unpack that the selected packed route already
requires. No payload-size crossover is claimed. The focused command
`tools/measure/compute/run --batch metal` sweeps 64, 256, 1,024, 4,096,
16,384, and 65,536 elements per Job in balanced serial-first/Batch-first pairs.
Those observations remain the authority for deciding whether a future
packed/nonpacked threshold is justified.

A warm packed group captures exactly `g` prepared-resource projections in
admission order and passes those non-owning pointers through pack, encode, and
unpack. Prepared-resource validation and pointer chasing are therefore
`Theta(g)` while the required payload work remains `gR + gW` copies. Copy
offsets, dispatch geometry, and command order follow admission order. Storage
uses the bounded 64-entry caller-stack envelope with no heap allocation or
retained resource authority.

## Verification

`compute.batch` owns typed admission, allocation-free fixed-storage Batch
construction/admission, the 64-Job bound, all-claim atomicity, heterogeneous
graph execution, one-submit evidence ownership, serial/Batched hash parity,
Job/Batch move ownership, reusable execution after explicit input writes, and
exact collective overflow propagation on Metal and Vulkan. A deterministic
reentrant boundary test directly holds one prepared submission, calls the
product Batch while its Job phase is idle, and proves `BatchBusy` without
changing existing Job or Batch evidence. Its warm native run may expose
driver-owned command allocation, but cannot exceed the C++ allocation calls
observed for the same Job set executed serially.

The installed Compute consumer compiles and runs the same public Batch surface
through `runD::sdk`. `tools/measure/compute/run` owns the Release A/B method;
its parser rejects column drift and treats only serial and Batch wall medians
as performance-baseline authorities.
