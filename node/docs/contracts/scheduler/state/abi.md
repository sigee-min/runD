# Scheduler State ABI

This page routes the stable scheduler reason schema and owns the source-private
operation tags, public observation ABI, and canonical trace-hash field order.

## Reason ABI

The checked-in [`reason.def`](../../../../include/rund/reason.def)
schema is the sole authority for each assigned numeric value, C++ enumerator
name, stable text, and prepared-memory membership. The public
[`reason.hpp`](../../../../include/rund/reason.hpp) projects the
`std::uint16_t` enum from that schema. One compiled lookup in
[`reason.cpp`](../../../../src/runtime/reason.cpp) projects `ReasonString`,
`ValidReasonCode`, and `ValidPreparedMemoryReason`; no string table is
emitted into public-header consumers and there is no reverse string parser.

The launch ABI is the dense range 0 through 91. Unknown numeric values project
to `unknown_reason_code` and fail both
membership checks. Prepared-memory results accept exactly `Ok` and the ten
assigned values from 82 through 91. The lifecycle contract expands the schema
independently to verify every value, enumerator, string, category, contiguous
identity, and unknown-value boundary.

Lane residual-policy diagnostic strings describe optimization rejection, not
task failure codes; their authority remains the Lane contract.

## Operation ABI

Scheduler-visible operations use this stable tag set:

| Tag | Operation |
| ---: | --- |
| 0 | `None` |
| 1 | `RootSubmit` |
| 2 | `Spawn` |
| 3 | `Complete` |
| 4 | `Fail` |
| 5 | `Yield` |
| 6 | `SleepZero` |
| 7 | `TimerPark` |
| 8 | `TimerWake` |
| 9 | `JoinPark` |
| 10 | `JoinWake` |
| 11 | `ScopeEnter` |
| 12 | `ScopePark` |
| 13 | `ScopeWake` |
| 14 | `ChannelMake` |
| 15 | `ChannelSend` |
| 16 | `ChannelRecv` |
| 17 | `ChannelClose` |
| 18 | `ChannelWake` |
| 19 | `IoPark` |
| 20 | `IoWake` |
| 21 | `DeadlockWake` |
| 22 | `ChannelMatch` |
| 23 | `ChannelMatchBatch` |
| 24 | `TaskSpawnBatch` |
| 25 | `TaskTerminalBatch` |
| 26 | `YieldBatch` |
| 27 | `JoinBatch` |
| 28 | `PrimitiveTrap` |
| 29 | `TaskRootJoinEpochBatch` |
| 32 | `ExternalPark` |
| 33 | `ExternalWake` |

The tags have one non-product owner,
`rund::detail::task::OperationKind`, under
`task/operation/kind.hpp`. Neither the tag nor an operation record is part of
`rund::task` or `<rund/task.hpp>`. Public callers observe the derived
`task::Stats::trace_hash()` only.

An operation is mixed directly from its canonical scalar values. There is no
aggregate operation object, aggregate return, retained operation vector, or
second field-order owner on the recording path. After the operation domain
tag, the exact scalar order is:

```text
sequence, kind, task_id, target_id, wait_id, channel_id, fd, interest,
revents, deadline_ns, value_count, match_sequence, baton_epoch,
task_op_ordinal, target_op_ordinal, region_id, epoch_id, logical_count,
order_hash, side_exit_code, reason_code
```

## Trace Hash Law

The trace hash is a sequence of typed 64-bit scalar transitions. Starting from
`1469598103934665603`, each scalar `x_i` applies

```text
h_(i+1) = ((h_i xor x_i) * 1099511628211) mod 2^64.
```

Every record begins with exactly one 64-bit domain scalar:

| Record | Domain scalar |
| --- | ---: |
| operation | `0x6f7065726174696f` |
| observation | `0x6f62736572766174` |
| host event | `0x686f73746576656e` |

Unsigned 64-bit fields enter unchanged. A 16-bit enum or reason is
zero-extended. `fd` is a 32-bit signed value sign-extended to 64 bits.
`interest` and `revents` enter as their unsigned 16-bit bit patterns.
`deadline_ns` enters as its two's-complement 64-bit bit pattern. Host events
then mix the single canonical 64-bit value returned by `host::hash_event`.
These conversions are independent of object padding, host endianness, pointer
identity, and compiler aggregate-return ABI.

The hash is deterministic evidence, not a cryptographic collision proof.

## Observation ABI

External timer and fd-readiness events are observations. They are copied into
runtime reports up to `observation_capacity`.

| Tag | Observation |
| ---: | --- |
| 0 | `None` |
| 1 | `TimerDue` |
| 2 | `IoReady` |
| 3 | `IoInvalid` |
| 4 | `IoPollFailed` |

The public observation record is:

```cpp fragment
struct task::Observation {
  uint64_t sequence;
  task::ObservationKind kind;
  uint64_t task_id;
  uint64_t wait_id;
  int fd;
  short interest;
  short revents;
  int64_t deadline_ns;
  ReasonCode reason_code;
};
```

After the observation domain scalar, the hash order is exactly the public field
order above and uses the same width-conversion law. The retained copy and the
hash consume the same normalized observation value.

When the copied observation vector is full, the newest copied record is
dropped, `observation_dropped` is incremented, and the observation is still
mixed into `trace_hash`. Dropping an evidence copy is not a runtime failure.

The deterministic replay claim for timers, IO, and network readiness is
limited: with the same observation sequence, node-owned ordering after
observation is deterministic. This contract does not claim identical
wall-clock wake time or external fd/network readiness timing. Timer and
fd-readiness observations remain scheduler evidence; host replay fields and
mismatches are owned by [Host](../../host.md).
