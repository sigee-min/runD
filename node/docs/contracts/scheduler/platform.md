# Platform Boundary

This page owns the scheduler's platform adapters. Task suspension itself is
portable C++20 coroutine state and has no ISA-specific context-switch backend.

## Portable Scheduler Core

`Task<T>` coroutine frames, task records, ready queues, completion cells, lane
dispatch, deterministic commit, and cancellation are platform-independent.
Configuration must not reject a host merely because it is not the primary
shipping platform. Individual unavailable accelerator or readiness backends
fail when selected or used.

The scheduler does not allocate task stacks, switch register contexts, use
platform record APIs, or maintain a second suspension representation.
`spawn(lambda)` is a completion-bound leaf and cannot call suspending task
primitives. Suspending work uses `Task<T>`.

## Reactor Boundary

The dependency direction is fixed:

```text
Scheduler
  -> runtime/reactor platform-neutral interface
  -> runtime/platform/<host>/reactor backend
```

The scheduler owns logical waits, public descriptor generations,
deterministic ready order, task park/wake, and commit. The platform-neutral
owner is `runtime/reactor/`: `readiness/state.hpp` defines the width-safe
`ReactorHandle`, `ReactorInterest`, and `ReactorEvent` state;
`readiness/mask.hpp` owns logical bit composition, encoding, decoding, and
matching; `readiness/handle.hpp` owns checked public/native handle conversion;
`platform.hpp` owns the complete backend state, operation, registration,
probe, and normalized-result contract; `diagnostics.hpp` owns observation-only
backend and scheduler-policy counters. None contains POSIX readiness masks,
`kevent`, `epoll_event`, IOCP records, or another platform SDK type. Public
admitted `io::Fd` carriers and public evidence bits are converted only at
host/scheduler API edges; raw integer readiness is not a public scheduler
surface.

Native-handle conversion, registration syscalls, event scratch, polling, and
teardown are owned independently by:

- `runtime/platform/mac/reactor/`: kqueue backend;
- `runtime/platform/linux/reactor/`: epoll backend;
- `runtime/platform/portable/reactor/`: POSIX poll backend;
- `runtime/platform/unavailable/reactor.cpp`: explicit non-Unix unavailable
  backend.

CMake selects exactly one backend source set. Backends do not include one
another and cannot own task identity or completion. Native readiness order is
normalized at the interface before scheduler wake and deterministic commit.
The scheduler source tree does not include POSIX, kqueue, or epoll headers;
platform event masks are normalized through the interface owner.
Plain-handle identity lookup, close-on-exec retention, and release also cross
the neutral interface; scheduler registration policy never calls `fstat`,
`fcntl`, `dup`, or `close` directly.
The lookup result is one typed value with exactly two dispositions:
`Invalid`, whose device, inode, and mode are fixed to zero, or `Described`,
which carries all three fields. The value itself owns same-object comparison;
two invalid values never prove identity. The scheduler stores that complete
snapshot once rather than mirroring its payload and validity in separate
fields.
Adding a Windows IOCP backend requires a separate
`runtime/platform/windows/reactor/` owner implementing the same interface; it
does not add IOCP branches to scheduler code. An IOCP owner drains completed
operations into logical `ReactorEvent` records; the scheduler contract does
not require a POSIX readiness mask or expose an `OVERLAPPED` record.
Configuration does not reject a host merely because a native readiness backend
is absent. On a non-Unix host, CMake selects the real
`runtime/platform/unavailable/` owner. Its capacity preparation is stateless
and succeeds, so the portable scheduler and ordinary product work remain
usable. An actual readiness probe, registration, or poll fails with
`ReactorBackendUnavailable`; it never reports capacity pressure. The same
selected owner implements every interface operation and the state deleter, so
there is no empty component, link-time hole, or success-masquerading stub.

One-shot immediate-probe orchestration is owned once by
`scheduler/reactor/poll.cpp`; it reuses the Runtime-owned platform and
pre-reserved result scratch. Backend lifecycle and scheduler-policy diagnostic
counters are owned once by `scheduler/reactor/diagnostic/platform.cpp`. Platform
backends only translate the neutral contract and issue native calls; they do
not copy either orchestration or policy counters.

Each native lifecycle or single-registration operation has exactly one
disposition: `Success`, `Invalid`, `Failed`, or `BackendUnavailable`.
`Success` and `BackendUnavailable` fix the platform error to zero, while
`Invalid` and `Failed` carry the native or allocation error. Stateless
preparation in the unavailable owner remains `Success`; opening or mutating
readiness state reports `BackendUnavailable`. Removing an invalid descriptor
also retires the platform-private interest mirror, so a consumed best-effort
remove cannot be rediscovered indefinitely by a later blocking poll. Logical
change retry remains owned by the scheduler queue.

The registration-batch operation has exactly one disposition: `Success`,
`Invalid`, `Failed`, or `BackendUnavailable`. `Success` and
`BackendUnavailable` fix the platform error and failed index to zero.
`Invalid` and `Failed` carry the native or allocation error plus the first
failed logical change index. That index acknowledges the successful logical
prefix; it is not a native rollback frontier. Linux and the portable backend
stop at that change, while a kqueue receipt batch may already have applied a
later native filter. The scheduler removes only the acknowledged prefix and
normalizes the remaining queue. `Failed` and `BackendUnavailable` retain the
failed logical change and suffix for retry. `Invalid` remains queue-owned until
the scheduler has prepared every invalid wait for that descriptor for one
drain transaction; acknowledgement then retires all strict changes for that
descriptor while preserving its best-effort cleanup remove and every other
descriptor's suffix.

The native poll operation has exactly one disposition: `Success`, `Invalid`,
`Failed`, or `BackendUnavailable`. The caller-provided
`ReactorPlatformReady` vector is the only ready payload and count authority;
the result stores no pointer or count mirror. Every call clears that vector
before observing the backend. `Success` may publish an empty or nonempty
vector, while every other disposition leaves it empty. `Success` and
`BackendUnavailable` fix the platform error to zero; `Invalid` and `Failed`
carry the native or allocation error. A successful vector entry may itself be
marked invalid for one descriptor; that per-entry classification is distinct
from an operation-level `Invalid` poll.

The batch immediate-probe operation has exactly one disposition: `Success`,
`Failed`, or `BackendUnavailable`. `Success` includes an empty ready set. The
caller-provided `BatchIoReady` vector is the only ready payload and count
authority; the result does not mirror its size. `Success` and
`BackendUnavailable` fix the platform error to zero, while `Failed` carries
the native or allocation error. The unavailable owner clears the output and
reports `BackendUnavailable` without pretending capacity pressure. Invalid
descriptors remain per-entry `BatchIoReady` classifications rather than an
operation-level failure.

## Native Byte Substrate

Portable host and scheduler sources depend only on
`runtime/platform/io.hpp`, `runtime/platform/net.hpp`, and the narrow
`runtime/reactor/readiness/{state,mask,handle}.hpp` or
`runtime/reactor/platform.hpp` contract they consume. There is no readiness
aggregate; a state-only consumer cannot acquire mask or conversion algorithms
transitively. `OpenOptions::mode` is a fixed-width
`uint32_t`; `mode_t` is confined to the POSIX implementation. Network address
operations cross the neutral boundary as canonical `rund::net::Address`.
`sockaddr`,
`socklen_t`, native family-size validation, and byte conversion are confined to
`runtime/platform/posix/address.*`.

- `runtime/platform/posix/io.cpp`: fd configuration, admitted-fd identity
  lookup, and the POSIX implementation of reactor handle retention.
- `runtime/platform/posix/probe.cpp`: the single POSIX immediate readiness
  probe algorithm. Each selected reactor backend owns its reusable probe
  buffer and delegates here; there are no per-backend copies and no warm probe
  allocation within configured capacity.
- `runtime/platform/posix/net.cpp`: raw stream, listener, accept, and connect
  operations. Direct stream verbs preserve descriptor blocking semantics;
  `try` stream verbs carry `MSG_DONTWAIT` on the native call.
- `runtime/platform/posix/net/socket/call.hpp`: the single native owner for
  per-call nonblocking and signal-safe send flags plus admission-time Darwin
  `SO_NOSIGPIPE` preparation.
- `runtime/platform/posix/net/datagram.cpp`: raw datagram byte/address
  operations with per-call nonblocking semantics.
- `runtime/platform/posix/net/options.cpp`: selected socket options.
- `runtime/platform/posix/net/vectored.cpp`: per-call nonblocking vectored byte
  operations over caller-owned spans.
- `runtime/platform/unavailable/io.cpp` and `net.cpp`: complete native surface
  owners that return `IoUnsupported` without issuing host calls.

These files own native calls only. Protocol, retry, session, queue, and
gameplay meaning remain outside the scheduler.
