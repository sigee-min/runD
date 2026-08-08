#include "local.hpp"

#include "../../../../replay/host/payload/hash.hpp"
#include "../../reactor/close.hpp"

#include <cerrno>
namespace rund::node {
namespace host_io {

::rund::host::io::ReadResult
ReadResult(const HostIoCompletion completion) noexcept {
  return completion.code == ReasonCode::Ok
             ? ::rund::host::io::detail::Access::read(completion.bytes,
                                                      completion.native_error)
             : ::rund::host::io::detail::Access::read(completion.code,
                                                      completion.native_error);
}

::rund::host::io::WriteResult
WriteResult(const HostIoCompletion completion) noexcept {
  return completion.code == ReasonCode::Ok
             ? ::rund::host::io::detail::Access::write(completion.bytes,
                                                       completion.native_error)
             : ::rund::host::io::detail::Access::write(completion.code,
                                                       completion.native_error);
}

ReasonCode OutcomeCode(const HostIoOutcome &outcome) noexcept {
  if (outcome.kind == HostIoOutcomeKind::Unsupported) {
    return ReasonCode::IoUnsupported;
  }
  if (outcome.kind == HostIoOutcomeKind::InvalidBuffer) {
    return ReasonCode::TaskInvalid;
  }
  if (outcome.kind == HostIoOutcomeKind::Ready && outcome.value >= 0) {
    return ReasonCode::Ok;
  }
  if (outcome.error == EAGAIN || outcome.error == EWOULDBLOCK) {
    return ReasonCode::IoWouldBlock;
  }
  return ReasonCode::IoSyscallFailed;
}

::rund::host::Status EventStatus(const ReasonCode code) noexcept {
  switch (code) {
  case ReasonCode::Ok:
    return ::rund::host::Status::Ok;
  case ReasonCode::TaskCapacityExceeded:
    return ::rund::host::Status::CapacityExceeded;
  case ReasonCode::IoWouldBlock:
    return ::rund::host::Status::WouldBlock;
  case ReasonCode::HostReplayEventMismatch:
  case ReasonCode::HostReplayPayloadMismatch:
    return ::rund::host::Status::ReplayMismatch;
  case ReasonCode::TaskInvalid:
  case ReasonCode::IoFdInvalid:
    return ::rund::host::Status::Invalid;
  case ReasonCode::IoUnsupported:
    return ::rund::host::Status::Unsupported;
  default:
    return ::rund::host::Status::SyscallFailed;
  }
}

} // namespace host_io

::rund::host::io::CloseResult
Scheduler::CloseHostFd(const ::rund::host::io::FdView fd) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();

  const ::rund::host::io::detail::FdIdentity identity =
      ::rund::host::io::detail::Project(fd);
  const ::rund::ReasonCode invalidation =
      ReactorCloseInvalidateFd(*this, identity.native);
  const NativeIoResult closed = NativeClose(identity.native);
  CompletePrimitiveCommit();

  if (invalidation != ::rund::ReasonCode::Ok) {
    return ::rund::host::io::detail::Access::close(invalidation,
                                                   closed.native_error());
  }
  switch (closed.disposition()) {
  case NativeIoDisposition::Complete:
    return ::rund::host::io::detail::Access::close();
  case NativeIoDisposition::Unsupported:
    return ::rund::host::io::detail::Access::close(ReasonCode::IoUnsupported,
                                                   closed.native_error());
  case NativeIoDisposition::InvalidBuffer:
  case NativeIoDisposition::Failed:
    return ::rund::host::io::detail::Access::close(ReasonCode::IoSyscallFailed,
                                                   closed.native_error());
  }
  std::abort();
}

bool Scheduler::SuspendHostIo(const HostIoOperation &operation,
                              void **const token,
                              ReasonCode &failure) noexcept {
  failure = ReasonCode::Ok;
  if (!CurrentTaskIsCoroutine()) {
    failure = ReasonCode::TaskContextMissing;
    return false;
  }
  if (operation.native < 0 ||
      operation.host_id != static_cast<std::uint64_t>(operation.native) + 1u) {
    failure = ReasonCode::IoFdInvalid;
    return false;
  }
  if (operation.data == nullptr && operation.size != 0u) {
    failure = ReasonCode::TaskInvalid;
    return false;
  }
  if (!ReserveHostPayloadCapacity(operation.size)) {
    failure = ReasonCode::TaskCapacityExceeded;
    FailCurrentTaskOrScheduler(failure);
    return false;
  }
  HostIoSlot *const slot = host_io::Claim(*this, *state_, operation);
  if (slot == nullptr) {
    ReleaseHostPayloadCapacity(operation.size);
    failure = ReasonCode::TaskCapacityExceeded;
    FailCurrentTaskOrScheduler(failure);
    return false;
  }
  if (!ParkExternal(slot->phase, slot->wake)) {
    ReleaseHostPayloadCapacity(operation.size);
    host_io::Release(state_->host_io, *slot);
    failure = ReasonCode::TaskContextMissing;
    return false;
  }
  *token = slot;
  host_io::Queue(state_->host_io, *slot);
  return true;
}

bool Scheduler::SuspendHostIoRead(
    const ::rund::host::io::FdView fd, const std::span<std::byte> buffer,
    void **const token, ::rund::host::io::ReadResult *const result) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  if (token == nullptr || result == nullptr) {
    CompletePrimitiveCommit();
    return false;
  }
  *token = nullptr;
  const auto identity = ::rund::host::io::detail::Project(fd);
  HostIoOperation operation{
      .data = buffer.data(),
      .host_id = identity.host_id,
      .size = buffer.size(),
      .native = identity.native,
      .kind = HostIoKind::Read,
  };
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Replay) {
    *result = host_io::ReadResult(ReplayHostIo(operation));
    CompletePrimitiveCommit();
    return false;
  }
  ReasonCode failure = ReasonCode::Ok;
  if (!SuspendHostIo(operation, token, failure)) {
    *result = host_io::ReadResult(HostIoCompletion{.code = failure});
    CompletePrimitiveCommit();
    return false;
  }
  return true;
}

bool Scheduler::SuspendHostIoWrite(
    const ::rund::host::io::FdView fd, const std::span<const std::byte> buffer,
    void **const token, ::rund::host::io::WriteResult *const result) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  if (token == nullptr || result == nullptr) {
    CompletePrimitiveCommit();
    return false;
  }
  *token = nullptr;
  const auto identity = ::rund::host::io::detail::Project(fd);
  HostIoOperation operation{
      .data = buffer.data(),
      .host_id = identity.host_id,
      .size = buffer.size(),
      .native = identity.native,
      .kind = HostIoKind::Write,
  };
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Replay) {
    *result = host_io::WriteResult(ReplayHostIo(operation));
    CompletePrimitiveCommit();
    return false;
  }
  ReasonCode failure = ReasonCode::Ok;
  if (!SuspendHostIo(operation, token, failure)) {
    *result = host_io::WriteResult(HostIoCompletion{.code = failure});
    CompletePrimitiveCommit();
    return false;
  }
  return true;
}

HostIoCompletion Scheduler::CompleteHostIo(void *const token,
                                           const HostIoKind kind) noexcept {
  EnsureCurrentCommit();
  auto *const slot = static_cast<HostIoSlot *>(token);
  if (!host_io::Valid(state_->host_io, slot, kind)) {
    CompletePrimitiveCommit();
    return {};
  }
  const HostIoOperation &operation = slot->operation;
  const HostIoOutcome &outcome = slot->outcome;
  ReleaseHostPayloadCapacity(operation.size);

  ReasonCode code = host_io::OutcomeCode(outcome);
  if (outcome.value >= 0 &&
      static_cast<std::uint64_t>(outcome.value) > operation.size) {
    code = ReasonCode::TaskInvalid;
  }
  const std::size_t completed =
      code == ReasonCode::Ok ? static_cast<std::size_t>(outcome.value) : 0u;
  const replay_detail::payload::Capture payload =
      replay_detail::payload::Capture::read(
          operation.write_buffer().first(completed));
  const ::rund::StableHash payload_hash =
      code == ReasonCode::Ok ? payload.hash() : ::rund::StableHash{};
  const ::rund::host::EventKind event_kind =
      operation.kind == HostIoKind::Read ? ::rund::host::EventKind::IoRead
                                         : ::rund::host::EventKind::IoWrite;
  const HostEventCommitResult commit = CommitHostEvent(::rund::host::Event{
      .kind = event_kind,
      .status = host_io::EventStatus(code),
      .task_id = CurrentTaskId(),
      .host_handle_id = operation.host_id,
      .requested_bytes = static_cast<std::uint64_t>(operation.size),
      .completed_bytes = static_cast<std::uint64_t>(completed),
      .native_errno = outcome.error,
      .payload_hash = payload_hash,
  });
  ReasonCode final_code = commit.ok() ? code : commit.code();
  if (final_code == ReasonCode::Ok) {
    final_code =
        RecordHostPayloadForCommittedEvent(commit, event_kind, payload);
  }
  const std::int64_t native_value = outcome.value;
  const int native_error = outcome.error;
  host_io::Release(state_->host_io, *slot);
  CompletePrimitiveCommit();
  return HostIoCompletion{
      .code = final_code,
      .bytes = final_code == ReasonCode::Ok ? native_value : -1,
      .native_error = native_error,
  };
}

::rund::host::io::ReadResult
Scheduler::CompleteHostIoRead(void *const token) noexcept {
  return host_io::ReadResult(CompleteHostIo(token, HostIoKind::Read));
}

::rund::host::io::WriteResult
Scheduler::CompleteHostIoWrite(void *const token) noexcept {
  return host_io::WriteResult(CompleteHostIo(token, HostIoKind::Write));
}

} // namespace rund::node
