#include "local.hpp"

#include "../../../../replay/host/payload/hash.hpp"

#include <limits>

namespace rund::node {
namespace {

constexpr ::rund::replay::Code kPayloadMismatch =
    ::rund::replay::Code::HostPayloadMismatch;

[[nodiscard]] ReasonCode
ReplayCode(const ::rund::host::Status status) noexcept {
  switch (status) {
  case ::rund::host::Status::Ok:
    return ReasonCode::Ok;
  case ::rund::host::Status::CapacityExceeded:
    return ReasonCode::TaskCapacityExceeded;
  case ::rund::host::Status::WouldBlock:
    return ReasonCode::IoWouldBlock;
  case ::rund::host::Status::ReplayMismatch:
    return ReasonCode::HostReplayEventMismatch;
  case ::rund::host::Status::Invalid:
    return ReasonCode::TaskInvalid;
  case ::rund::host::Status::SyscallFailed:
    return ReasonCode::IoSyscallFailed;
  case ::rund::host::Status::Unsupported:
    return ReasonCode::IoUnsupported;
  }
  return ReasonCode::IoSyscallFailed;
}

[[nodiscard]] const ::rund::host::Event *
NextEvent(const SchedulerState &state) noexcept {
  const std::size_t index = state.identity.next_expected_host_event;
  const auto &events = state.plan.value.expected->events();
  return index < events.size() ? &events[index] : nullptr;
}

} // namespace

void Scheduler::MarkHostIoPayloadMismatch(
    const ::rund::replay::Code code) noexcept {
  const char *const reason = ::rund::replay::error(code).data();
  {
    std::lock_guard lock{state_->evidence.mutex};
    state_->identity.host_replay_failed = true;
    state_->identity.host_replay_reason = reason;
    state_->identity.host_replay_payload_failed = true;
    state_->identity.host_replay_payload_reason = reason;
  }
  FailCurrentTaskOrScheduler(ReasonCode::HostReplayPayloadMismatch);
}

HostIoCompletion
Scheduler::ReplayHostIo(const HostIoOperation &operation) noexcept {
  if (!CurrentTaskIsCoroutine()) {
    return HostIoCompletion{.code = ReasonCode::TaskContextMissing};
  }
  if (operation.native != -1 || operation.host_id == 0u) {
    return HostIoCompletion{.code = ReasonCode::IoFdInvalid};
  }
  if (operation.data == nullptr && operation.size != 0u) {
    return HostIoCompletion{.code = ReasonCode::TaskInvalid};
  }

  const ::rund::host::EventKind event_kind =
      operation.kind == HostIoKind::Read ? ::rund::host::EventKind::IoRead
                                         : ::rund::host::EventKind::IoWrite;
  const ::rund::host::Event *const expected = NextEvent(*state_);
  const ::rund::host::Status status =
      expected == nullptr ? ::rund::host::Status::Invalid : expected->status;
  const std::uint64_t completed =
      expected == nullptr ? 0u : expected->completed_bytes;
  const std::int32_t native_error =
      expected == nullptr ? 0 : expected->native_errno;
  const ::rund::StableHash payload_hash =
      expected == nullptr ? ::rund::StableHash{} : expected->payload_hash;
  const HostEventCommitResult commit = CommitHostEvent(::rund::host::Event{
      .kind = event_kind,
      .status = status,
      .task_id = CurrentTaskId(),
      .host_handle_id = operation.host_id,
      .requested_bytes = static_cast<std::uint64_t>(operation.size),
      .completed_bytes = completed,
      .native_errno = native_error,
      .payload_hash = payload_hash,
  });
  if (!commit.ok) {
    return HostIoCompletion{.code = commit.code};
  }
  const ReasonCode code = ReplayCode(status);
  if (code != ReasonCode::Ok) {
    return HostIoCompletion{.code = code, .native_error = native_error};
  }
  if (completed > static_cast<std::uint64_t>(operation.size) ||
      completed > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max())) {
    MarkHostIoPayloadMismatch(kPayloadMismatch);
    return HostIoCompletion{.code = ReasonCode::HostReplayPayloadMismatch};
  }

  const std::size_t payload_index =
      state_->plan.value.expected->payloads().host_record_index(
          state_->identity.next_expected_host_payload);
  const replay_detail::payload::Binding binding{
      .event_sequence = commit.sequence,
      .kind = event_kind,
      .completed_bytes = completed,
      .payload_hash = payload_hash,
  };
  const std::size_t bytes = static_cast<std::size_t>(completed);
  const replay_detail::payload::MatchResult payload =
      operation.kind == HostIoKind::Read
          ? state_->plan.value.expected->payloads().ReadInto(
                payload_index, binding, operation.read_buffer().first(bytes))
          : state_->plan.value.expected->payloads().Matches(
                payload_index, binding, operation.write_buffer().first(bytes));
  if (!payload.ok()) {
    MarkHostIoPayloadMismatch(payload.code);
    return HostIoCompletion{.code = ReasonCode::HostReplayPayloadMismatch};
  }
  ++state_->identity.next_expected_host_payload;
  return HostIoCompletion{
      .code = ReasonCode::Ok,
      .bytes = static_cast<std::int64_t>(completed),
      .native_error = native_error,
  };
}

} // namespace rund::node
