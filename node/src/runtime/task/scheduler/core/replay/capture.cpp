#include "internal.hpp"

#include "../../host.hpp"
#include "../../state/model/join.hpp"
#include "../../state/model/task.hpp"
#include "../../state/model/timer.hpp"
#include "../../state/storage.hpp"

#include <rund/task/stats/slots.hpp>

#include <optional>
#include <utility>

namespace rund::node {

scheduler_host::ReplayInputMode Scheduler::ReplayInputMode() const noexcept {
  switch (state_->plan.mode()) {
  case ::rund::replay::detail::scope::Mode::Live:
    return scheduler_host::ReplayInputMode::Live;
  case ::rund::replay::detail::scope::Mode::Record:
    return scheduler_host::ReplayInputMode::Record;
  case ::rund::replay::detail::scope::Mode::Replay:
    return scheduler_host::ReplayInputMode::Replay;
  case ::rund::replay::detail::scope::Mode::Scenario:
    return scheduler_host::ReplayInputMode::Scenario;
  }
  return scheduler_host::ReplayInputMode::Unavailable;
}

scheduler_host::ReplayInputCapture Scheduler::BeginReplayInput(
    const replay_detail::payload::InputBinding &binding) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const auto fail = [this](const ::rund::replay::Code code) {
    (void)scheduler_replay_detail::poison_input(*state_, code);
    return scheduler_host::ReplayInputCapture{.code = code};
  };
  const ::rund::replay::detail::scope::Mode mode = state_->plan.mode();
  if (mode != ::rund::replay::detail::scope::Mode::Live &&
      mode != ::rund::replay::detail::scope::Mode::Record) {
    return fail(::rund::replay::Code::InputModeMismatch);
  }
  if (binding.source == 0u) {
    return fail(::rund::replay::Code::InputIdInvalid);
  }
  if (binding.schema == 0u) {
    return fail(::rund::replay::Code::InputSchemaInvalid);
  }
  if (state_->evidence.input_count >= state_->evidence.input_capacity) {
    return fail(::rund::replay::Code::InputCapacityExceeded);
  }
  if (CurrentTaskId() != 0u ||
      state_->resources.live_tasks.load(std::memory_order_acquire) != 0u) {
    return fail(::rund::replay::Code::InputCaptureNotRoot);
  }
  ReplayInputCaptureState &capture = state_->evidence.input_capture;
  if (state_->evidence.input_capture_active.load(std::memory_order_relaxed)) {
    capture.mutated = true;
    return fail(::rund::replay::Code::InputCaptureNested);
  }
  const std::uint64_t physical_event =
      state_->identity.next_host_event_sequence;
  if (physical_event == 0u || state_->evidence.next_input_capture_token == 0u) {
    return fail(::rund::replay::Code::InputCaptureUnavailable);
  }
  const std::uint64_t canonical_event = state_->plan.event(physical_event);
  if (canonical_event == 0u ||
      (mode == ::rund::replay::detail::scope::Mode::Record &&
       state_->evidence.host_events.size() != canonical_event - 1u)) {
    return fail(::rund::replay::Code::InputCaptureUnavailable);
  }
  if (state_->evidence.input_bytes == nullptr ||
      state_->evidence.input_byte_size > state_->evidence.input_bytes->size()) {
    return fail(::rund::replay::Code::InputWriterUnavailable);
  }
  const std::uint64_t token = state_->evidence.next_input_capture_token++;
  capture = ReplayInputCaptureState{
      .mutated = false,
      .token = token,
      .binding = binding,
      .event_offset =
          static_cast<std::uint64_t>(state_->evidence.host_events.size()),
      .event_sequence = state_->identity.next_host_event_sequence,
      .payload_offset = static_cast<std::uint64_t>(
          state_->evidence.host_payload_store.host_record_count()),
      .operation_count = ::rund::detail::task::Stat(
          state_->evidence.metrics, ::rund::detail::task::StatSlot::Operations),
      .simulation_fingerprint =
          scheduler_replay_detail::simulation_fingerprint(*state_),
      .byte_offset = state_->evidence.input_byte_size,
  };
  state_->evidence.input_capture_active.store(true, std::memory_order_release);
  const std::span<std::byte> bytes{*state_->evidence.input_bytes};
  return scheduler_host::ReplayInputCapture{
      .token = token,
      .code = ::rund::replay::Code::Ok,
      .bytes = bytes.subspan(state_->evidence.input_byte_size),
  };
}

void Scheduler::FailReplayInput(const ::rund::replay::Code code) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  (void)scheduler_replay_detail::poison_input(
      *state_, code == ::rund::replay::Code::Ok
                   ? ::rund::replay::Code::InputInvalid
                   : code);
}

void Scheduler::CancelReplayInput(
    const scheduler_host::ReplayInputCapture capture) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  ReplayInputCaptureState &active = state_->evidence.input_capture;
  if (!state_->evidence.input_capture_active.load(std::memory_order_relaxed) ||
      !capture.ok() || active.token != capture.token) {
    (void)scheduler_replay_detail::poison_input(
        *state_, ::rund::replay::Code::InputCaptureMismatch);
    return;
  }
  state_->evidence.input_capture_active.store(false, std::memory_order_release);
  active = {};
}

replay_detail::payload::ResolveResult
Scheduler::RejectReplayInput(const scheduler_host::ReplayInputCapture capture,
                             const ::rund::replay::Code code) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  ReplayInputCaptureState &active = state_->evidence.input_capture;
  if (!state_->evidence.input_capture_active.load(std::memory_order_relaxed) ||
      !capture.ok() || active.token != capture.token) {
    (void)scheduler_replay_detail::poison_input(
        *state_, ::rund::replay::Code::InputCaptureMismatch);
    return replay_detail::payload::ResolveResult{
        .code = ::rund::replay::Code::InputCaptureMismatch};
  }
  state_->evidence.input_capture_active.store(false, std::memory_order_release);
  active = {};
  const ::rund::replay::Code failure =
      code == ::rund::replay::Code::Ok
          ? ::rund::replay::Code::InputWriterInvalid
          : code;
  (void)scheduler_replay_detail::poison_input(*state_, failure);
  return replay_detail::payload::ResolveResult{.code = failure};
}

replay_detail::payload::ResolveResult Scheduler::FinishReplayInput(
    const replay_detail::payload::InputBinding &binding,
    const scheduler_host::ReplayInputCapture capture,
    const std::size_t byte_count) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const auto fail = [this](const ::rund::replay::Code code) {
    (void)scheduler_replay_detail::poison_input(*state_, code);
    return replay_detail::payload::ResolveResult{.code = code};
  };
  const bool capture_active =
      state_->evidence.input_capture_active.load(std::memory_order_relaxed);
  ReplayInputCaptureState active = state_->evidence.input_capture;
  state_->evidence.input_capture_active.store(false, std::memory_order_release);
  state_->evidence.input_capture = {};
  if (!capture_active || !capture.ok() || active.token != capture.token ||
      !scheduler_replay_detail::same_identity(active.binding, binding)) {
    return fail(::rund::replay::Code::InputCaptureMismatch);
  }
  if (active.mutated || CurrentTaskId() != 0u ||
      state_->resources.live_tasks.load(std::memory_order_acquire) != 0u ||
      active.operation_count !=
          ::rund::detail::task::Stat(
              state_->evidence.metrics,
              ::rund::detail::task::StatSlot::Operations) ||
      active.simulation_fingerprint !=
          scheduler_replay_detail::simulation_fingerprint(*state_)) {
    return fail(::rund::replay::Code::InputCaptureMutated);
  }
  if (state_->evidence.input_bytes == nullptr ||
      active.byte_offset != state_->evidence.input_byte_size ||
      active.byte_offset > state_->evidence.input_bytes->size() ||
      byte_count > state_->evidence.input_bytes->size() - active.byte_offset ||
      !scheduler_replay_detail::can_account_input(*state_, byte_count)) {
    return fail(::rund::replay::Code::InputWriterCapacityExceeded);
  }
  ::rund::node::replay_detail::payload::Bytes bytes =
      ::rund::node::replay_detail::payload::Bytes::share(
          state_->evidence.input_bytes, active.byte_offset, byte_count);
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Live) {
    state_->evidence.input_byte_size += byte_count;
    scheduler_replay_detail::account_input(*state_, byte_count);
    return replay_detail::payload::ResolveResult{.code =
                                                     ::rund::replay::Code::Ok,
                                                 .sequence = binding.sequence,
                                                 .bytes = std::move(bytes)};
  }
  if (active.event_offset > state_->evidence.host_events.size() ||
      active.payload_offset >
          state_->evidence.host_payload_store.host_record_count()) {
    return fail(::rund::replay::Code::InputCaptureCorrupt);
  }
  const std::uint64_t event_count =
      static_cast<std::uint64_t>(state_->evidence.host_events.size()) -
      active.event_offset;
  if (active.event_sequence > state_->identity.next_host_event_sequence ||
      event_count !=
          state_->identity.next_host_event_sequence - active.event_sequence) {
    return fail(::rund::replay::Code::InputCaptureIncomplete);
  }
  const std::uint64_t payload_count =
      static_cast<std::uint64_t>(
          state_->evidence.host_payload_store.host_record_count()) -
      active.payload_offset;
  const std::span<const ::rund::host::Event> events{
      state_->evidence.host_events};
  const std::span<const ::rund::host::Event> source_events =
      events.subspan(static_cast<std::size_t>(active.event_offset),
                     static_cast<std::size_t>(event_count));
  const std::optional<std::uint64_t> source_hash =
      state_->evidence.host_payload_store.SourceRangeHash(
          active.event_offset, source_events, active.payload_offset,
          payload_count);
  if (!source_hash.has_value()) {
    return fail(::rund::replay::Code::InputCaptureCorrupt);
  }
  try {
    const replay_detail::payload::Capture payload =
        replay_detail::payload::Capture::read(bytes.span());
    const replay_detail::payload::MatchResult stored =
        scheduler_replay_detail::store_replay_input(
            *state_, binding,
            replay_detail::payload::InputSourceRange{
                .event_offset = active.event_offset,
                .event_count = event_count,
                .payload_offset = active.payload_offset,
                .payload_count = payload_count,
                .hash = *source_hash,
            },
            bytes, payload);
    if (!stored.ok()) {
      return replay_detail::payload::ResolveResult{.code = stored.code};
    }
    state_->evidence.input_byte_size += byte_count;
    scheduler_replay_detail::account_input(*state_, byte_count);
    return replay_detail::payload::ResolveResult{.code =
                                                     ::rund::replay::Code::Ok,
                                                 .sequence = binding.sequence,
                                                 .bytes = std::move(bytes)};
  } catch (...) {
    return fail(::rund::replay::Code::InputRecordFailed);
  }
}

} // namespace rund::node
