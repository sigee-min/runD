#include "../host.hpp"
#include "../reactor/registry.hpp"
#include "../state/model/join.hpp"
#include "../state/model/task.hpp"
#include "../state/model/timer.hpp"
#include "../state/storage.hpp"

#include "../../../replay/host/payload/hash.hpp"
#include "../../../replay/input/plan.hpp"

#include <kernel/core/checked.hpp>
#include <rund/task/stats/slots.hpp>

#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace rund::node {
namespace {

[[nodiscard]] replay_detail::payload::MatchResult
PoisonInput(SchedulerState &state, const ::rund::replay::Code code) noexcept {
  const char *const reason = ::rund::replay::error(code).data();
  state.identity.host_replay_failed = true;
  state.identity.host_replay_reason = reason;
  state.identity.host_replay_payload_failed = true;
  state.identity.host_replay_payload_reason = reason;
  return replay_detail::payload::MatchResult{.code = code};
}

[[nodiscard]] replay_detail::payload::MatchResult
StoreReplayInput(SchedulerState &state,
                 const replay_detail::payload::InputBinding &binding,
                 const replay_detail::payload::InputSourceRange source_range,
                 ::rund::node::replay_detail::payload::Bytes bytes,
                 const replay_detail::payload::Capture payload) noexcept {
  if (binding.source == 0u) {
    return PoisonInput(state, ::rund::replay::Code::InputIdInvalid);
  }
  if (binding.schema == 0u) {
    return PoisonInput(state, ::rund::replay::Code::InputSchemaInvalid);
  }
  if (!payload) {
    return PoisonInput(state, ::rund::replay::Code::InputRecordFailed);
  }
  const std::uint64_t byte_count = static_cast<std::uint64_t>(bytes.size());
  const std::uint64_t retained =
      state.evidence.host_payload_store.logical_bytes();
  const std::uint64_t capacity =
      state.resources.limits.host_payload_capacity_bytes;
  if (retained > capacity || byte_count > capacity - retained) {
    return PoisonInput(state, ::rund::replay::Code::InputCapacityExceeded);
  }
  try {
    if (!state.evidence.host_payload_store.AppendInput(
            binding.source, binding.schema, binding.sequence, source_range,
            std::move(bytes), payload)) {
      return PoisonInput(state, ::rund::replay::Code::InputRecordFailed);
    }
  } catch (...) {
    return PoisonInput(state, ::rund::replay::Code::InputRecordFailed);
  }
  return replay_detail::payload::MatchResult{.code = ::rund::replay::Code::Ok};
}

[[nodiscard]] bool
SameIdentity(const replay_detail::payload::InputBinding &left,
             const replay_detail::payload::InputBinding &right) noexcept {
  return left.source == right.source && left.schema == right.schema;
}

[[nodiscard]] bool ToSize(const std::uint64_t value,
                          std::size_t &out) noexcept {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  out = static_cast<std::size_t>(value);
  return true;
}

[[nodiscard]] bool CanAccountInput(const SchedulerState &state,
                                   const std::size_t byte_count) noexcept {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  const std::uint64_t bytes = static_cast<std::uint64_t>(byte_count);
  return rund::kernel::checked::add(state.evidence.input_consumed_bytes, bytes);
}

void AccountInput(SchedulerState &state,
                  const std::size_t byte_count) noexcept {
  state.evidence.input_consumed_bytes += static_cast<std::uint64_t>(byte_count);
  ++state.evidence.input_count;
}

[[nodiscard]] std::uint64_t
SimulationFingerprint(const SchedulerState &state) noexcept {
  std::uint64_t hash = kFnvOffset;
  MixHash(hash, state.identity.next_task_id);
  MixHash(hash, state.identity.next_scope_id);
  MixHash(hash, state.identity.next_spawn_index);
  MixHash(hash, state.identity.next_wait_id);
  MixHash(hash, state.identity.next_reactor_many_group_id);
  MixHash(hash, state.identity.next_reactor_ready_set_id);
  MixHash(hash, state.identity.next_stop_source_id);
  MixHash(hash, state.identity.next_timer_sequence);
  MixHash(hash, state.identity.next_channel_id);
  MixHash(hash, state.resources.live_tasks.load(std::memory_order_acquire));
  MixHash(hash, state.resources.live_channels);
  MixHash(hash, state.resources.live_channel_buffer_slots);
  MixHash(hash, state.resources.live_channel_waits);
  MixHash(hash, state.ready.records.size());
  MixHash(hash, state.ready.timers.size());
  MixHash(hash, state.ready.join_waits.size());
  MixHash(hash, ReactorRegistrySize(state.reactor.reactor));
  return hash;
}

} // namespace
} // namespace rund::node

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
    (void)PoisonInput(*state_, code);
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
      .simulation_fingerprint = SimulationFingerprint(*state_),
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
  (void)PoisonInput(*state_, code == ::rund::replay::Code::Ok
                                 ? ::rund::replay::Code::InputInvalid
                                 : code);
}

void Scheduler::CancelReplayInput(
    const scheduler_host::ReplayInputCapture capture) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  ReplayInputCaptureState &active = state_->evidence.input_capture;
  if (!state_->evidence.input_capture_active.load(std::memory_order_relaxed) ||
      !capture.ok() || active.token != capture.token) {
    (void)PoisonInput(*state_, ::rund::replay::Code::InputCaptureMismatch);
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
    (void)PoisonInput(*state_, ::rund::replay::Code::InputCaptureMismatch);
    return replay_detail::payload::ResolveResult{
        .code = ::rund::replay::Code::InputCaptureMismatch};
  }
  state_->evidence.input_capture_active.store(false, std::memory_order_release);
  active = {};
  const ::rund::replay::Code failure =
      code == ::rund::replay::Code::Ok
          ? ::rund::replay::Code::InputWriterInvalid
          : code;
  (void)PoisonInput(*state_, failure);
  return replay_detail::payload::ResolveResult{.code = failure};
}

replay_detail::payload::ResolveResult Scheduler::FinishReplayInput(
    const replay_detail::payload::InputBinding &binding,
    const scheduler_host::ReplayInputCapture capture,
    const std::size_t byte_count) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const auto fail = [this](const ::rund::replay::Code code) {
    (void)PoisonInput(*state_, code);
    return replay_detail::payload::ResolveResult{.code = code};
  };
  const bool capture_active =
      state_->evidence.input_capture_active.load(std::memory_order_relaxed);
  ReplayInputCaptureState active = state_->evidence.input_capture;
  state_->evidence.input_capture_active.store(false, std::memory_order_release);
  state_->evidence.input_capture = {};
  if (!capture_active || !capture.ok() || active.token != capture.token ||
      !SameIdentity(active.binding, binding)) {
    return fail(::rund::replay::Code::InputCaptureMismatch);
  }
  if (active.mutated || CurrentTaskId() != 0u ||
      state_->resources.live_tasks.load(std::memory_order_acquire) != 0u ||
      active.operation_count !=
          ::rund::detail::task::Stat(
              state_->evidence.metrics,
              ::rund::detail::task::StatSlot::Operations) ||
      active.simulation_fingerprint != SimulationFingerprint(*state_)) {
    return fail(::rund::replay::Code::InputCaptureMutated);
  }
  if (state_->evidence.input_bytes == nullptr ||
      active.byte_offset != state_->evidence.input_byte_size ||
      active.byte_offset > state_->evidence.input_bytes->size() ||
      byte_count > state_->evidence.input_bytes->size() - active.byte_offset ||
      !CanAccountInput(*state_, byte_count)) {
    return fail(::rund::replay::Code::InputWriterCapacityExceeded);
  }
  ::rund::node::replay_detail::payload::Bytes bytes =
      ::rund::node::replay_detail::payload::Bytes::share(
          state_->evidence.input_bytes, active.byte_offset, byte_count);
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Live) {
    state_->evidence.input_byte_size += byte_count;
    AccountInput(*state_, byte_count);
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
        StoreReplayInput(*state_, binding,
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
    AccountInput(*state_, byte_count);
    return replay_detail::payload::ResolveResult{.code =
                                                     ::rund::replay::Code::Ok,
                                                 .sequence = binding.sequence,
                                                 .bytes = std::move(bytes)};
  } catch (...) {
    return fail(::rund::replay::Code::InputRecordFailed);
  }
}

replay_detail::payload::ResolveResult Scheduler::ReplayInput(
    const replay_detail::payload::InputBinding &binding) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const auto fail = [this](const ::rund::replay::Code code) {
    (void)PoisonInput(*state_, code);
    return replay_detail::payload::ResolveResult{.code = code};
  };
  const ::rund::replay::detail::scope::Mode mode = state_->plan.mode();
  if (mode != ::rund::replay::detail::scope::Mode::Replay &&
      mode != ::rund::replay::detail::scope::Mode::Scenario) {
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
  const std::size_t input_index = state_->identity.next_expected_replay_input;
  const auto &expected = *state_->plan.value.expected;
  const auto &expected_payloads = expected.payloads();
  const std::size_t record_index =
      expected_payloads.input_record_index(input_index);
  if (record_index >= expected_payloads.records().size()) {
    return fail(::rund::replay::Code::InputCorrupt);
  }
  const replay_detail::payload::StoredRecord &expected_input =
      expected_payloads.records()[record_index];
  const replay_detail::payload::InputBinding canonical{
      .source = binding.source,
      .schema = binding.schema,
      .sequence = expected_input.metadata.input_sequence,
  };
  const replay_detail::payload::InputSourceRange source_range{
      .event_offset = expected_input.metadata.source_event_offset,
      .event_count = expected_input.metadata.source_event_count,
      .payload_offset = expected_input.metadata.source_payload_offset,
      .payload_count = expected_input.metadata.source_payload_count,
      .hash = expected_input.metadata.source_hash,
  };
  if (source_range.event_offset != state_->identity.next_expected_host_event ||
      source_range.payload_offset !=
          state_->identity.next_expected_host_payload) {
    return fail(::rund::replay::Code::InputSourceOrderMismatch);
  }
  std::size_t event_offset = 0u;
  std::size_t event_count = 0u;
  std::size_t payload_offset = 0u;
  std::size_t payload_count = 0u;
  if (!ToSize(source_range.event_offset, event_offset) ||
      !ToSize(source_range.event_count, event_count) ||
      !ToSize(source_range.payload_offset, payload_offset) ||
      !ToSize(source_range.payload_count, payload_count) ||
      event_offset > expected.events().size() ||
      event_count > expected.events().size() - event_offset ||
      payload_offset > expected_payloads.host_record_count() ||
      payload_count > expected_payloads.host_record_count() - payload_offset) {
    return fail(::rund::replay::Code::InputSourceRangeInvalid);
  }
  const std::span<const ::rund::host::Event> expected_events{expected.events()};
  const std::span<const ::rund::host::Event> source_events =
      expected_events.subspan(event_offset, event_count);
  const std::optional<std::uint64_t> expected_source_hash =
      expected_payloads.SourceRangeHash(
          source_range.event_offset, source_events, source_range.payload_offset,
          source_range.payload_count);
  if (!expected_source_hash.has_value() ||
      *expected_source_hash != source_range.hash) {
    return fail(::rund::replay::Code::InputSourceHashMismatch);
  }
  std::vector<::rund::node::replay_detail::payload::Bytes> adopted_payloads{};
  if (mode == ::rund::replay::detail::scope::Mode::Scenario) {
    try {
      adopted_payloads.reserve(payload_count);
      for (std::size_t index = 0u; index < payload_count; ++index) {
        const std::size_t payload_record =
            expected_payloads.host_record_index(payload_offset + index);
        replay_detail::payload::ResolveResult resolved =
            expected_payloads.Resolve(payload_record);
        if (!resolved.ok()) {
          return fail(resolved.code);
        }
        adopted_payloads.push_back(std::move(resolved.bytes));
      }
    } catch (...) {
      return fail(::rund::replay::Code::InputSourceCorrupt);
    }
  }
  const replay_detail::InputPlan *const choices =
      mode == ::rund::replay::detail::scope::Mode::Scenario
          ? state_->plan.choices()
          : nullptr;
  const replay_detail::InputPatch *const patch =
      choices == nullptr ? nullptr
                         : choices->find(canonical.source, canonical.schema,
                                         canonical.sequence);
  std::size_t input_bytes = 0u;
  if (patch == nullptr) {
    if (!ToSize(expected_input.metadata.completed_bytes, input_bytes)) {
      return fail(::rund::replay::Code::InputCorrupt);
    }
  } else {
    input_bytes = patch->size;
  }
  if (!CanAccountInput(*state_, input_bytes)) {
    return fail(::rund::replay::Code::InputCapacityExceeded);
  }
  replay_detail::payload::ResolveResult result{
      .code = ::rund::replay::Code::Ok,
      .sequence = canonical.sequence,
  };
  try {
    if (patch == nullptr) {
      if (state_->evidence.input_bytes == nullptr ||
          state_->evidence.input_byte_size >
              state_->evidence.input_bytes->size() ||
          input_bytes > state_->evidence.input_bytes->size() -
                            state_->evidence.input_byte_size) {
        return fail(::rund::replay::Code::InputCapacityExceeded);
      }
      const std::span<std::byte> prepared{*state_->evidence.input_bytes};
      const std::span<std::byte> output =
          prepared.subspan(state_->evidence.input_byte_size, input_bytes);
      const replay_detail::payload::MatchResult read =
          expected_payloads.ReadInput(input_index, canonical, output);
      if (!read.ok()) {
        return fail(read.code);
      }
      result.bytes = ::rund::node::replay_detail::payload::Bytes::share(
          state_->evidence.input_bytes, state_->evidence.input_byte_size,
          input_bytes);
    } else {
      const replay_detail::payload::MatchResult checked =
          expected_payloads.CheckInput(input_index, canonical);
      if (!checked.ok()) {
        return fail(checked.code);
      }
      result.bytes = choices->bytes(*patch);
    }
  } catch (...) {
    return fail(::rund::replay::Code::InputCorrupt);
  }
  for (const ::rund::host::Event &event : source_events) {
    const HostEventCommitResult committed = CommitHostEvent(event);
    if (!committed.ok || committed.sequence != event.sequence) {
      return fail(::rund::replay::Code::InputSourceEventMismatch);
    }
    if (mode == ::rund::replay::detail::scope::Mode::Scenario) {
      ++state_->identity.next_expected_host_event;
    }
  }
  for (std::size_t index = 0u; index < payload_count; ++index) {
    const std::size_t payload_record =
        expected_payloads.host_record_index(payload_offset + index);
    const replay_detail::payload::StoredRecord &expected_payload =
        expected_payloads.records()[payload_record];
    if (mode == ::rund::replay::detail::scope::Mode::Scenario &&
        !state_->evidence.host_payload_store.Append(
            expected_payload.metadata.event_sequence,
            expected_payload.metadata.kind, adopted_payloads[index],
            replay_detail::payload::Capture::verify(
                adopted_payloads[index].span(),
                expected_payload.metadata.payload_hash))) {
      return fail(::rund::replay::Code::InputSourceRecordFailed);
    }
    ++state_->identity.next_expected_host_payload;
  }
  if (mode == ::rund::replay::detail::scope::Mode::Scenario) {
    ::rund::StableHash payload_hash{};
    payload_hash =
        expected_payloads.records()[record_index].metadata.payload_hash;
    if (patch != nullptr) {
      payload_hash = ::rund::StableHash{patch->payload_hash};
    }
    const replay_detail::payload::MatchResult stored =
        StoreReplayInput(*state_, canonical, source_range, result.bytes,
                         replay_detail::payload::Capture::verify(
                             result.bytes.span(), payload_hash));
    if (!stored.ok()) {
      return replay_detail::payload::ResolveResult{.code = stored.code};
    }
  }
  ++state_->identity.next_expected_replay_input;
  if (patch == nullptr) {
    state_->evidence.input_byte_size += input_bytes;
  }
  AccountInput(*state_, input_bytes);
  return result;
}

} // namespace rund::node
