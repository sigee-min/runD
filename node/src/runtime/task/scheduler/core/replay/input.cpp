#include "internal.hpp"

#include "../../host.hpp"
#include "../../state/model/join.hpp"
#include "../../state/model/task.hpp"
#include "../../state/model/timer.hpp"
#include "../../state/storage.hpp"

#include <rund/task/stats/slots.hpp>

#include <utility>
#include <vector>

namespace rund::node {

replay_detail::payload::ResolveResult Scheduler::ReplayInput(
    const replay_detail::payload::InputBinding &binding) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  const auto fail = [this](const ::rund::replay::Code code) {
    (void)scheduler_replay_detail::poison_input(*state_, code);
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
  if (!scheduler_replay_detail::to_size(source_range.event_offset,
                                        event_offset) ||
      !scheduler_replay_detail::to_size(source_range.event_count,
                                        event_count) ||
      !scheduler_replay_detail::to_size(source_range.payload_offset,
                                        payload_offset) ||
      !scheduler_replay_detail::to_size(source_range.payload_count,
                                        payload_count) ||
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
    if (!scheduler_replay_detail::to_size(
            expected_input.metadata.completed_bytes, input_bytes)) {
      return fail(::rund::replay::Code::InputCorrupt);
    }
  } else {
    input_bytes = patch->size;
  }
  if (!scheduler_replay_detail::can_account_input(*state_, input_bytes)) {
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
    if (!committed.ok() || committed.sequence() != event.sequence) {
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
        scheduler_replay_detail::store_replay_input(
            *state_, canonical, source_range, result.bytes,
            replay_detail::payload::Capture::verify(result.bytes.span(),
                                                    payload_hash));
    if (!stored.ok()) {
      return replay_detail::payload::ResolveResult{.code = stored.code};
    }
  }
  ++state_->identity.next_expected_replay_input;
  if (patch == nullptr) {
    state_->evidence.input_byte_size += input_bytes;
  }
  scheduler_replay_detail::account_input(*state_, input_bytes);
  return result;
}

} // namespace rund::node
