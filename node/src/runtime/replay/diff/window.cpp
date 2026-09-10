#include "local.hpp"

#include <algorithm>

namespace rund::node {
namespace {

struct InputMismatch final {
  bool found = false;
  std::size_t index = 0u;
  std::size_t expected_count = 0u;
  std::size_t actual_count = 0u;
};

[[nodiscard]] const ::rund::node::replay_detail::payload::ArchiveRecord *
NextInput(const ::rund::node::replay_detail::payload::Archive &archive,
          std::size_t &cursor) noexcept {
  while (cursor < archive.records.size()) {
    const auto &record = archive.records[cursor++];
    if (record.metadata.role ==
        ::rund::node::replay_detail::payload::Role::Input) {
      return &record;
    }
  }
  return nullptr;
}

[[nodiscard]] bool
SameInput(const ::rund::node::replay_detail::payload::ArchiveRecord &expected,
          const ::rund::node::replay_detail::payload::ArchiveRecord
              &actual) noexcept {
  return expected.metadata.input_source == actual.metadata.input_source &&
         expected.metadata.input_schema == actual.metadata.input_schema &&
         expected.metadata.input_sequence == actual.metadata.input_sequence &&
         expected.metadata.completed_bytes == actual.metadata.completed_bytes &&
         expected.metadata.payload_hash.value ==
             actual.metadata.payload_hash.value;
}

[[nodiscard]] InputMismatch FindInputMismatch(
    const ::rund::node::replay_detail::payload::Archive &expected,
    const ::rund::node::replay_detail::payload::Archive &actual) noexcept {
  InputMismatch mismatch{};
  std::size_t expected_cursor = 0u;
  std::size_t actual_cursor = 0u;
  while (true) {
    const auto *const left = NextInput(expected, expected_cursor);
    const auto *const right = NextInput(actual, actual_cursor);
    if (left == nullptr && right == nullptr) {
      return mismatch;
    }
    if (!mismatch.found &&
        (left == nullptr || right == nullptr || !SameInput(*left, *right))) {
      mismatch.found = true;
      mismatch.index = std::min(mismatch.expected_count, mismatch.actual_count);
    }
    mismatch.expected_count += left != nullptr ? 1u : 0u;
    mismatch.actual_count += right != nullptr ? 1u : 0u;
  }
}

[[nodiscard]] ReplayInputPoint
Point(const ::rund::node::replay_detail::payload::ArchiveRecord &record,
      const std::size_t index) noexcept {
  return ReplayInputPoint{.index = index,
                          .source = record.metadata.input_source,
                          .schema = record.metadata.input_schema,
                          .sequence = record.metadata.input_sequence,
                          .size = record.metadata.completed_bytes,
                          .hash = record.metadata.payload_hash.value};
}

void AppendInputWindow(
    std::vector<ReplayInputPoint> &out,
    const ::rund::node::replay_detail::payload::Archive &archive,
    const std::size_t count, const std::size_t center,
    const std::size_t context) {
  const replay_diff_detail::WindowRange range =
      replay_diff_detail::bound_window(count, center, context);
  if (range.begin == range.end) {
    return;
  }
  out.reserve(range.end - range.begin);
  std::size_t cursor = 0u;
  std::size_t input = 0u;
  while (const auto *const record = NextInput(archive, cursor)) {
    if (input >= range.end) {
      break;
    }
    if (input >= range.begin) {
      out.push_back(Point(*record, input));
    }
    ++input;
  }
}

} // namespace

RuntimeReplayMismatchWindow
MinimizeRuntimeReplayMismatch(const RuntimeReplayRecord &expected,
                              const RuntimeReplayRecord &actual,
                              const std::size_t context) {
  RuntimeReplayMismatchWindow window{};
  const InputMismatch input =
      expected.input_count != actual.input_count ||
              expected.input_hash != actual.input_hash
          ? FindInputMismatch(expected.host.payload_archive,
                              actual.host.payload_archive)
          : InputMismatch{};
  if (input.found) {
    window.code = ::rund::replay::Code::InputHashMismatch;
    window.has_input_mismatch = true;
    window.input_index = input.index;
    AppendInputWindow(window.expected_inputs, expected.host.payload_archive,
                      input.expected_count, input.index, context);
    AppendInputWindow(window.actual_inputs, actual.host.payload_archive,
                      input.actual_count, input.index, context);
    return window;
  }
  std::size_t observation_index = 0u;
  if (FindFirstObservationMismatch(expected.observations, actual.observations,
                                   observation_index)) {
    window.code = ::rund::replay::Code::CodecObservationDetailMismatch;
    window.has_observation_mismatch = true;
    window.observation_index = observation_index;
    AppendObservationWindow(window.expected_observations, expected.observations,
                            observation_index, context);
    AppendObservationWindow(window.actual_observations, actual.observations,
                            observation_index, context);
    return window;
  }
  std::size_t host_event_index = 0u;
  if (replay_detail::FindFirstHostReplayEventMismatch(
          expected.host.events, actual.host.events, host_event_index)) {
    window.code = ::rund::replay::Code::CodecHostDetailMismatch;
    window.has_host_mismatch = true;
    window.host_event_index = host_event_index;
    replay_detail::AppendHostReplayWindow(window.expected_host_events,
                                          expected.host.events,
                                          host_event_index, context);
    replay_detail::AppendHostReplayWindow(window.actual_host_events,
                                          actual.host.events, host_event_index,
                                          context);
    return window;
  }
  std::size_t trace_record_index = 0u;
  if (FindFirstTraceRecordMismatch(expected.trace.records, actual.trace.records,
                                   trace_record_index)) {
    window.code = ::rund::replay::Code::CodecTraceDetailMismatch;
    window.has_trace_mismatch = true;
    window.trace_record_index = trace_record_index;
    AppendTraceRecordWindow(window.expected_trace_records,
                            expected.trace.records, trace_record_index,
                            context);
    AppendTraceRecordWindow(window.actual_trace_records, actual.trace.records,
                            trace_record_index, context);
    return window;
  }
  const RuntimeReplayCheck check = check_runtime_replay(expected, actual);
  window.code = check.ok() ? ::rund::replay::Code::Ok : check.code;
  return window;
}

} // namespace rund::node
