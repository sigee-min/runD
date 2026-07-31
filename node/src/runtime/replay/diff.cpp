#include <node/runtime/replay/diff.hpp>

#include <node/runtime/replay/hash.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace rund::node {
namespace {

bool ObservationEqual(const task::Observation &expected,
                      const task::Observation &actual) noexcept {
  return expected.sequence == actual.sequence && expected.kind == actual.kind &&
         expected.task_id == actual.task_id &&
         expected.wait_id == actual.wait_id && expected.fd == actual.fd &&
         expected.interest == actual.interest &&
         expected.revents == actual.revents &&
         expected.deadline_ns == actual.deadline_ns &&
         expected.reason_code == actual.reason_code;
}

RuntimeReplayTraceRecordEvidence
CaptureTraceRecordEvidence(const ::rund::TraceRecord &record) {
  return RuntimeReplayTraceRecordEvidence{
      .event = static_cast<std::uint64_t>(record.event),
      .code = record.code,
      .snapshot_state = static_cast<std::uint64_t>(record.snapshot.state),
      .snapshot_active_compute_jobs = record.snapshot.active_compute_jobs,
      .snapshot_scope_active = record.snapshot.scope_active,
      .snapshot_code = record.snapshot.code,
      .sequence = record.sequence};
}

bool TraceRecordEqual(const ::rund::TraceRecord &expected,
                      const ::rund::TraceRecord &actual) noexcept {
  return expected.event == actual.event && expected.code == actual.code &&
         expected.snapshot.state == actual.snapshot.state &&
         expected.snapshot.active_compute_jobs ==
             actual.snapshot.active_compute_jobs &&
         expected.snapshot.scope_active == actual.snapshot.scope_active &&
         expected.snapshot.code == actual.snapshot.code &&
         expected.sequence == actual.sequence;
}

void AddMismatch(
    std::vector<RuntimeReplayFieldMismatch> &mismatches,
    const std::string_view field, const std::uint64_t expected,
    const std::uint64_t actual,
    const ::rund::replay::Code code = ::rund::replay::Code::OutcomeMismatch) {
  mismatches.push_back(
      {.code = code, .field = field, .expected = expected, .actual = actual});
}

void AddHostPayloadDiff(
    std::vector<RuntimeReplayFieldMismatch> &mismatches,
    const ::rund::node::replay_detail::payload::Archive &expected,
    const ::rund::node::replay_detail::payload::Archive &actual) {
  if (expected.records.size() != actual.records.size()) {
    AddMismatch(mismatches, "host.payload.count",
                static_cast<std::uint64_t>(expected.records.size()),
                static_cast<std::uint64_t>(actual.records.size()));
    return;
  }
  if (expected.payload_hash != actual.payload_hash) {
    AddMismatch(mismatches, "host.payload.hash", expected.payload_hash,
                actual.payload_hash);
  }
  for (std::size_t index = 0u; index < expected.records.size(); ++index) {
    const ::rund::node::replay_detail::payload::ArchiveRecord &left =
        expected.records[index];
    const ::rund::node::replay_detail::payload::ArchiveRecord &right =
        actual.records[index];
    if (left.metadata != right.metadata) {
      AddMismatch(mismatches, "host.payload.detail",
                  left.metadata.payload_hash.value,
                  right.metadata.payload_hash.value);
      return;
    }
  }
}

struct ReplayWindowRange {
  std::size_t begin = 0u;
  std::size_t end = 0u;
};

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
    const ::rund::node::replay_detail::payload::ArchiveRecord &record =
        archive.records[cursor++];
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
    const ::rund::node::replay_detail::payload::ArchiveRecord *const left =
        NextInput(expected, expected_cursor);
    const ::rund::node::replay_detail::payload::ArchiveRecord *const right =
        NextInput(actual, actual_cursor);
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

[[nodiscard]] ReplayWindowRange
BoundReplayWindow(const std::size_t size, const std::size_t center,
                  const std::size_t context) noexcept {
  if (size == 0u) {
    return ReplayWindowRange{};
  }
  const std::size_t bounded_center = center < size ? center : size - 1u;
  const std::size_t begin =
      bounded_center > context ? bounded_center - context : 0u;
  const std::size_t available_right = size - bounded_center - 1u;
  const std::size_t right =
      context < available_right ? context : available_right;
  return {.begin = begin, .end = bounded_center + right + 1u};
}

void AppendInputWindow(
    std::vector<ReplayInputPoint> &out,
    const ::rund::node::replay_detail::payload::Archive &archive,
    const std::size_t count, const std::size_t center,
    const std::size_t context) {
  const ReplayWindowRange range = BoundReplayWindow(count, center, context);
  if (range.begin == range.end) {
    return;
  }
  out.reserve(range.end - range.begin);
  std::size_t cursor = 0u;
  std::size_t input = 0u;
  while (
      const ::rund::node::replay_detail::payload::ArchiveRecord *const record =
          NextInput(archive, cursor)) {
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

void AppendObservationWindow(std::vector<task::Observation> &out,
                             const std::vector<task::Observation> &observations,
                             const std::size_t center,
                             const std::size_t context) {
  if (observations.empty()) {
    return;
  }
  const ReplayWindowRange range =
      BoundReplayWindow(observations.size(), center, context);
  out.insert(out.end(),
             observations.begin() + static_cast<std::ptrdiff_t>(range.begin),
             observations.begin() + static_cast<std::ptrdiff_t>(range.end));
}

void AppendTraceRecordWindow(std::vector<RuntimeReplayTraceRecordEvidence> &out,
                             const std::vector<::rund::TraceRecord> &records,
                             const std::size_t center,
                             const std::size_t context) {
  if (records.empty()) {
    return;
  }
  const ReplayWindowRange range =
      BoundReplayWindow(records.size(), center, context);
  for (std::size_t index = range.begin; index < range.end; ++index) {
    out.push_back(CaptureTraceRecordEvidence(records[index]));
  }
}

bool FindFirstObservationMismatch(
    const std::vector<task::Observation> &expected,
    const std::vector<task::Observation> &actual, std::size_t &index) noexcept {
  const std::size_t common =
      expected.size() < actual.size() ? expected.size() : actual.size();
  for (std::size_t current = 0u; current < common; ++current) {
    if (!ObservationEqual(expected[current], actual[current])) {
      index = current;
      return true;
    }
  }
  if (expected.size() != actual.size()) {
    index = common;
    return true;
  }
  return false;
}

bool FindFirstTraceRecordMismatch(
    const std::vector<::rund::TraceRecord> &expected,
    const std::vector<::rund::TraceRecord> &actual,
    std::size_t &index) noexcept {
  const std::size_t common =
      expected.size() < actual.size() ? expected.size() : actual.size();
  for (std::size_t current = 0u; current < common; ++current) {
    if (!TraceRecordEqual(expected[current], actual[current])) {
      index = current;
      return true;
    }
  }
  if (expected.size() != actual.size()) {
    index = common;
    return true;
  }
  return false;
}

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
  if (!check.ok()) {
    window.code = check.code;
    return window;
  }
  window.code = ::rund::replay::Code::Ok;
  return window;
}

RuntimeReplayDiff DiffRuntimeReplayRecords(const RuntimeReplayRecord &expected,
                                           const RuntimeReplayRecord &actual) {
  RuntimeReplayDiff diff{};
  if (expected.code != actual.code) {
    AddMismatch(diff.mismatches, "code", ::rund::replay::raw(expected.code),
                ::rund::replay::raw(actual.code));
  }
  if (expected.semantic_hash != actual.semantic_hash) {
    AddMismatch(diff.mismatches, "semantic_hash", expected.semantic_hash,
                actual.semantic_hash, ::rund::replay::Code::HashMismatch);
  }
  if (expected.operation_hash != actual.operation_hash) {
    AddMismatch(diff.mismatches, "operation_hash", expected.operation_hash,
                actual.operation_hash,
                ::rund::replay::Code::OperationHashMismatch);
  }
  if (expected.observation_hash != actual.observation_hash) {
    AddMismatch(diff.mismatches, "observation_hash", expected.observation_hash,
                actual.observation_hash,
                ::rund::replay::Code::ObservationHashMismatch);
  }
  if (expected.host_event_hash != actual.host_event_hash) {
    AddMismatch(diff.mismatches, "host_event_hash", expected.host_event_hash,
                actual.host_event_hash,
                ::rund::replay::Code::HostEventHashMismatch);
  }
  if (expected.input_count != actual.input_count) {
    AddMismatch(diff.mismatches, "input_count", expected.input_count,
                actual.input_count);
  }
  if (expected.input_hash != actual.input_hash) {
    AddMismatch(diff.mismatches, "input_hash", expected.input_hash,
                actual.input_hash, ::rund::replay::Code::InputHashMismatch);
  }
  if (expected.transcript_hash != actual.transcript_hash) {
    AddMismatch(diff.mismatches, "transcript_hash", expected.transcript_hash,
                actual.transcript_hash,
                ::rund::replay::Code::TranscriptHashMismatch);
  }
  if (expected.trace_hash != actual.trace_hash) {
    AddMismatch(diff.mismatches, "trace_hash", expected.trace_hash,
                actual.trace_hash, ::rund::replay::Code::TraceHashMismatch);
  }
  if (expected.replay_hash != actual.replay_hash) {
    AddMismatch(diff.mismatches, "replay_hash", expected.replay_hash,
                actual.replay_hash, ::rund::replay::Code::HashMismatch);
  }
  if (expected.observations.size() != actual.observations.size()) {
    AddMismatch(diff.mismatches, "observation.count",
                static_cast<std::uint64_t>(expected.observations.size()),
                static_cast<std::uint64_t>(actual.observations.size()));
  } else {
    std::size_t observation_index = 0u;
    if (FindFirstObservationMismatch(expected.observations, actual.observations,
                                     observation_index)) {
      AddMismatch(diff.mismatches, "observation.detail",
                  replay_detail::HashObservation(
                      expected.observations[observation_index]),
                  replay_detail::HashObservation(
                      actual.observations[observation_index]),
                  ::rund::replay::Code::CodecObservationDetailMismatch);
    }
  }
  const replay_detail::HostReplayFieldDiff host_diff =
      replay_detail::DiffHostReplayEvidence(expected.host, actual.host);
  if (host_diff.mismatch) {
    AddMismatch(diff.mismatches, host_diff.field, host_diff.expected,
                host_diff.actual,
                ::rund::replay::Code::CodecHostDetailMismatch);
  }
  AddHostPayloadDiff(diff.mismatches, expected.host.payload_archive,
                     actual.host.payload_archive);
  if (expected.trace.records.size() != actual.trace.records.size()) {
    AddMismatch(diff.mismatches, "trace.record.count",
                static_cast<std::uint64_t>(expected.trace.records.size()),
                static_cast<std::uint64_t>(actual.trace.records.size()));
  } else {
    std::size_t trace_record_index = 0u;
    if (FindFirstTraceRecordMismatch(
            expected.trace.records, actual.trace.records, trace_record_index)) {
      AddMismatch(diff.mismatches, "trace.record.detail",
                  replay_detail::HashTraceRecord(
                      expected.trace.records[trace_record_index]),
                  replay_detail::HashTraceRecord(
                      actual.trace.records[trace_record_index]),
                  ::rund::replay::Code::CodecTraceDetailMismatch);
    }
  }
  if (diff.mismatches.empty()) {
    diff.code = ::rund::replay::Code::Ok;
    return diff;
  }
  diff.code = diff.mismatches.front().code;
  return diff;
}

} // namespace rund::node
