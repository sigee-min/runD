#include "local.hpp"

namespace rund::node::replay_diff_detail {

bool observation_equal(const task::Observation &expected,
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
capture_trace_record(const ::rund::TraceRecord &record) {
  return RuntimeReplayTraceRecordEvidence{
      .event = static_cast<std::uint64_t>(record.event),
      .code = record.code,
      .snapshot_state = static_cast<std::uint64_t>(record.snapshot.state),
      .snapshot_active_compute_jobs = record.snapshot.active_compute_jobs,
      .snapshot_scope_active = record.snapshot.scope_active,
      .snapshot_code = record.snapshot.code,
      .sequence = record.sequence};
}

bool trace_record_equal(const ::rund::TraceRecord &expected,
                        const ::rund::TraceRecord &actual) noexcept {
  return expected.event == actual.event && expected.code == actual.code &&
         expected.snapshot.state == actual.snapshot.state &&
         expected.snapshot.active_compute_jobs ==
             actual.snapshot.active_compute_jobs &&
         expected.snapshot.scope_active == actual.snapshot.scope_active &&
         expected.snapshot.code == actual.snapshot.code &&
         expected.sequence == actual.sequence;
}

WindowRange bound_window(const std::size_t size, const std::size_t center,
                         const std::size_t context) noexcept {
  if (size == 0u) {
    return WindowRange{};
  }
  const std::size_t bounded_center = center < size ? center : size - 1u;
  const std::size_t begin =
      bounded_center > context ? bounded_center - context : 0u;
  const std::size_t available_right = size - bounded_center - 1u;
  const std::size_t right =
      context < available_right ? context : available_right;
  return {.begin = begin, .end = bounded_center + right + 1u};
}

void add_mismatch(std::vector<RuntimeReplayFieldMismatch> &mismatches,
                  const std::string_view field, const std::uint64_t expected,
                  const std::uint64_t actual, const ::rund::replay::Code code) {
  mismatches.push_back(
      {.code = code, .field = field, .expected = expected, .actual = actual});
}

void add_host_payload_diff(
    std::vector<RuntimeReplayFieldMismatch> &mismatches,
    const ::rund::node::replay_detail::payload::Archive &expected,
    const ::rund::node::replay_detail::payload::Archive &actual) {
  if (expected.records.size() != actual.records.size()) {
    add_mismatch(mismatches, "host.payload.count",
                 static_cast<std::uint64_t>(expected.records.size()),
                 static_cast<std::uint64_t>(actual.records.size()));
    return;
  }
  if (expected.payload_hash != actual.payload_hash) {
    add_mismatch(mismatches, "host.payload.hash", expected.payload_hash,
                 actual.payload_hash);
  }
  for (std::size_t index = 0u; index < expected.records.size(); ++index) {
    const auto &left = expected.records[index];
    const auto &right = actual.records[index];
    if (left.metadata != right.metadata) {
      add_mismatch(mismatches, "host.payload.detail",
                   left.metadata.payload_hash.value,
                   right.metadata.payload_hash.value);
      return;
    }
  }
}

} // namespace rund::node::replay_diff_detail
