#include "local.hpp"

#include <cstddef>

namespace rund::node {

void AppendObservationWindow(std::vector<task::Observation> &out,
                             const std::vector<task::Observation> &observations,
                             const std::size_t center,
                             const std::size_t context) {
  if (observations.empty()) {
    return;
  }
  const replay_diff_detail::WindowRange range =
      replay_diff_detail::bound_window(observations.size(), center, context);
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
  const replay_diff_detail::WindowRange range =
      replay_diff_detail::bound_window(records.size(), center, context);
  for (std::size_t index = range.begin; index < range.end; ++index) {
    out.push_back(replay_diff_detail::capture_trace_record(records[index]));
  }
}

bool FindFirstObservationMismatch(
    const std::vector<task::Observation> &expected,
    const std::vector<task::Observation> &actual, std::size_t &index) noexcept {
  const std::size_t common =
      expected.size() < actual.size() ? expected.size() : actual.size();
  for (std::size_t current = 0u; current < common; ++current) {
    if (!replay_diff_detail::observation_equal(expected[current],
                                               actual[current])) {
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
    if (!replay_diff_detail::trace_record_equal(expected[current],
                                                actual[current])) {
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

} // namespace rund::node
