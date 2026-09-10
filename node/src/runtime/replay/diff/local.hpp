#pragma once

#include <node/runtime/replay/diff.hpp>

namespace rund::node::replay_diff_detail {

struct WindowRange final {
  std::size_t begin = 0u;
  std::size_t end = 0u;
};

[[nodiscard]] bool observation_equal(const task::Observation &expected,
                                     const task::Observation &actual) noexcept;
[[nodiscard]] bool
trace_record_equal(const ::rund::TraceRecord &expected,
                   const ::rund::TraceRecord &actual) noexcept;
[[nodiscard]] RuntimeReplayTraceRecordEvidence
capture_trace_record(const ::rund::TraceRecord &record);
[[nodiscard]] WindowRange bound_window(std::size_t size, std::size_t center,
                                       std::size_t context) noexcept;
void add_mismatch(
    std::vector<RuntimeReplayFieldMismatch> &mismatches, std::string_view field,
    std::uint64_t expected, std::uint64_t actual,
    ::rund::replay::Code code = ::rund::replay::Code::OutcomeMismatch);
void add_host_payload_diff(
    std::vector<RuntimeReplayFieldMismatch> &mismatches,
    const ::rund::node::replay_detail::payload::Archive &expected,
    const ::rund::node::replay_detail::payload::Archive &actual);

} // namespace rund::node::replay_diff_detail
