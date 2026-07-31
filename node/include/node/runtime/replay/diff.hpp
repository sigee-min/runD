#pragma once

#include <node/runtime/replay/host.hpp>
#include <node/runtime/replay/record.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace rund::node {

struct RuntimeReplayFieldMismatch {
  ::rund::replay::Code code = ::rund::replay::Code::OutcomeMismatch;
  std::string_view field{};
  std::uint64_t expected = 0u;
  std::uint64_t actual = 0u;
};

struct RuntimeReplayDiff {
  ::rund::replay::Code code = ::rund::replay::Code::CodecNotDiffed;
  std::vector<RuntimeReplayFieldMismatch> mismatches{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

struct RuntimeReplayTraceRecordEvidence {
  std::uint64_t event = 0u;
  ::rund::TraceCode code{};
  std::uint64_t snapshot_state = 0u;
  std::uint32_t snapshot_active_compute_jobs = 0u;
  bool snapshot_scope_active = false;
  ::rund::ReasonCode snapshot_code = ::rund::ReasonCode::Ok;
  std::uint64_t sequence = 0u;
};

struct ReplayInputPoint {
  std::size_t index = 0u;
  std::uint64_t source = 0u;
  std::uint64_t schema = 0u;
  std::uint64_t sequence = 0u;
  std::uint64_t size = 0u;
  std::uint64_t hash = 0u;
};

struct RuntimeReplayMismatchWindow {
  ::rund::replay::Code code = ::rund::replay::Code::CodecNotMinimized;
  bool has_observation_mismatch = false;
  std::size_t observation_index = 0u;
  std::vector<task::Observation> expected_observations{};
  std::vector<task::Observation> actual_observations{};
  bool has_host_mismatch = false;
  std::size_t host_event_index = 0u;
  std::vector<::rund::host::Event> expected_host_events{};
  std::vector<::rund::host::Event> actual_host_events{};
  bool has_input_mismatch = false;
  std::size_t input_index = 0u;
  std::vector<ReplayInputPoint> expected_inputs{};
  std::vector<ReplayInputPoint> actual_inputs{};
  bool has_trace_mismatch = false;
  std::size_t trace_record_index = 0u;
  std::vector<RuntimeReplayTraceRecordEvidence> expected_trace_records{};
  std::vector<RuntimeReplayTraceRecordEvidence> actual_trace_records{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }
};

void AppendObservationWindow(std::vector<task::Observation> &out,
                             const std::vector<task::Observation> &observations,
                             std::size_t center, std::size_t context);

void AppendTraceRecordWindow(std::vector<RuntimeReplayTraceRecordEvidence> &out,
                             const std::vector<::rund::TraceRecord> &records,
                             std::size_t center, std::size_t context);

[[nodiscard]] bool
FindFirstObservationMismatch(const std::vector<task::Observation> &expected,
                             const std::vector<task::Observation> &actual,
                             std::size_t &index) noexcept;

[[nodiscard]] bool
FindFirstTraceRecordMismatch(const std::vector<::rund::TraceRecord> &expected,
                             const std::vector<::rund::TraceRecord> &actual,
                             std::size_t &index) noexcept;

[[nodiscard]] RuntimeReplayMismatchWindow
MinimizeRuntimeReplayMismatch(const RuntimeReplayRecord &expected,
                              const RuntimeReplayRecord &actual,
                              std::size_t context = 2u);

[[nodiscard]] RuntimeReplayDiff
DiffRuntimeReplayRecords(const RuntimeReplayRecord &expected,
                         const RuntimeReplayRecord &actual);

} // namespace rund::node
