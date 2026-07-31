#pragma once

#include <node/runtime/replay/host/evidence.hpp>
#include <rund/replay/code.hpp>
#include <rund/session/trace.hpp>
#include <rund/task/observation.hpp>
#include <rund/task/stats.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace rund::node {

struct RuntimeReplayRecordDesc;

struct RuntimeReplayRecord {
  RuntimeReplayRecord() = default;
  RuntimeReplayRecord(RuntimeReplayRecordDesc desc,
                      std::uint64_t trusted_trace_hash);
  RuntimeReplayRecord(const RuntimeReplayRecord &) = delete;
  RuntimeReplayRecord &operator=(const RuntimeReplayRecord &) = delete;
  RuntimeReplayRecord(RuntimeReplayRecord &&other) noexcept;
  RuntimeReplayRecord &operator=(RuntimeReplayRecord &&other) noexcept;

  ::rund::replay::Code code = ::rund::replay::Code::SessionResultMissing;
  task::Stats tasks{};
  std::vector<task::Observation> observations{};
  HostReplayEvidence host{};
  ::rund::Trace trace{};
  std::uint64_t start_hash = 0u;
  std::uint64_t semantic_hash = 0u;
  std::uint64_t operation_hash = 0u;
  std::uint64_t observation_hash = 0u;
  std::uint64_t host_event_hash = 0u;
  std::uint64_t input_count = 0u;
  std::uint64_t input_hash = 0u;
  std::uint64_t transcript_hash = 0u;
  std::uint64_t trace_hash = 0u;
  std::uint64_t replay_hash = 0u;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ::rund::replay::error(code);
  }

private:
  static void reset_moved_from(RuntimeReplayRecord &record) noexcept;
};

} // namespace rund::node
