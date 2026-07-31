#pragma once

#include <rund/host/event.hpp>
#include <rund/session/memory.hpp>
#include <node/runtime/replay/host/archive.hpp>
#include <rund/task/observation.hpp>
#include <rund/task/stats.hpp>

#include <cstdint>
#include <vector>

namespace rund::node {

struct ScopeEvidence final {
  task::Stats tasks{};
  ::rund::PreparedMemory memory{};
  std::vector<task::Observation> observations{};
  std::vector<::rund::host::Event> events{};
  ::rund::node::replay_detail::payload::Archive payloads{};
  std::uint64_t input_rows = 0u;
  std::uint64_t input_bytes = 0u;
  std::uint64_t ready_capacity = 0u;
};

} // namespace rund::node
