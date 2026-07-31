#pragma once

#include <rund/compute/compile.hpp>
#include <rund/replay/storage.hpp>
#include <rund/session/scheduler.hpp>
#include <rund/telemetry/sink.hpp>

#include <cstddef>
#include <cstdint>

namespace rund {

struct ReplayConfig {
  std::uint32_t input_capacity = 1024u;
  ::rund::replay::Storage storage{};
  ::rund::replay::Diagnostic diagnostic{};
};

struct SessionConfig {
  std::uint64_t id = 1u;
  std::uint32_t workers = 0u;
  bool require_verified_numa = false;
  bool require_verified_affinity = false;
  bool require_verified_worker_capacity = false;
  telemetry::Sink telemetry{};
  std::size_t trace_capacity = 1024u;
  SchedulerConfig scheduler{};
  compute::Compile compile{};
  ReplayConfig replay{};
  std::uint64_t random_seed = 0xC2B2AE3D27D4EB4Full;
};

} // namespace rund
