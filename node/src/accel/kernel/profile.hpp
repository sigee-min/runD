#pragma once

#include <rund/compute/stats.hpp>

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

enum class PreparedPipelineStepClock : std::uint8_t {
  Unavailable,
  Device,
};

enum class PreparedPipelineStepTimingRelation : std::uint8_t {
  Unavailable,
  NonAdditive,
};

// Backend-owned evidence for one public declaration ordinal. Backends retain
// these rows beside the prepared native stream and expose only a span at the
// existing completion boundary.
struct PreparedPipelineStepEvidence final {
  std::uint64_t original_dispatch_count{};
  std::uint64_t final_dispatch_count{};
  std::uint64_t physical_dispatch_count{};
  std::uint64_t workgroup_count{};
  std::uint64_t work_item_count{};
  rund::compute::ControlStats control{};
  std::uint64_t duration_ns{};
  std::uint64_t timing_sample_count{};
  std::uint64_t work_sample_count{};
  PreparedPipelineStepClock clock{PreparedPipelineStepClock::Unavailable};
  PreparedPipelineStepTimingRelation relation{
      PreparedPipelineStepTimingRelation::Unavailable};
};

struct PreparedPipelineProfileEvidence final {
  std::span<const PreparedPipelineStepEvidence> steps{};
  std::uint64_t instrumentation_command_count{};
  std::uint64_t instrumentation_byte_count{};
  bool observed{};
};

} // namespace rund::node::accel::detail
