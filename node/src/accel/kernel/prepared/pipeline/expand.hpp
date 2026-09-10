#pragma once

#include "../../backend/pipeline/failure.hpp"
#include "../../recurrence.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

struct ExpandedPipeline final {
  std::vector<BackendBatchEntry> commands;
  std::vector<std::uint8_t> barriers;
  std::vector<BackendWindow> windows;
  std::vector<TileTransducer> transducers;
  std::vector<NestedAggregate> aggregates;
  std::uint32_t command_count{};
  bool compact_aggregate{};
  const char *reason = "accel_kernel_run_invalid";
  PreparedPipelineFailureContext failure{};
};

[[nodiscard]] bool expand_pipeline(
    std::span<const BackendBatchEntry> templates,
    std::span<const std::uint8_t> template_barriers,
    std::span<const BackendPublish> publications,
    std::span<const std::uint32_t> declared_steps,
    std::uint32_t declared_step_count, bool profile_steps,
    std::uint32_t direct_aggregate_commands, ExpandedPipeline &expanded);

} // namespace rund::node::accel::detail
