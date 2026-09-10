#pragma once

#include "../local.hpp"
#include "../../prepare.hpp"

#include "../../../../../accel/kernel/recurrence.hpp"

#include <cstdint>
#include <vector>

namespace rund::compute::detail {

struct PipelineAccelPreparationDraft final {
  const DeviceOps *ops{};
  std::vector<std::uint64_t> entry_counts;
  std::vector<std::uint64_t> occurrence_counts;
  std::vector<std::uint64_t> window_counts;
  std::vector<std::uint64_t> nested_group_counts;
  std::vector<std::uint64_t> map_recurrence_group_counts;
  std::vector<std::uint64_t> map_recurrence_history_group_counts;
  std::vector<std::uint64_t> recurrence_hi;
  std::vector<std::uint64_t> recurrence_lo;
  std::vector<std::uint8_t> active_window_states;
  std::vector<node::accel::detail::PreparedKernelProgramRoute> routes;
  std::vector<
      std::vector<node::accel::detail::PreparedKernelProgramBindingIdentity>>
      route_program_bindings;
  std::uint64_t window_state_count{};
  std::uint64_t window_descriptor_state_count{};
  std::uint32_t route_copies{};
};

[[nodiscard]] Status collect_pipeline_accel_occurrences(
    const PipelineBuildState &build, PipelineMemoryPlan &plan,
    PipelineAccelPreparationDraft &draft);
[[nodiscard]] Status collect_pipeline_accel_routes(
    const PipelineBuildState &build, PipelineMemoryPlan &plan,
    PipelineAccelPreparationDraft &draft);
[[nodiscard]] Status finalize_pipeline_accel_preparation(
    const PipelineBuildState &build, PipelineMemoryPlan &plan,
    PipelineAccelPreparationDraft &draft);

} // namespace rund::compute::detail
