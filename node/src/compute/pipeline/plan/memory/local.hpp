#pragma once

#include "../local.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::compute::detail {

struct PipelineBufferCommitment final {
  std::uint64_t logical{};
  std::uint64_t committed{};
};

struct PipelinePlanningModel final {
  std::vector<const ProgramState *> unique_programs;
};

struct PipelinePlanningTotals final {
  std::uint64_t nested_commands{};
  std::uint64_t logical_workspace{};
  std::uint64_t live_workspace{};
  std::size_t internal_resource_count{};
};

[[nodiscard]] Result<PipelinePlanningModel>
initialize_pipeline_plan(const PipelineBuildState &build,
                         PipelineMemoryPlan &plan);
[[nodiscard]] Result<PipelinePlanningTotals>
project_pipeline_workload(const PipelineBuildState &build,
                          PipelineMemoryPlan &plan);
[[nodiscard]] Status finalize_pipeline_plan(
    const PipelineBuildState &build, PipelineMemoryPlan &plan,
    const PipelinePlanningModel &model, const PipelinePlanningTotals &workload);

[[nodiscard]] Result<PipelineBufferCommitment>
plan_buffer_commitment(const DeviceState &device,
                       const PipelineMemoryPlan &plan) noexcept;
[[nodiscard]] Status
validate_buffer_commitment(const DeviceState &device,
                           const BufferState &buffer) noexcept;
[[nodiscard]] Status
seal_pipeline_workspace_routes(const PipelineBuildState &build,
                               PipelineMemoryPlan &plan);
[[nodiscard]] Status plan_pipeline_cpu_views(const PipelineBuildState &build,
                                             PipelineMemoryPlan &plan);
[[nodiscard]] Status
plan_pipeline_cpu_prepared_storage(const PipelineBuildState &build,
                                   PipelineMemoryPlan &plan);
[[nodiscard]] Status
plan_pipeline_accel_preparation(const PipelineBuildState &build,
                                PipelineMemoryPlan &plan);
[[nodiscard]] Status
plan_pipeline_host_preparation(const PipelineBuildState &build,
                               PipelineMemoryPlan &plan);

} // namespace rund::compute::detail
