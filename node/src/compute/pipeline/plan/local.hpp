#pragma once

#include "../state.hpp"

#include <memory>
#include <span>
#include <vector>

namespace rund::compute::detail {

struct PipelineMemorySet final {
  std::vector<std::shared_ptr<JobWorkspace>> steps;
  std::vector<std::shared_ptr<BufferState>> buffers;
  std::vector<std::shared_ptr<BufferState>> prepared;
  std::shared_ptr<JobArena> arena;
};

[[nodiscard]] Result<std::shared_ptr<const PipelineMemoryPlan>>
plan_memory(const PipelineBuildState &build);
[[nodiscard]] Status materialize_pipeline(PipelineBuildState &build);
[[nodiscard]] Result<PipelineMemorySet>
make_pipeline_memory(const std::shared_ptr<DeviceState> &device,
                     std::span<const PipelineBuildStep> steps,
                     const PipelineMemoryPlan &plan);
[[nodiscard]] Status prepare_backend(PipelineState &state) noexcept;

} // namespace rund::compute::detail
