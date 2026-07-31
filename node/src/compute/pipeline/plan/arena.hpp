#pragma once

#include <rund/compute/status.hpp>

#include <span>

namespace rund::compute::detail {

struct DeviceState;
struct PipelineBuildStep;
struct PipelineMemoryPlan;

[[nodiscard]] Status
plan_pipeline_arena(const DeviceState &device,
                    std::span<const PipelineBuildStep> steps,
                    PipelineMemoryPlan &plan) noexcept;
[[nodiscard]] Status
plan_pipeline_views(const DeviceState &device,
                    std::span<const PipelineBuildStep> steps,
                    PipelineMemoryPlan &plan) noexcept;
[[nodiscard]] Status
plan_pipeline_scratch(const DeviceState &device,
                      std::span<const PipelineBuildStep> steps,
                      PipelineMemoryPlan &plan);

} // namespace rund::compute::detail
