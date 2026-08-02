#pragma once

#include <rund/compute/status.hpp>

#include <span>

namespace rund::compute::detail {

struct DeviceState;
struct PipelineBuildStep;
struct PipelineMemoryPlan;
struct ProgramState;

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
                      std::span<const ProgramState *const> programs,
                      PipelineMemoryPlan &plan);

} // namespace rund::compute::detail
