#pragma once

#include "prepare.hpp"
#include "resource.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] Result<PipelineScheduleSuccess>
plan_pipeline_publications(const PipelineBuildState &build,
                           std::span<const std::uint32_t> window_states,
                           PipelineScheduleResources &resources,
                           PipelineMemoryPlan &plan);

} // namespace rund::compute::detail
