#pragma once

#include "../publication.hpp"

#include <optional>
#include <span>

namespace rund::compute::detail::pipeline_publication_detail {

struct SealedPublicationOutput final {
  const PipelineResolvedOutputPlan *output{};
  PipelinePhysicalOutputOrdinal physical{};
};

[[nodiscard]] Location
publication_location(const PipelineBuildState &,
                     PipelinePublicationStepOrdinal) noexcept;

[[nodiscard]] Result<PipelinePublicationViewPlan>
seal_publication_view(PipelineScheduleResources &, const PipelineBinding &,
                      std::optional<ResourceAccess>, std::uint32_t, Location);

[[nodiscard]] Result<SealedPublicationOutput>
resolve_publication_output(const PipelineBuildState &,
                           std::span<const PipelineStepResourcePlan>,
                           PipelineBuildOutputCoordinate, Location);

[[nodiscard]] Result<PipelineScheduleSuccess>
plan_window_controls(const PipelineBuildState &, std::span<const std::uint32_t>,
                     PipelineScheduleResources &, PipelineMemoryPlan &);

[[nodiscard]] Result<PipelinePublicationPlan> plan_window_publication(
    const PipelineBuildState &, const PipelineBuildWindowPublication &,
    std::span<const std::uint32_t>, std::span<const PipelineStepResourcePlan>,
    std::span<const PipelineWindowControl>, PipelineScheduleResources &);

[[nodiscard]] Result<PipelinePublicationPlan> plan_terminal_publication(
    const PipelineBuildState &, const PipelineBuildTerminalPublication &,
    std::span<const std::uint32_t>, std::span<const PipelineStepResourcePlan>,
    std::span<const PipelineWindowControl>, PipelineScheduleResources &);

} // namespace rund::compute::detail::pipeline_publication_detail
