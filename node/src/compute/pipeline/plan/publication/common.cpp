#include "../../state/assembly.hpp"
#include "internal.hpp"

#include "../../output.hpp"

namespace rund::compute::detail::pipeline_publication_detail {

Location
publication_location(const PipelineBuildState &build,
                     const PipelinePublicationStepOrdinal step) noexcept {
  Location location{};
  if (step.value >= build.steps.size()) {
    return location;
  }
  const PipelineBuildStep &authored = build.steps[step.value];
  location.step = authored.logical_step;
  location.iteration = authored.iteration;
  location.nested_phase = pipeline_nested_phase(authored.route);
  return location;
}

Result<PipelinePublicationViewPlan>
seal_publication_view(PipelineScheduleResources &resources,
                      const PipelineBinding &binding,
                      const std::optional<ResourceAccess> expected_access,
                      const std::uint32_t usage, const Location location) {
  return resources.publication_view(binding, binding.type, binding.count,
                                    binding.format, expected_access, usage,
                                    location);
}

Result<SealedPublicationOutput> resolve_publication_output(
    const PipelineBuildState &build,
    const std::span<const PipelineStepResourcePlan> steps,
    const PipelineBuildOutputCoordinate coordinate, const Location location) {
  auto projected = resolve_build_output(build, coordinate);
  if (!projected || coordinate.step.value >= steps.size()) {
    return Result<SealedPublicationOutput>::fail(
        projected ? Reason::PipelineInvalid : projected.reason(), location);
  }
  const PipelineStepResourcePlan &step = steps[coordinate.step.value];
  if (projected->physical.value >= step.physical_sources.size()) {
    return Result<SealedPublicationOutput>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  const std::uint32_t source = step.physical_sources[projected->physical.value];
  if (source >= step.outputs.size() ||
      step.outputs[source].physical != projected->physical.value) {
    return Result<SealedPublicationOutput>::fail(Reason::PipelineInvalid,
                                                 location);
  }
  return Result<SealedPublicationOutput>::success({
      .output = &step.outputs[source],
      .physical = projected->physical,
  });
}

} // namespace rund::compute::detail::pipeline_publication_detail
