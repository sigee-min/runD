#include "output.hpp"

#include "../../program/output.hpp"
#include "compare.hpp"

#include <cstddef>
#include <utility>

namespace rund::compute::detail {

Result<OutputProjection> project_outputs(const PipelineBuildStep &step) {
  const ProgramState &program = *step.program;
  const std::size_t physical = program.output_types.size();
  const std::size_t logical = output_count(program.output_aliases, physical);
  if (logical > PipelineLeafCapacity || physical > PipelineLeafCapacity) {
    return Result<OutputProjection>::fail(Reason::PipelineCapacity);
  }
  if (physical == 0u || step.outputs.size() != logical ||
      program.output_sizes.size() != physical ||
      program.output_formats.size() != physical) {
    return Result<OutputProjection>::fail(Reason::BindingCountMismatch);
  }
  OutputProjection projection{};
  projection.physical_sources.fill(OutputProjection::unassigned);
  projection.physical_count = physical;
  std::size_t assigned = 0u;
  for (std::size_t index = 0u; index < logical; ++index) {
    if (step.outputs[index].owner == PipelineBinding::external &&
        step.outputs[index].buffer == nullptr) {
      return Result<OutputProjection>::fail(Reason::BindingInvalid);
    }
    const std::size_t target = output_index(program.output_aliases, index);
    if (target >= physical) {
      return Result<OutputProjection>::fail(Reason::GraphBindingInvalid);
    }
    projection.logical_to_physical[index] = static_cast<std::uint32_t>(target);
    std::uint32_t &source = projection.physical_sources[target];
    if (source != OutputProjection::unassigned) {
      if (!same_view(step.outputs[source], step.outputs[index])) {
        return Result<OutputProjection>::fail(Reason::BindingAliasUnsupported);
      }
      continue;
    }
    source = static_cast<std::uint32_t>(index);
    ++assigned;
  }
  if (assigned != physical) {
    return Result<OutputProjection>::fail(Reason::BindingOutputMissing);
  }
  return Result<OutputProjection>::success(std::move(projection));
}

} // namespace rund::compute::detail
