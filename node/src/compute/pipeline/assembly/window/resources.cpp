#include "internal.hpp"

#include <rund/compute/pipeline.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace rund::compute::detail {

bool materialize_window_resources(PipelineBuildState &build,
                                  const WindowAssemblyInput &input,
                                  const WindowAssemblyCounts &counts,
                                  WindowAssemblyResources &resources,
                                  PipelineBuildMutation &mutation) {
  const std::shared_ptr<ProgramState> &seed = *input.seed;
  const std::shared_ptr<ProgramState> &fold = *input.fold;
  const auto internal = [&](const Type type, const FixedFormat format,
                            const std::size_t count,
                            const ResourceAccess access) {
    const auto owner = static_cast<std::uint32_t>(build.internals.size());
    build.internals.push_back(
        PipelineInternal{.type = type, .format = format, .count = count});
    return bind(owner, build.internals.back(), access, 0u, count, true);
  };

  resources.outer_seed.reserve(counts.recurrent_count);
  resources.seed_external.reserve(counts.seed_external_count);
  resources.outer_first.reserve(counts.recurrent_count);
  resources.outer_second.reserve(counts.recurrent_count);
  resources.final.reserve(counts.recurrent_count);
  resources.window_tile.reserve(counts.window_count);
  resources.window_target.reserve(counts.window_count);
  for (std::size_t index = 0u; index < counts.recurrent_count; ++index) {
    PipelineBinding routed{};
    const Status status = route(build, input.inputs[index], routed);
    if (!status) {
      mutation.fail(status.reason());
      return false;
    }
    resources.outer_seed.push_back(std::move(routed));
    resources.outer_first.push_back(
        internal(fold->output_types[index], fold->output_formats[index],
                 fold->output_sizes[index], ResourceAccess::Write));
    resources.outer_second.push_back(
        internal(fold->output_types[index], fold->output_formats[index],
                 fold->output_sizes[index], ResourceAccess::Write));
    resources.final.push_back(bind(input.final_outputs[index]));
  }
  for (std::size_t index = 0u; index < counts.window_count; ++index) {
    const std::size_t output = counts.recurrent_count + index;
    resources.window_tile.push_back(
        internal(fold->output_types[output], fold->output_formats[output],
                 input.tile, ResourceAccess::Write));
    resources.window_target.push_back(bind(input.window_outputs[index]));
  }
  for (std::size_t index = 0u; index < counts.seed_external_count; ++index) {
    PipelineBinding routed{};
    const Status status =
        route(build, input.inputs[counts.recurrent_count + index], routed);
    if (!status) {
      mutation.fail(status.reason());
      return false;
    }
    resources.seed_external.push_back(std::move(routed));
  }

  resources.tile_first.reserve(counts.seed_output_count);
  resources.tile_second.reserve(counts.action_output_count);
  for (std::size_t index = 0u; index < counts.seed_output_count; ++index) {
    resources.tile_first.push_back(
        internal(seed->output_types[index], seed->output_formats[index],
                 seed->output_sizes[index], ResourceAccess::Write));
    if (index < counts.action_output_count) {
      resources.tile_second.push_back(internal(
          (*input.action)->output_types[index],
          (*input.action)->output_formats[index],
          (*input.action)->output_sizes[index], ResourceAccess::Write));
    }
  }

  resources.ordinal_owner = static_cast<std::uint32_t>(build.internals.size());
  build.internals.push_back(
      PipelineInternal{.type = Type::U32,
                       .count = counts.nested_shape.seed_count(),
                       .fill = PipelineFill::Ordinal});

  const auto nested =
      static_cast<std::uint16_t>(build.nested_windows.size() + 1u);
  if (nested == 0u || build.window_controls.size() >=
                          PipelineBuildWindowControlOrdinal::unassigned) {
    mutation.fail(Reason::PipelineCapacity);
    return false;
  }
  resources.nested = nested;
  resources.window_control = PipelineBuildWindowControlOrdinal{
      .value = static_cast<std::uint32_t>(build.window_controls.size()),
  };
  return true;
}

} // namespace rund::compute::detail
