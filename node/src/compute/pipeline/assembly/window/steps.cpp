#include "internal.hpp"

#include <rund/compute/pipeline.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace rund::compute::detail {

bool emit_window_steps(PipelineBuildState &build,
                       const WindowAssemblyInput &input,
                       const WindowAssemblyCounts &counts,
                       const WindowAssemblyResources &resources,
                       PipelineBuildMutation &mutation) {
  const std::shared_ptr<ProgramState> &seed = *input.seed;
  const std::shared_ptr<ProgramState> &action = *input.action;
  const std::shared_ptr<ProgramState> &fold = *input.fold;
  const ResourceView &resident = *input.resident;
  const node::accel::detail::NestedTemplateShape &nested_shape =
      counts.nested_shape;
  const auto apply_nested_projection =
      [&](const std::size_t template_index,
          const node::accel::detail::NestedTemplatePhase expected_phase,
          PipelineBuildStep &step,
          node::accel::detail::NestedTemplateRouteProjection &projection) {
        if (!nested_shape.project(template_index, projection) ||
            projection.phase != expected_phase) {
          return false;
        }
        step.iteration = projection.iteration;
        step.iteration_bound = projection.bound;
        step.route = pipeline_route(projection.phase);
        return true;
      };

  const std::size_t seed_first = nested_shape.seed_first();
  if (seed_first != build.steps.size()) {
    mutation.fail(Reason::PipelineInvalid);
    return false;
  }
  for (std::size_t template_index = seed_first;
       template_index < nested_shape.action_first(); ++template_index) {
    PipelineBuildStep step{};
    node::accel::detail::NestedTemplateRouteProjection projection{};
    if (!apply_nested_projection(template_index,
                                 node::accel::detail::NestedTemplatePhase::Seed,
                                 step, projection)) {
      mutation.fail(Reason::PipelineInvalid);
      return false;
    }
    step.program = seed;
    step.logical_step = static_cast<std::uint32_t>(build.logical_step_count);
    step.window_control = resources.window_control;
    step.nested = resources.nested;
    step.inputs.reserve(seed->input_types.size());
    step.outputs.reserve(counts.seed_output_count);
    for (PipelineBinding binding : resources.seed_external) {
      binding.access = ResourceAccess::Read;
      step.inputs.push_back(std::move(binding));
    }
    step.inputs.push_back(bind(resident));
    step.inputs.push_back(
        bind(resources.ordinal_owner, build.internals[resources.ordinal_owner],
             ResourceAccess::Read, projection.outer_iteration, 1u, true));
    for (PipelineBinding binding : resources.tile_first) {
      binding.access = ResourceAccess::Write;
      binding.hidden = true;
      step.outputs.push_back(std::move(binding));
    }
    build.steps.push_back(std::move(step));
  }

  const std::size_t action_first = nested_shape.action_first();
  if (action_first != build.steps.size()) {
    mutation.fail(Reason::PipelineInvalid);
    return false;
  }
  for (std::size_t template_index = action_first;
       template_index < nested_shape.fold_first(); ++template_index) {
    PipelineBuildStep step{};
    node::accel::detail::NestedTemplateRouteProjection projection{};
    if (!apply_nested_projection(
            template_index, node::accel::detail::NestedTemplatePhase::Action,
            step, projection)) {
      mutation.fail(Reason::PipelineInvalid);
      return false;
    }
    const bool even = (projection.inner_iteration & 1u) == 0u;
    step.program = action;
    step.logical_step = static_cast<std::uint32_t>(build.logical_step_count);
    step.window_control = resources.window_control;
    step.nested = resources.nested;
    step.inputs.reserve(counts.seed_output_count);
    step.outputs.reserve(counts.action_output_count);
    for (std::size_t index = 0u; index < counts.seed_output_count; ++index) {
      PipelineBinding binding = index < counts.action_output_count && !even
                                    ? resources.tile_second[index]
                                    : resources.tile_first[index];
      binding.access = ResourceAccess::Read;
      step.inputs.push_back(std::move(binding));
    }
    for (std::size_t index = 0u; index < counts.action_output_count; ++index) {
      PipelineBinding binding =
          even ? resources.tile_second[index] : resources.tile_first[index];
      binding.access = ResourceAccess::Write;
      binding.hidden = true;
      step.outputs.push_back(std::move(binding));
    }
    build.steps.push_back(std::move(step));
  }

  std::vector<PipelineBinding> tile_final = resources.tile_first;
  if ((nested_shape.inner_bound() & 1u) != 0u) {
    std::copy(resources.tile_second.begin(), resources.tile_second.end(),
              tile_final.begin());
  }
  const std::size_t fold_first = nested_shape.fold_first();
  if (fold_first != build.steps.size()) {
    mutation.fail(Reason::PipelineInvalid);
    return false;
  }
  for (std::size_t template_index = fold_first;
       template_index < nested_shape.end(); ++template_index) {
    PipelineBuildStep step{};
    node::accel::detail::NestedTemplateRouteProjection projection{};
    if (!apply_nested_projection(template_index,
                                 node::accel::detail::NestedTemplatePhase::Fold,
                                 step, projection)) {
      mutation.fail(Reason::PipelineInvalid);
      return false;
    }
    const std::uint32_t route_index = projection.route;
    const std::span<const PipelineBinding> current =
        route_index == 0u
            ? std::span<const PipelineBinding>{resources.outer_seed}
            : (route_index == 1u
                   ? std::span<const PipelineBinding>{resources.outer_first}
                   : std::span<const PipelineBinding>{resources.outer_second});
    const std::span<const PipelineBinding> destination =
        route_index == 1u
            ? std::span<const PipelineBinding>{resources.outer_second}
            : std::span<const PipelineBinding>{resources.outer_first};
    step.program = fold;
    step.logical_step = static_cast<std::uint32_t>(build.logical_step_count);
    step.window_control = resources.window_control;
    step.nested = resources.nested;
    step.inputs.reserve(fold->input_types.size());
    step.outputs.reserve(counts.fold_output_count);
    for (PipelineBinding binding : current) {
      binding.access = ResourceAccess::Read;
      step.inputs.push_back(std::move(binding));
    }
    for (PipelineBinding binding : tile_final) {
      binding.access = ResourceAccess::Read;
      step.inputs.push_back(std::move(binding));
    }
    for (PipelineBinding binding : destination) {
      binding.access = ResourceAccess::Write;
      binding.hidden = true;
      step.outputs.push_back(std::move(binding));
    }
    for (PipelineBinding binding : resources.window_tile) {
      binding.access = ResourceAccess::Write;
      binding.hidden = true;
      step.outputs.push_back(std::move(binding));
    }
    build.steps.push_back(std::move(step));
  }
  return true;
}

} // namespace rund::compute::detail
