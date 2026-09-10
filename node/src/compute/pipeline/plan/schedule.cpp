#include "../state/assembly.hpp"
#include "publication.hpp"
#include "schedule/internal.hpp"

#include <rund/compute/resource/plan.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail {

Result<PipelineScheduleSuccess>
plan_pipeline_schedule(const PipelineBuildState &build,
                       PipelineMemoryPlan &plan) {
  if (build.steps.empty() ||
      build.steps.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
  }

  PipelineScheduleResources resources(build);
  plan.step_resources.clear();
  plan.step_resources.resize(build.steps.size());
  plan.window_states.assign(build.steps.size(), PipelineResourceUnassigned);
  std::vector<std::uint32_t> &window_states = plan.window_states;
  if (build.window_controls.size() >=
      PipelineBuildWindowControlOrdinal::unassigned) {
    return Result<PipelineScheduleSuccess>::fail(Reason::PipelineCapacity);
  }
  std::vector<bool> referenced_window_controls(build.window_controls.size(),
                                               false);
  for (std::size_t index = 0u; index < build.steps.size(); ++index) {
    const PipelineBuildStep &step = build.steps[index];
    if (step.window_control.value ==
        PipelineBuildWindowControlOrdinal::unassigned) {
      if (step.nested != 0u) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      continue;
    }
    if (step.window_control.value >= build.window_controls.size()) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    const PipelineBuildWindowControl &control =
        build.window_controls[step.window_control.value];
    if (step.nested != control.nested) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    if (step.nested != 0u) {
      const std::size_t nested_index = step.nested - 1u;
      if (nested_index >= build.nested_windows.size()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      const PipelineBuildNestedWindow &nested =
          build.nested_windows[nested_index];
      if (index < nested.shape.first() || index >= nested.shape.end()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
    }
    window_states[index] = step.window_control.value;
    referenced_window_controls[step.window_control.value] = true;
  }
  if (std::find(referenced_window_controls.begin(),
                referenced_window_controls.end(),
                false) != referenced_window_controls.end()) {
    return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
  }
  for (std::size_t state = 0u; state < build.window_controls.size(); ++state) {
    const PipelineBuildWindowControl &control = build.window_controls[state];
    if (control.nested != 0u) {
      continue;
    }
    const std::size_t first = control.ordinary_step.value;
    if (first >= build.steps.size()) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    const PipelineBuildStep &head = build.steps[first];
    const std::size_t bound = head.iteration_bound;
    if (head.program == nullptr || head.iteration != 0u || bound == 0u ||
        bound > build.steps.size() - first) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    for (std::size_t iteration = 0u; iteration < bound; ++iteration) {
      const PipelineBuildStep &step = build.steps[first + iteration];
      if (step.program != head.program ||
          step.logical_step != head.logical_step ||
          step.iteration != iteration || step.iteration_bound != bound ||
          step.window_control.value != state || step.nested != 0u ||
          step.route != PipelineRoute::Ordinary) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
    }
  }

  for (std::size_t step_index = 0u; step_index < build.steps.size();
       ++step_index) {
    const PipelineBuildStep &step = build.steps[step_index];
    if (step.program == nullptr) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    if (step.inputs.empty() && step.outputs.empty() &&
        step.program->input_types.empty() &&
        step.program->output_types.empty()) {
      continue;
    }
    auto projection = project_outputs(step);
    if (!projection) {
      return Result<PipelineScheduleSuccess>::fail(projection.reason());
    }
    PipelineStepResourcePlan &sealed = plan.step_resources[step_index];
    sealed.inputs.clear();
    sealed.inputs.reserve(step.inputs.size());
    sealed.outputs.clear();
    sealed.outputs.reserve(step.outputs.size());
    sealed.physical_sources.assign(projection->physical_sources.begin(),
                                   projection->physical_sources.begin() +
                                       projection->physical_count);
    for (std::size_t input_index = 0u; input_index < step.inputs.size();
         ++input_index) {
      const PipelineBinding &input = step.inputs[input_index];
      const Type slot_type = input_index < step.program->input_types.size()
                                 ? step.program->input_types[input_index]
                                 : input.type;
      const FixedFormat slot_format =
          input_index < step.program->input_formats.size()
              ? step.program->input_formats[input_index]
              : input.format;
      auto view = resources.resolve(input, slot_type, slot_format);
      if (!view) {
        return Result<PipelineScheduleSuccess>::fail(view.reason(),
                                                     view.location());
      }
      if (!PipelineScheduleResources::append(
              resources.accesses, *view, static_cast<std::uint32_t>(step_index),
              resource::AccessMode::Read)) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineCapacity);
      }
      resources.use_evidence[view->resource].first_input =
          std::min(resources.use_evidence[view->resource].first_input,
                   static_cast<std::uint32_t>(step_index));
      sealed.inputs.push_back(*view);
    }
    for (std::size_t output = 0u; output < step.outputs.size(); ++output) {
      const std::uint32_t physical = projection->logical_to_physical[output];
      if (physical >= projection->physical_count ||
          physical >= step.program->output_types.size() ||
          physical >= step.program->output_sizes.size() ||
          physical >= step.program->output_formats.size()) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      auto view = resources.resolve(step.outputs[output],
                                    step.program->output_types[physical],
                                    step.program->output_formats[physical]);
      if (!view) {
        return Result<PipelineScheduleSuccess>::fail(view.reason(),
                                                     view.location());
      }
      const std::uint32_t canonical = sealed.physical_sources[physical];
      if (canonical >= step.outputs.size() || canonical > output) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
      if (canonical != output && (canonical >= sealed.outputs.size() ||
                                  sealed.outputs[canonical].view != *view)) {
        return Result<PipelineScheduleSuccess>::fail(
            Reason::BindingAliasUnsupported);
      }
      PipelineResolvedResourcePlan &resource =
          resources.resources[view->resource];
      resource.first_write = std::min(resource.first_write,
                                      static_cast<std::uint32_t>(step_index));
      if (view->offset == 0u && view->stride == 1u &&
          view->count == resource.count) {
        resources.use_evidence[view->resource].first_full_write =
            std::min(resources.use_evidence[view->resource].first_full_write,
                     static_cast<std::uint32_t>(step_index));
      }
      resource.output = resource.output || !step.outputs[output].hidden;
      sealed.outputs.push_back(PipelineResolvedOutputPlan{
          .view = *view,
          .physical = physical,
          .hidden = step.outputs[output].hidden,
      });
    }
    for (std::size_t physical = 0u; physical < projection->physical_count;
         ++physical) {
      const std::uint32_t source = sealed.physical_sources[physical];
      if (source == PipelineResourceUnassigned ||
          source >= sealed.outputs.size() ||
          !PipelineScheduleResources::append(
              resources.accesses, sealed.outputs[source].view,
              static_cast<std::uint32_t>(step_index),
              resource::AccessMode::Write)) {
        return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
      }
    }
  }
  auto publications =
      plan_pipeline_publications(build, window_states, resources, plan);
  if (!publications) {
    return Result<PipelineScheduleSuccess>::fail(publications.reason(),
                                                 publications.location());
  }
  // Device-side window publication is part of each frozen Fold route even
  // though publication descriptors are admitted after the authored steps.
  // Restore the analyzer's required nondecreasing node order while retaining
  // the original order of accesses within a route.
  const auto node_order = [](const resource::Access &left,
                             const resource::Access &right) {
    return left.node < right.node;
  };
  if (!std::is_sorted(resources.accesses.begin(), resources.accesses.end(),
                      node_order)) {
    std::stable_sort(resources.accesses.begin(), resources.accesses.end(),
                     node_order);
  }
  plan.state_pair_resources.clear();
  plan.state_pair_resources.reserve(build.state_pairs.size());
  for (const PipelineBuildStatePair &pair : build.state_pairs) {
    auto published = resources.resolve(pair.published, pair.published.type,
                                       pair.published.format);
    auto pending =
        resources.resolve(pair.pending, pair.pending.type, pair.pending.format);
    if (!published || !pending) {
      return Result<PipelineScheduleSuccess>::fail(
          published ? pending.reason() : published.reason());
    }
    if (pending->resource >= resources.use_evidence.size()) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    const PipelineResourceUseEvidence &pending_use =
        resources.use_evidence[pending->resource];
    plan.state_pair_resources.push_back(PipelineStatePairResourcePlan{
        .published = *published,
        .pending = *pending,
        .pending_first_input = pending_use.first_input,
        .pending_first_full_write = pending_use.first_full_write,
    });
  }
  const Status internals = resources.complete_internal_resources();
  if (!internals) {
    return Result<PipelineScheduleSuccess>::fail(internals.reason());
  }

  // Sealed repetitions may coalesce repeated invocations only when no
  // caller-owned write becomes a caller-owned read in the next invocation.
  // Program and Pipeline-private storage remains governed by the reusable-
  // execution reset/overwrite contract; caller-owned state is never inferred
  // from it.
  const Status repetitions = prove_sealed_repetitions(
      build, resources.shapes, resources.accesses, resources.external_flags,
      resources.publication_accesses);
  if (!repetitions) {
    return Result<PipelineScheduleSuccess>::fail(repetitions.reason());
  }

  resource::Plan hazards;
  if (resources.shapes.empty()) {
    if (!resources.accesses.empty()) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
  } else {
    auto analyzed =
        resource::analyze(resources.shapes, resources.accesses,
                          static_cast<std::uint32_t>(build.steps.size()));
    if (!analyzed) {
      return Result<PipelineScheduleSuccess>::fail(analyzed.reason());
    }
    hazards = std::move(*analyzed);
  }
  const std::size_t dependency_capacity = build.nested_windows.empty()
                                              ? PipelineBindingCapacity
                                              : PipelineRouteBindingCapacity;
  if (hazards.dependencies.size() > dependency_capacity) {
    return Result<PipelineScheduleSuccess>::fail(Reason::PipelineCapacity);
  }
  if (hazards.barriers.size() != hazards.dependencies.size() ||
      hazards.lifetimes.size() != resources.shapes.size()) {
    return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
  }

  plan.schedule_barriers.assign(build.steps.size(), 0u);
  for (std::size_t index = 0u; index < hazards.dependencies.size(); ++index) {
    const resource::Dependency dependency = hazards.dependencies[index];
    const resource::Barrier &witness = hazards.barriers[index];
    if (dependency.before_node >= build.steps.size() ||
        dependency.after_node >= plan.schedule_barriers.size() ||
        witness.before_node != dependency.before_node ||
        witness.after_node != dependency.after_node) {
      return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
    }
    plan.schedule_barriers[dependency.after_node] = 1u;
  }
  bool shared_chunks = false;
  for (std::size_t step_index = 0u; step_index < build.steps.size();
       ++step_index) {
    const bool current = !build.steps[step_index].program->chunks.empty();
    if (current && shared_chunks && step_index != 0u) {
      plan.schedule_barriers[step_index] = 1u;
    }
    shared_chunks = shared_chunks || current;
  }
  plan.summary.barrier_count = static_cast<std::uint64_t>(std::count(
      plan.schedule_barriers.begin(), plan.schedule_barriers.end(), 1u));
  plan.summary.resource_count =
      static_cast<std::uint64_t>(resources.shapes.size());
  if (resources.resources.size() != resources.shapes.size() ||
      resources.use_evidence.size() != resources.resources.size()) {
    return Result<PipelineScheduleSuccess>::fail(Reason::PipelineInvalid);
  }
  plan.resources = std::move(resources.resources);
  plan.hazards = std::move(hazards);
  return Result<PipelineScheduleSuccess>::success({});
}

} // namespace rund::compute::detail
