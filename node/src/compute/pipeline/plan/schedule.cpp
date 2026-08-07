#include "publication.hpp"

#include <rund/compute/resource/plan.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status prove_sealed_repetitions(
    const PipelineBuildState &build,
    const std::span<const resource::Resource> resource_shapes,
    const std::span<const resource::Access> resource_accesses,
    const std::span<const std::uint8_t> external_resources,
    const std::span<const resource::Access> publication_accesses) {
  if (build.sealed_repetitions <= 1u) {
    return Status::success();
  }
  if (!build.state_pairs.empty()) {
    return Status::fail(Reason::PipelineTemporalDependency);
  }

  // Place every caller-owned write in invocation t before every caller-owned
  // read in invocation t + 1. resource::analyze remains the sole exact-range
  // authority, including genuinely strided Views. Any resulting W -> R witness
  // is a temporal carry and makes repeated execution observably stateful.
  std::vector<resource::Access> temporal_accesses;
  temporal_accesses.reserve(resource_accesses.size() +
                            publication_accesses.size());
  const auto append_mode = [&](const resource::AccessMode mode) {
    const auto append = [&](const resource::Access &source) {
      if (source.resource == 0u ||
          source.resource > external_resources.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (source.mode != mode ||
          external_resources[source.resource - 1u] == 0u) {
        return Status::success();
      }
      if (temporal_accesses.size() >=
          std::numeric_limits<std::uint32_t>::max()) {
        return Status::fail(Reason::PipelineCapacity);
      }
      resource::Access access = source;
      access.node = static_cast<std::uint32_t>(temporal_accesses.size());
      temporal_accesses.push_back(access);
      return Status::success();
    };
    for (const resource::Access &access : resource_accesses) {
      const Status appended = append(access);
      if (!appended) {
        return appended;
      }
    }
    for (const resource::Access &access : publication_accesses) {
      const Status appended = append(access);
      if (!appended) {
        return appended;
      }
    }
    return Status::success();
  };
  const Status writes = append_mode(resource::AccessMode::Write);
  if (!writes) {
    return writes;
  }
  const Status reads = append_mode(resource::AccessMode::Read);
  if (!reads || temporal_accesses.empty()) {
    return reads;
  }

  auto temporal =
      resource::analyze(resource_shapes, temporal_accesses,
                        static_cast<std::uint32_t>(temporal_accesses.size()));
  if (!temporal) {
    return Status::fail(temporal.reason());
  }
  for (const resource::Barrier &barrier : temporal->barriers) {
    if (barrier.before == resource::AccessMode::Write &&
        barrier.after == resource::AccessMode::Read) {
      return Status::fail(Reason::PipelineTemporalDependency);
    }
  }
  return Status::success();
}

} // namespace

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
  std::stable_sort(
      resources.accesses.begin(), resources.accesses.end(),
      [](const resource::Access &left, const resource::Access &right) {
        return left.node < right.node;
      });
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

Status schedule_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                         PipelinePrepare &prepare) {
  std::shared_ptr<PipelineState> &state = prepare.state;
  PipelineHash &hash = prepare.hash;
  const std::size_t output_count = prepare.output_count;
  if (state == nullptr || build->memory == nullptr ||
      build->memory->frozen == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineBuildSnapshot &frozen = *build->memory->frozen;
  // plan() and prepare() consume the same resource::analyze result and the
  // same executable boundary projection. No command-count proxy is allowed to
  // reconstruct scheduling here.
  const resource::Plan &planned = build->memory->hazards;
  const std::span<const std::uint8_t> barriers =
      build->memory->schedule_barriers;
  const std::size_t dependency_capacity = frozen.nested_windows.empty()
                                              ? PipelineBindingCapacity
                                              : PipelineRouteBindingCapacity;
  if (planned.dependencies.size() > dependency_capacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (planned.barriers.size() != planned.dependencies.size() ||
      planned.lifetimes.size() != state->resources.size() ||
      barriers.size() != state->barriers.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::copy(barriers.begin(), barriers.end(), state->barriers.begin());
  state->dependencies.resize(planned.dependencies.size());
  for (std::size_t dependency_index = 0u;
       dependency_index < planned.dependencies.size(); ++dependency_index) {
    const resource::Dependency dependency =
        planned.dependencies[dependency_index];
    const resource::Barrier &witness = planned.barriers[dependency_index];
    if (dependency.before_node >= state->steps.size() ||
        dependency.after_node >= state->barriers.size() ||
        witness.before_node != dependency.before_node ||
        witness.after_node != dependency.after_node) {
      return Status::fail(Reason::PipelineInvalid);
    }
    state->dependencies[dependency_index] = PipelineDependency{
        .before = dependency.before_node,
        .after = dependency.after_node,
        .resource = witness.after_resource - 1u,
        .before_access = witness.before == resource::AccessMode::Write
                             ? PipelineAccess::Write
                             : PipelineAccess::Read,
        .after_access = witness.after == resource::AccessMode::Write
                            ? PipelineAccess::Write
                            : PipelineAccess::Read,
    };
  }
  hash.number(planned.barriers.size());
  for (const resource::Barrier &barrier : planned.barriers) {
    hash.number(barrier.alias_group);
    hash.number(barrier.before_resource);
    hash.number(barrier.after_resource);
    hash.number(barrier.offset_bytes);
    hash.number(barrier.size_bytes);
    hash.number(barrier.before_offset_bytes);
    hash.number(barrier.before_element_bytes);
    hash.number(barrier.before_element_count);
    hash.number(barrier.before_stride_bytes);
    hash.number(barrier.after_offset_bytes);
    hash.number(barrier.after_element_bytes);
    hash.number(barrier.after_element_count);
    hash.number(barrier.after_stride_bytes);
    hash.number(barrier.before_node);
    hash.number(barrier.after_node);
    hash.byte(static_cast<std::uint8_t>(barrier.before));
    hash.byte(static_cast<std::uint8_t>(barrier.after));
  }
  hash.number(state->dependencies.size());
  for (const PipelineDependency dependency : state->dependencies) {
    hash.number(dependency.before);
    hash.number(dependency.after);
    hash.number(dependency.resource);
    hash.byte(static_cast<std::uint8_t>(dependency.before_access));
    hash.byte(static_cast<std::uint8_t>(dependency.after_access));
  }
  hash.number(state->barriers.size());
  for (const std::uint8_t barrier : state->barriers) {
    hash.byte(barrier);
    if (barrier != 0u) {
      ++state->stats.pipeline.barrier_count;
    }
  }
  if (state->stats.pipeline.barrier_count != state->plan.barrier_count) {
    return Status::fail(Reason::PipelineInvalid);
  }

  state->claims.resize(state->resources.size());
  if (state->transactional) {
    state->alternate_claims.resize(state->resources.size());
  }
  state->outputs.resize(output_count);
  state->output_lookup.resize(output_count);
  std::size_t output_index = 0u;
  for (std::uint32_t ordinal = 0u; ordinal < state->resources.size();
       ++ordinal) {
    PipelineResource &resource = state->resources[ordinal];
    const bool write = resource.output != PipelineResource::no_output;
    state->claims[ordinal] = BufferClaim{
        .buffer = resource.buffer.get(),
        .write = write,
        .transactional_state = resource.partner != PipelineResource::no_output,
        .gated_publish = resource.terminal_publish,
    };
    if (state->transactional) {
      state->alternate_claims[ordinal] = state->claims[ordinal];
      const PipelineResource *const selected =
          selected_pipeline_resource(*state, ordinal, true);
      if (selected == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      state->alternate_claims[ordinal].buffer = selected->buffer.get();
    }
    if (write) {
      resource.output = static_cast<std::uint32_t>(output_index);
      state->outputs[output_index] = PipelineOutputState{
          .resource = ordinal,
      };
      state->output_lookup[output_index] =
          static_cast<std::uint32_t>(output_index);
      ++output_index;
    }
  }
  std::sort(state->output_lookup.begin(), state->output_lookup.end(),
            [&](const std::uint32_t left, const std::uint32_t right) {
              const BufferState *const left_buffer =
                  state->resources[state->outputs[left].resource].buffer.get();
              const BufferState *const right_buffer =
                  state->resources[state->outputs[right].resource].buffer.get();
              return std::less<const BufferState *>{}(left_buffer,
                                                      right_buffer);
            });
  state->publication->fingerprint = hash.finish();
  state->stats.backend = state->device->backend;
  state->stats.graph_hash = state->publication->fingerprint.lo;
  state->logical_step_count = frozen.logical_step_count;
  state->stats.pipeline.step_count = state->logical_step_count;
  state->stats.pipeline.resource_count = state->resources.size();
  state->stats.pipeline.sealed_repetition_count = state->sealed_repetitions;

  return Status::success();
}

} // namespace rund::compute::detail
