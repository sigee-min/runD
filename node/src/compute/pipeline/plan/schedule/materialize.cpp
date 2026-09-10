#include "../../state/assembly.hpp"
#include "internal.hpp"

#include "../../residency/integration.hpp"

#include <algorithm>
#include <functional>

namespace rund::compute::detail {

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
      state->outputs[output_index] = PipelineOutputState{.resource = ordinal};
      state->output_lookup[output_index] =
          static_cast<std::uint32_t>(output_index);
      ++output_index;
    }
  }
  const Status residency = bind_pipeline_residency(*build->memory, *state);
  if (!residency) {
    return residency;
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
  state->stats.pipeline.preparation_evidence =
      state->device->backend == Backend::Cpu
          ? PreparationEvidenceSource::NoNativeProducer
          : PreparationEvidenceSource::Unavailable;
  state->stats.graph_hash = state->publication->fingerprint.lo;
  state->logical_step_count = frozen.logical_step_count;
  state->stats.pipeline.step_count = state->logical_step_count;
  state->stats.pipeline.resource_count = state->resources.size();
  state->stats.pipeline.sealed_repetition_count = state->sealed_repetitions;

  return Status::success();
}

} // namespace rund::compute::detail
