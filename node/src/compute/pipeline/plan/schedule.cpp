#include "prepare.hpp"

#include "../../size.hpp"
#include "../../type.hpp"
#include "../state.hpp"

#include <rund/compute/resource/plan.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rund::compute::detail {

Status plan_pipeline_schedule(const PipelineBuildState &build,
                              PipelineMemoryPlan &plan) {
  constexpr std::uint32_t unassigned =
      std::numeric_limits<std::uint32_t>::max();
  if (build.steps.empty() ||
      build.steps.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::fail(Reason::PipelineInvalid);
  }

  std::vector<resource::Resource> resource_shapes;
  std::vector<resource::Access> resource_accesses;
  resource_shapes.reserve(
      std::min(build.binding_count, PipelineResourceCapacity));
  resource_accesses.reserve(build.binding_count);
  std::unordered_map<const BufferState *, std::uint32_t> external;
  external.reserve(std::min(build.binding_count, PipelineResourceCapacity));
  std::vector<std::uint32_t> internals(build.internals.size(), unassigned);

  const auto admit =
      [&](const PipelineBinding &binding) -> Result<std::uint32_t> {
    const BufferState *external_buffer = nullptr;
    std::uint32_t *internal_ordinal = nullptr;
    std::uint64_t bytes = 0u;
    if (binding.owner == PipelineBinding::external) {
      if (binding.buffer == nullptr) {
        return Result<std::uint32_t>::fail(Reason::BindingInvalid);
      }
      external_buffer = binding.buffer.get();
      const auto found = external.find(external_buffer);
      if (found != external.end()) {
        return Result<std::uint32_t>::success(found->second);
      }
      bytes = binding.buffer->bytes;
    } else {
      if (binding.owner >= build.internals.size()) {
        return Result<std::uint32_t>::fail(Reason::PipelineInvalid);
      }
      internal_ordinal = &internals[binding.owner];
      if (*internal_ordinal != unassigned) {
        return Result<std::uint32_t>::success(*internal_ordinal);
      }
      const PipelineInternal &internal = build.internals[binding.owner];
      std::size_t internal_bytes = 0u;
      const std::size_t width = type_bytes(internal.type);
      if (width == 0u ||
          !size::multiply(internal.count, width, internal_bytes)) {
        return Result<std::uint32_t>::fail(Reason::PipelineCapacity);
      }
      bytes = internal_bytes;
    }
    if (resource_shapes.size() >= PipelineResourceCapacity) {
      return Result<std::uint32_t>::fail(Reason::PipelineCapacity);
    }
    const auto ordinal = static_cast<std::uint32_t>(resource_shapes.size());
    resource_shapes.push_back(resource::Resource{
        .id = ordinal + 1u,
        .bytes = bytes,
        .alias_group = ordinal + 1u,
    });
    if (internal_ordinal != nullptr) {
      *internal_ordinal = ordinal;
    } else {
      external.emplace(external_buffer, ordinal);
    }
    return Result<std::uint32_t>::success(ordinal);
  };
  const auto append_access = [&](const PipelineBinding &binding,
                                 const std::uint32_t node,
                                 const std::uint32_t ordinal,
                                 const resource::AccessMode mode) {
    if (binding.count == 0u) {
      return true;
    }
    std::size_t offset_bytes = 0u;
    std::size_t stride_bytes = 0u;
    if (binding.element_bytes == 0u ||
        !size::multiply(binding.offset, binding.element_bytes, offset_bytes) ||
        !size::multiply(binding.stride, binding.element_bytes, stride_bytes)) {
      return false;
    }
    resource_accesses.push_back(resource::Access{
        .node = node,
        .resource = ordinal + 1u,
        .mode = mode,
        .offset_bytes = offset_bytes,
        .element_bytes = binding.element_bytes,
        .element_count = binding.count,
        .stride_bytes = stride_bytes,
    });
    return true;
  };

  for (std::size_t step_index = 0u; step_index < build.steps.size();
       ++step_index) {
    const PipelineBuildStep &step = build.steps[step_index];
    if (step.program == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (step.inputs.empty() && step.outputs.empty() &&
        step.program->input_types.empty() &&
        step.program->output_types.empty()) {
      continue;
    }
    auto projection = project_outputs(step);
    if (!projection) {
      return Status::fail(projection.reason());
    }
    for (const PipelineBinding &input : step.inputs) {
      auto ordinal = admit(input);
      if (!ordinal) {
        return Status::fail(ordinal.reason());
      }
      if (!append_access(input, static_cast<std::uint32_t>(step_index),
                         *ordinal, resource::AccessMode::Read)) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    std::array<std::uint32_t, PipelineLeafCapacity> output_ordinals{};
    output_ordinals.fill(unassigned);
    for (std::size_t output = 0u; output < step.outputs.size(); ++output) {
      auto ordinal = admit(step.outputs[output]);
      if (!ordinal) {
        return Status::fail(ordinal.reason());
      }
      const std::uint32_t physical = projection->logical_to_physical[output];
      if (physical >= projection->physical_count) {
        return Status::fail(Reason::PipelineInvalid);
      }
      output_ordinals[physical] = *ordinal;
    }
    for (std::size_t physical = 0u; physical < projection->physical_count;
         ++physical) {
      const std::uint32_t source = projection->physical_sources[physical];
      if (source >= step.outputs.size() ||
          output_ordinals[physical] == unassigned ||
          !append_access(
              step.outputs[source], static_cast<std::uint32_t>(step_index),
              output_ordinals[physical], resource::AccessMode::Write)) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
  }
  for (const PipelineBuildPublish &publication : build.publications) {
    auto source = admit(publication.source);
    auto target = admit(publication.target);
    if (!source || !target) {
      return Status::fail(source ? target.reason() : source.reason());
    }
  }
  for (const PipelineBuildStatePair &pair : build.state_pairs) {
    auto published = admit(pair.published);
    auto pending = admit(pair.pending);
    if (!published || !pending) {
      return Status::fail(published ? pending.reason() : published.reason());
    }
  }

  resource::Plan hazards;
  if (resource_shapes.empty()) {
    if (!resource_accesses.empty()) {
      return Status::fail(Reason::PipelineInvalid);
    }
  } else {
    auto analyzed =
        resource::analyze(resource_shapes, resource_accesses,
                          static_cast<std::uint32_t>(build.steps.size()));
    if (!analyzed) {
      return Status::fail(analyzed.reason());
    }
    hazards = std::move(*analyzed);
  }
  const std::size_t dependency_capacity = build.nested_windows.empty()
                                              ? PipelineBindingCapacity
                                              : PipelineRouteBindingCapacity;
  if (hazards.dependencies.size() > dependency_capacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (hazards.barriers.size() != hazards.dependencies.size() ||
      hazards.lifetimes.size() != resource_shapes.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }

  plan.schedule_barriers.assign(build.steps.size(), 0u);
  for (std::size_t index = 0u; index < hazards.dependencies.size(); ++index) {
    const resource::Dependency dependency = hazards.dependencies[index];
    const resource::Barrier &witness = hazards.barriers[index];
    if (dependency.before_node >= build.steps.size() ||
        dependency.after_node >= plan.schedule_barriers.size() ||
        witness.before_node != dependency.before_node ||
        witness.after_node != dependency.after_node) {
      return Status::fail(Reason::PipelineInvalid);
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
      static_cast<std::uint64_t>(resource_shapes.size());
  plan.hazards = std::move(hazards);
  return Status::success();
}

Status schedule_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                         PipelinePrepare &prepare) {
  std::shared_ptr<PipelineState> &state = prepare.state;
  std::vector<PipelineResourceAdmission> &resource_admissions =
      prepare.admissions;
  PipelineHash &hash = prepare.hash;
  const std::size_t output_count = prepare.output_count;
  if (state == nullptr || build->memory == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // plan() and prepare() consume the same resource::analyze result and the
  // same executable boundary projection. No command-count proxy is allowed to
  // reconstruct scheduling here.
  const resource::Plan &planned = build->memory->hazards;
  const std::span<const std::uint8_t> barriers =
      build->memory->schedule_barriers;
  const std::size_t dependency_capacity = build->nested_windows.empty()
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
        .transactional_state = resource_admissions[ordinal].partner !=
                               PipelineResourceAdmission::none,
        .gated_publish = resource.terminal_publish,
    };
    if (state->transactional) {
      state->alternate_claims[ordinal] = state->claims[ordinal];
      if (resource_admissions[ordinal].partner !=
          PipelineResourceAdmission::none) {
        state->alternate_claims[ordinal].buffer =
            state->resources[resource_admissions[ordinal].partner].buffer.get();
      }
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
  state->fingerprint = hash.finish();
  state->stats.backend = state->device->backend;
  state->stats.graph_hash = state->fingerprint.lo;
  state->logical_step_count = build->logical_step_count;
  state->stats.pipeline.step_count = state->logical_step_count;
  state->stats.pipeline.resource_count = state->resources.size();

  return Status::success();
}

} // namespace rund::compute::detail
