#include "prepare.hpp"

#include "../../size.hpp"
#include "../state.hpp"

#include <rund/compute/resource/plan.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace rund::compute::detail {

Status schedule_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                         PipelinePrepare &prepare) {
  std::shared_ptr<PipelineState> &state = prepare.state;
  std::vector<PipelineResourceAdmission> &resource_admissions =
      prepare.admissions;
  std::vector<PipelinePlanStep> &plan_steps = prepare.steps;
  auto &physical_outputs = prepare.outputs;
  PipelineHash &hash = prepare.hash;
  const std::size_t output_count = prepare.output_count;
  const std::size_t observed_bindings = prepare.binding_count;
  if (state == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // Phase 2 delegates exact strided/subrange overlap to the canonical
  // resource planner. Declaration order remains the scheduling authority;
  // disjoint views of one Buffer do not manufacture a dependency.
  std::vector<resource::Resource> resource_shapes;
  std::vector<resource::Access> resource_accesses;
  resource_shapes.reserve(state->resources.size());
  resource_accesses.reserve(observed_bindings);
  for (std::uint32_t ordinal = 0u; ordinal < state->resources.size();
       ++ordinal) {
    resource_shapes.push_back(resource::Resource{
        .id = ordinal + 1u,
        .bytes = state->resources[ordinal].bytes,
        .alias_group = ordinal + 1u,
    });
  }
  for (std::size_t step_index = 0u; step_index < state->steps.size();
       ++step_index) {
    const PipelinePlanStep &planned_step = plan_steps[step_index];
    const PipelineBuildStep &declared = build->steps[step_index];
    const PhysicalOutputProjection &physical_output =
        physical_outputs[step_index];
    const auto append_access = [&](const PipelineBinding &binding,
                                   const std::uint32_t ordinal,
                                   const resource::AccessMode mode) {
      if (binding.count == 0u) {
        return true;
      }
      std::size_t offset_bytes = 0u;
      std::size_t stride_bytes = 0u;
      if (binding.element_bytes == 0u ||
          !size::multiply(binding.offset, binding.element_bytes,
                          offset_bytes) ||
          !size::multiply(binding.stride, binding.element_bytes,
                          stride_bytes)) {
        return false;
      }
      resource_accesses.push_back(resource::Access{
          .node = static_cast<std::uint32_t>(step_index),
          .resource = ordinal + 1u,
          .mode = mode,
          .offset_bytes = offset_bytes,
          .element_bytes = binding.element_bytes,
          .element_count = binding.count,
          .stride_bytes = stride_bytes,
      });
      return true;
    };
    for (std::size_t index = 0u; index < planned_step.inputs.size(); ++index) {
      if (!append_access(declared.inputs[index], planned_step.inputs[index],
                         resource::AccessMode::Read)) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    for (std::size_t index = 0u; index < planned_step.outputs.size(); ++index) {
      if (!append_access(declared.outputs[physical_output.sources[index]],
                         planned_step.outputs[index],
                         resource::AccessMode::Write)) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
  }
  auto planned =
      resource::analyze(resource_shapes, resource_accesses,
                        static_cast<std::uint32_t>(state->steps.size()));
  if (!planned) {
    return Status::fail(planned.reason());
  }
  if (planned->dependencies.size() > PipelineBindingCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (planned->barriers.size() != planned->dependencies.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->dependencies.resize(planned->dependencies.size());
  for (std::size_t dependency_index = 0u;
       dependency_index < planned->dependencies.size(); ++dependency_index) {
    const resource::Dependency dependency =
        planned->dependencies[dependency_index];
    const resource::Barrier &witness = planned->barriers[dependency_index];
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
    state->barriers[dependency.after_node] = 1u;
  }
  bool shared_chunks = false;
  for (std::size_t step_index = 0u; step_index < build->steps.size();
       ++step_index) {
    const std::shared_ptr<ProgramState> &program =
        build->steps[step_index].program;
    if (program == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const bool current = !program->chunks.empty();
    if (current && shared_chunks && step_index != 0u) {
      state->barriers[step_index] = 1u;
    }
    shared_chunks = shared_chunks || current;
  }
  hash.number(planned->barriers.size());
  for (const resource::Barrier &barrier : planned->barriers) {
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
