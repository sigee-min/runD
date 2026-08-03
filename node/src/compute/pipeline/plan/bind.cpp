#include "prepare.hpp"

#include "../../job/local.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../state.hpp"
#include "compare.hpp"
#include "local.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {

Status bind_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                     PipelinePrepare &prepare) {
  std::shared_ptr<PipelineState> &state = prepare.state;
  std::vector<PipelinePlanStep> &plan_steps = prepare.steps;
  std::vector<PipelineResourceAdmission> &resource_admissions =
      prepare.admissions;
  auto &physical_outputs = prepare.outputs;
  const std::uint64_t status_entry_count = prepare.status_count;
  if (state == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // Phase 3 may allocate and prepare private Program entries only after the
  // complete public Pipeline contract has passed its cold validation.
  const Status resources_valid = validate_pipeline_resources(*state);
  if (!resources_valid) {
    return Status::fail(resources_valid.reason());
  }
  if (build->memory == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->accel_templates = {};
  if (state->device->backend != Backend::Cpu) {
    if (!build->memory->accel_preparation.ok) {
      return Status::fail(Reason::PipelineInvalid);
    }
    state->accel_templates.limit = build->memory->accel_preparation;
  }
  auto memory =
      make_pipeline_memory(state->device, build->steps, *build->memory);
  if (!memory) {
    return Status::fail(memory.reason());
  }
  state->prepared_buffers = std::move(memory->prepared);
  state->cpu_prepared_arena = std::move(memory->cpu_prepared_arena);
  state->cpu_storage = std::move(memory->cpu_storage);
  if (build->memory->job_owners.size() != state->steps.size() ||
      memory->cpu_storage_by_step.size() != state->steps.size() ||
      build->memory->cpu_view_layouts.size() != state->steps.size() ||
      build->memory->cpu_route_slices.size() != state->steps.size() ||
      (state->device->backend == Backend::Cpu &&
       build->memory->cpu_job_slices.size() != state->steps.size()) ||
      (state->transactional && state->device->backend == Backend::Cpu &&
       build->memory->cpu_alternate_job_slices.size() != state->steps.size()) ||
      (state->transactional
           ? build->memory->cpu_alternate_route_slices.size() !=
                 state->steps.size()
           : !build->memory->cpu_alternate_route_slices.empty())) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto alternate_owner =
      [&](const std::uint32_t ordinal) -> const std::shared_ptr<BufferState> & {
    const std::uint32_t partner = resource_admissions[ordinal].partner;
    return state
        ->resources[partner == PipelineResourceAdmission::none ? ordinal
                                                               : partner]
        .buffer;
  };
  for (std::size_t step_index = 0u; step_index < state->steps.size();
       ++step_index) {
    PipelineStep &step = state->steps[step_index];
    const PipelinePlanStep &planned_step = plan_steps[step_index];
    const PipelineBuildStep &declared = build->steps[step_index];
    const PhysicalOutputProjection &physical_output =
        physical_outputs[step_index];
    const std::shared_ptr<JobWorkspace> &workspace = memory->steps[step_index];
    const std::size_t job_owner = build->memory->job_owners[step_index];
    if (job_owner != step_index) {
      if (job_owner >= step_index || state->steps[job_owner].job == nullptr ||
          (state->transactional &&
           state->steps[job_owner].alternate_job == nullptr)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      step.job = state->steps[job_owner].job;
      step.alternate_job = state->steps[job_owner].alternate_job;
      continue;
    }
    if (state->device->backend == Backend::Cpu) {
      std::shared_ptr<CpuGraphStorage> cpu_storage;
      const CpuRunRoutePlan *cpu_route = nullptr;
      const CpuRunRouteSlice *cpu_route_slice = nullptr;
      const std::size_t cpu_owner = memory->cpu_storage_by_step[step_index];
      if (cpu_owner != std::numeric_limits<std::size_t>::max()) {
        if (cpu_owner >= state->cpu_storage.size() ||
            cpu_owner >= build->memory->cpu_route_plans.size() ||
            state->cpu_prepared_arena == nullptr) {
          return Status::fail(Reason::PipelineInvalid);
        }
        cpu_storage = state->cpu_storage[cpu_owner];
        cpu_route = &build->memory->cpu_route_plans[cpu_owner];
        cpu_route_slice = &build->memory->cpu_route_slices[step_index];
      }
      const CpuViewTransferLayout &cpu_views =
          build->memory->cpu_view_layouts[step_index];
      if (state->cpu_prepared_arena == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const auto prepare_cpu_job =
          [&](const bool alternate, const CpuJobBindingSlice &binding_slice,
              const CpuRunRouteSlice *const route_slice)
          -> Result<std::shared_ptr<JobState>> {
        CpuJobBindingStorage bindings{};
        if (!state->cpu_prepared_arena->view(binding_slice, bindings) ||
            bindings.inputs.size() != declared.inputs.size() ||
            bindings.outputs.size() != planned_step.outputs.size() ||
            bindings.input_views.size() != declared.inputs.size() ||
            bindings.output_views.size() != planned_step.outputs.size() ||
            bindings.kernel_views.size() !=
                build->memory->views[step_index].size()) {
          return Result<std::shared_ptr<JobState>>::fail(
              Reason::PipelineInvalid);
        }
        for (std::size_t index = 0u; index < declared.inputs.size(); ++index) {
          const PipelineBinding &binding = declared.inputs[index];
          bindings.inputs[index] =
              alternate ? alternate_owner(planned_step.inputs[index])
                        : binding.buffer;
          bindings.input_views[index] =
              JobBufferView{.offset = binding.offset,
                            .count = binding.count,
                            .stride = binding.stride,
                            .element_bytes = binding.element_bytes,
                            .alignment = binding.alignment};
        }
        for (std::size_t index = 0u; index < planned_step.outputs.size();
             ++index) {
          const std::uint32_t source = physical_output.sources[index];
          if (source >= declared.outputs.size() ||
              (alternate && index >= planned_step.outputs.size())) {
            return Result<std::shared_ptr<JobState>>::fail(
                Reason::PipelineInvalid);
          }
          const PipelineBinding &binding = declared.outputs[source];
          bindings.outputs[index] =
              alternate ? alternate_owner(planned_step.outputs[index])
                        : binding.buffer;
          bindings.output_views[index] =
              JobBufferView{.offset = binding.offset,
                            .count = binding.count,
                            .stride = binding.stride,
                            .element_bytes = binding.element_bytes,
                            .alignment = binding.alignment};
        }
        std::copy(build->memory->views[step_index].begin(),
                  build->memory->views[step_index].end(),
                  bindings.kernel_views.begin());
        return prepare_pipeline_cpu_job(
            step.program, bindings, state->cpu_prepared_arena, workspace,
            cpu_storage, cpu_route, route_slice, cpu_views);
      };
      auto job = prepare_cpu_job(
          false, build->memory->cpu_job_slices[step_index], cpu_route_slice);
      if (!job) {
        prepare.failure = job.location();
        prepare.failure.step = declared.logical_step;
        prepare.failure.iteration = declared.iteration;
        return Status::fail(job.reason());
      }
      step.job = std::move(job).value();
      if (state->transactional) {
        const CpuRunRouteSlice *const alternate_route_slice =
            cpu_route == nullptr
                ? nullptr
                : &build->memory->cpu_alternate_route_slices[step_index];
        auto alternate = prepare_cpu_job(
            true, build->memory->cpu_alternate_job_slices[step_index],
            alternate_route_slice);
        if (!alternate) {
          prepare.failure = alternate.location();
          prepare.failure.step = declared.logical_step;
          prepare.failure.iteration = declared.iteration;
          return Status::fail(alternate.reason());
        }
        step.alternate_job = std::move(alternate).value();
      }
      continue;
    }
    std::vector<std::shared_ptr<BufferState>> inputs;
    std::vector<std::shared_ptr<BufferState>> outputs;
    std::vector<JobBufferView> input_views;
    std::vector<JobBufferView> output_views;
    inputs.reserve(planned_step.inputs.size());
    outputs.reserve(planned_step.outputs.size());
    input_views.reserve(planned_step.inputs.size());
    output_views.reserve(planned_step.outputs.size());
    for (const PipelineBinding &binding : declared.inputs) {
      inputs.push_back(binding.buffer);
      input_views.push_back(
          JobBufferView{.offset = binding.offset,
                        .count = binding.count,
                        .stride = binding.stride,
                        .element_bytes = binding.element_bytes,
                        .alignment = binding.alignment});
    }
    for (std::size_t index = 0u; index < planned_step.outputs.size(); ++index) {
      const PipelineBinding &binding =
          declared.outputs[physical_output.sources[index]];
      outputs.push_back(binding.buffer);
      output_views.push_back(
          JobBufferView{.offset = binding.offset,
                        .count = binding.count,
                        .stride = binding.stride,
                        .element_bytes = binding.element_bytes,
                        .alignment = binding.alignment});
    }
    auto job = prepare_pipeline_accel_job(
        step.program, std::move(inputs), std::move(outputs), input_views,
        output_views, build->memory->views[step_index], workspace);
    if (!job) {
      prepare.failure = job.location();
      prepare.failure.step = declared.logical_step;
      prepare.failure.iteration = declared.iteration;
      return Status::fail(job.reason());
    }
    step.job = std::move(job).value();
    if (state->transactional) {
      std::vector<std::shared_ptr<BufferState>> alternate_inputs;
      std::vector<std::shared_ptr<BufferState>> alternate_outputs;
      alternate_inputs.reserve(planned_step.inputs.size());
      alternate_outputs.reserve(planned_step.outputs.size());
      for (const std::uint32_t ordinal : planned_step.inputs) {
        alternate_inputs.push_back(alternate_owner(ordinal));
      }
      for (const std::uint32_t ordinal : planned_step.outputs) {
        alternate_outputs.push_back(alternate_owner(ordinal));
      }
      auto alternate = prepare_pipeline_accel_job(
          step.program, std::move(alternate_inputs),
          std::move(alternate_outputs), std::move(input_views),
          std::move(output_views), build->memory->views[step_index], workspace);
      if (!alternate) {
        prepare.failure = alternate.location();
        prepare.failure.step = declared.logical_step;
        prepare.failure.iteration = declared.iteration;
        return Status::fail(alternate.reason());
      }
      step.alternate_job = std::move(alternate).value();
    }
  }
  // The private Jobs now own the exact Buffer owners and Views.  Drop every
  // transient declaration/ordinal table before native stream construction
  // (and before a potentially large restore seed) instead of carrying a
  // second binding authority through the rest of preparation.
  std::vector<PipelinePlanStep>{}.swap(plan_steps);
  std::vector<PipelineResourceAdmission>{}.swap(resource_admissions);
  std::vector<PipelineBuildStep>{}.swap(build->steps);
  std::vector<PipelineBuildStatePair>{}.swap(build->state_pairs);
  std::vector<PipelineBuildPublication>{}.swap(build->publications);
  std::vector<PipelineInternal>{}.swap(build->internals);
  std::vector<PipelineBuildNestedWindow>{}.swap(build->nested_windows);
  build->memory.reset();
  state->status_entry_count = status_entry_count;
  state->stats.pipeline.status_entry_count = state->status_entry_count;

  const Status backend = prepare_backend(*state, prepare.failure);
  if (!backend) {
    return Status::fail(backend.reason());
  }
  return Status::success();
}

} // namespace rund::compute::detail
