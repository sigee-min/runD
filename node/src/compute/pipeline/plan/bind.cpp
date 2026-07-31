#include "prepare.hpp"

#include "../../job/local.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../state.hpp"
#include "compare.hpp"
#include "local.hpp"

#include <algorithm>
#include <cstdint>
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
  auto memory =
      make_pipeline_memory(state->device, build->steps, *build->memory);
  if (!memory) {
    return Status::fail(memory.reason());
  }
  state->shared_buffers = std::move(memory->buffers);
  state->prepared_buffers = std::move(memory->prepared);
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
    // A nested action begins directly on its seeded first bank, so its two
    // parity routes repeat from iteration two.  The legacy top-level repeat
    // keeps its distinct seed route and begins owner reuse at iteration three.
    // In both cases the native command stream still consumes every occurrence;
    // only identical cold Job/prepared ownership is shared.
    const std::uint32_t reusable_from =
        declared.route == PipelineRoute::NestedAction ? 2u : 3u;
    if (declared.iteration_bound > 1u &&
        declared.iteration >= reusable_from) {
      if (step_index < 2u) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (same_recurrence_phase(declared, build->steps[step_index - 2u])) {
        if (state->steps[step_index - 2u].job == nullptr ||
            (state->transactional &&
             state->steps[step_index - 2u].alternate_job == nullptr)) {
          return Status::fail(Reason::PipelineInvalid);
        }
        step.job = state->steps[step_index - 2u].job;
        step.alternate_job = state->steps[step_index - 2u].alternate_job;
        continue;
      }
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
    auto job = prepare_pipeline_job_buffers(
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
      auto alternate = prepare_pipeline_job_buffers(
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
  std::vector<PipelineBuildPublish>{}.swap(build->publications);
  std::vector<PipelineInternal>{}.swap(build->internals);
  std::vector<PipelineBuildNestedWindow>{}.swap(build->nested_windows);
  build->memory.reset();
  state->status_entry_count = status_entry_count;
  state->stats.pipeline.status_entry_count = state->status_entry_count;

  const Status backend = prepare_backend(*state);
  if (!backend) {
    return Status::fail(backend.reason());
  }
  return Status::success();
}

} // namespace rund::compute::detail
