#include "local.hpp"

#include "../backend.hpp"
#include "../pipeline/residency/integration.hpp"
#include "../pipeline/residency/planner.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/capacity.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool page_local_map(const ProgramState &program) noexcept {
  return !program.graph_info.nodes.empty() &&
         std::all_of(program.graph_info.nodes.begin(),
                     program.graph_info.nodes.end(),
                     [](const graph::Node &node) {
                       return node.operation == graph::Operation::Map;
                     });
}

[[nodiscard]] Status
virtual_pipeline_capability(const DeviceState &device) noexcept {
  if (device.backend == Backend::Cpu) {
    return Status::success();
  }
  if (device.ops == nullptr ||
      device.ops->virtual_pipeline_capability == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  return device.ops->virtual_pipeline_capability(device);
}

} // namespace

Status
validate_virtual_pipeline_program(const ProgramState &program,
                                  const VirtualBufferState &input,
                                  const VirtualBufferState &output) noexcept {
  if (program.device == nullptr || program.input_types.size() != 1u ||
      program.input_sizes.size() != 1u || program.input_formats.size() != 1u ||
      program.output_types.size() != 1u || program.output_sizes.size() != 1u ||
      program.output_formats.size() != 1u || program.input_sizes[0] == 0u ||
      program.input_sizes[0] != program.output_sizes[0] ||
      program.input_types[0] != input.type ||
      program.input_formats[0] != input.format ||
      program.output_types[0] != output.type ||
      program.output_formats[0] != output.format || input.count == 0u ||
      input.count != output.count || !page_local_map(program)) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  return Status::success();
}

Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_pipeline(const std::shared_ptr<ProgramState> &program,
                         const std::shared_ptr<VirtualBufferState> &input,
                         const std::shared_ptr<VirtualBufferState> &output,
                         const std::uint32_t requested_slots) noexcept {
  if (program == nullptr || input == nullptr || output == nullptr ||
      input == output || input->backing == nullptr ||
      output->backing == nullptr || input->backing == output->backing) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineInvalid);
  }
  const Status valid =
      validate_virtual_pipeline_program(*program, *input, *output);
  if (!valid) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(valid.reason());
  }
  if (requested_slots == 0u || requested_slots > PipelineIterationCapacity ||
      requested_slots > PipelineLeafCapacity ||
      input->backing->size_bytes() != input->bytes ||
      output->backing->size_bytes() != output->bytes) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  const Status capability = virtual_pipeline_capability(*program->device);
  if (!capability) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        capability.reason());
  }

  const std::uint64_t page_elements = program->input_sizes[0];
  const std::uint64_t page_count =
      input->count / page_elements +
      static_cast<std::uint64_t>(input->count % page_elements != 0u);
  std::uint64_t input_slot_bytes = 0u;
  std::uint64_t output_slot_bytes = 0u;
  std::uint64_t slot_pair_bytes = 0u;
  std::uint64_t logical_bytes = 0u;
  if (!kernel::checked::mul(page_elements, input->element_bytes,
                            input_slot_bytes) ||
      !kernel::checked::mul(page_elements, output->element_bytes,
                            output_slot_bytes) ||
      !kernel::checked::add(input_slot_bytes, output_slot_bytes,
                            slot_pair_bytes) ||
      !kernel::checked::add(input->bytes, output->bytes, logical_bytes) ||
      input_slot_bytes > std::numeric_limits<std::size_t>::max() ||
      output_slot_bytes > std::numeric_limits<std::size_t>::max()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }

  residency::PlanInput page_input{
      // One residency page is an input/output transform pair. This makes the
      // planner's page bytes, resident slot bytes, and admitted peak share one
      // physical unit instead of mixing a one-sided logical page with a paired
      // working-set charge.
      .page_bytes = slot_pair_bytes,
      .page_count = page_count,
      .requested_slots = requested_slots,
      .max_slots = PipelineIterationCapacity,
  };
  auto planned = residency::PlanResidency(page_input);
  if (!planned) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        planned.failure == residency::Failure::Infeasible
            ? Reason::PipelineMemoryBudget
            : Reason::PipelineCapacity);
  }
  const std::uint64_t slots = planned.plan.linear().slot_capacity();
  std::uint64_t staging_bytes = 0u;
  std::uint64_t resident_bytes = 0u;
  if (slots == 0u ||
      !kernel::checked::mul(slots, slot_pair_bytes, staging_bytes) ||
      !kernel::checked::mul(slots, slot_pair_bytes, resident_bytes) ||
      staging_bytes > std::numeric_limits<std::size_t>::max()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }

  try {
    auto pages =
        std::make_shared<residency::ResidencyPlan>(std::move(planned.plan));
    auto build = make_pipeline(program->device);
    if (build == nullptr) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineCapacity);
    }
    append_pipeline_residency(build, program, pages, logical_bytes,
                              input_slot_bytes, output_slot_bytes,
                              staging_bytes, resident_bytes,
                              static_cast<std::uint32_t>(slots));
    auto physical = prepare_pipeline(std::move(build));
    if (!physical) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          physical.reason(), physical.location());
    }
    std::shared_ptr<PipelineState> pipeline = std::move(physical).value();
    if (pipeline->residency != pages ||
        pipeline->residency_staging == nullptr ||
        pipeline->residency_staging_bytes != staging_bytes ||
        pipeline->residency_input_page_bytes != input_slot_bytes ||
        pipeline->residency_output_page_bytes != output_slot_bytes ||
        pipeline->residency_input ==
            std::numeric_limits<std::uint32_t>::max() ||
        pipeline->residency_output ==
            std::numeric_limits<std::uint32_t>::max() ||
        pipeline->residency_input == pipeline->residency_output ||
        pipeline->outputs.size() != 1u) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineInvalid);
    }
    if (pipeline->device->backend == Backend::Vulkan) {
      if (pipeline->device->ops == nullptr ||
          pipeline->device->ops->prepare_pipeline_transfer == nullptr) {
        return Result<std::shared_ptr<VirtualPipelineState>>::fail(
            Reason::TransferInvalid);
      }
      const Status transfer =
          pipeline->device->ops->prepare_pipeline_transfer(*pipeline);
      if (!transfer) {
        return Result<std::shared_ptr<VirtualPipelineState>>::fail(
            transfer.reason());
      }
      pipeline->residency_transfer_prepared = true;
    } else if (pipeline->device->backend == Backend::Metal) {
      if (pipeline->device->ops == nullptr ||
          pipeline->device->ops->prepare_pipeline_residency == nullptr) {
        return Result<std::shared_ptr<VirtualPipelineState>>::fail(
            Reason::PipelineMemoryBudget);
      }
      const Status residency =
          pipeline->device->ops->prepare_pipeline_residency(*pipeline);
      if (!residency) {
        return Result<std::shared_ptr<VirtualPipelineState>>::fail(
            residency.reason());
      }
    }

    auto state = std::make_shared<VirtualPipelineState>();
    state->input = input;
    state->output = output;
    state->pipeline = std::move(pipeline);
    state->stats = pipeline_stats(state->pipeline);
    state->stats.pipeline.residency = ResidencyStats{
        .logical_bytes = logical_bytes,
        .page_bytes = slot_pair_bytes,
        .page_count = page_count,
        .slot_capacity = slots,
        .active_slots_peak = 0u,
        .plan_identity_hi = pages->identity().hi,
        .plan_identity_lo = pages->identity().lo,
    };
    return Result<std::shared_ptr<VirtualPipelineState>>::success(
        std::move(state));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
