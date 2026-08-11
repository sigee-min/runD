#include "local.hpp"

#include "../backend.hpp"
#include "../device/residency_pool.hpp"
#include "../pipeline/residency/integration.hpp"
#include "../pipeline/residency/planner.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/reduce/operation.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/window/model.hpp>
#include <rund/compute/pipeline/capacity.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::optional<VirtualGeometry>
virtual_geometry(const ProgramState &program) noexcept {
  if (program.graph_info.nodes.empty() || program.input_sizes.size() != 1u ||
      program.output_sizes.size() != 1u || program.input_sizes[0] == 0u ||
      program.output_sizes[0] == 0u) {
    return std::nullopt;
  }
  const std::uint64_t input_elements = program.input_sizes[0];
  const std::uint64_t output_elements = program.output_sizes[0];
  const graph::Node *collective = nullptr;
  for (const graph::Node &node : program.graph_info.nodes) {
    if (node.operation == graph::Operation::Map) {
      // A pointwise transform after a collective changes the meaning of each
      // page partial and cannot be moved across the global merge.
      if (collective != nullptr &&
          collective->operation == graph::Operation::Reduce) {
        return std::nullopt;
      }
      continue;
    }
    if ((node.operation != graph::Operation::Window &&
         node.operation != graph::Operation::Reduce &&
         node.operation != graph::Operation::Scan) ||
        collective != nullptr) {
      return std::nullopt;
    }
    collective = &node;
  }
  if (collective == nullptr) {
    return input_elements == output_elements
               ? std::optional<VirtualGeometry>{VirtualGeometry{
                     .route = VirtualRoute::Pointwise,
                     .input_payload_elements = input_elements,
                     .output_payload_elements = output_elements,
                     .input_frame_elements = input_elements,
                     .output_frame_elements = output_elements,
                 }}
               : std::nullopt;
  }
  const graph::Footprint &footprint = collective->footprint;
  if (collective->operation == graph::Operation::Scan) {
    const auto operation = static_cast<kernel::ScanOp>(footprint.operation);
    const bool exclusive = operation == kernel::ScanOp::ExclusiveSum;
    const bool inclusive = operation == kernel::ScanOp::InclusiveSum;
    if (program.graph_info.nodes.size() != 1u ||
        footprint.pattern != graph::AccessPattern::Prefix ||
        footprint.input_elements != input_elements ||
        footprint.output_elements != output_elements ||
        input_elements != output_elements || footprint.tile_elements == 0u ||
        (!exclusive && !inclusive) || (exclusive && input_elements < 2u)) {
      return std::nullopt;
    }
    return VirtualGeometry{
        .route = VirtualRoute::Scan,
        .input_payload_elements =
            exclusive ? input_elements - 1u : input_elements,
        .output_payload_elements =
            exclusive ? output_elements - 1u : output_elements,
        .input_frame_elements = input_elements,
        .output_frame_elements = output_elements,
        .input_prefix_elements = exclusive ? 1u : 0u,
        .output_prefix_elements = exclusive ? 1u : 0u,
        .operation = footprint.operation,
        .materialization_hi = program.graph_info.fingerprint.hi,
        .materialization_lo = program.graph_info.fingerprint.lo,
    };
  }
  if (collective->operation == graph::Operation::Reduce) {
    const auto operation = static_cast<kernel::ReduceOp>(footprint.operation);
    const bool unsigned_sum = operation != kernel::ReduceOp::Sum ||
                              program.input_types[0] == Type::U32 ||
                              program.input_types[0] == Type::U64;
    if (footprint.pattern != graph::AccessPattern::Reduction ||
        footprint.input_elements != input_elements ||
        footprint.output_elements != 1u || output_elements != 1u ||
        footprint.tile_elements == 0u || !kernel::reduce::valid(operation) ||
        !unsigned_sum) {
      return std::nullopt;
    }
    return VirtualGeometry{
        .route = VirtualRoute::Reduction,
        .input_payload_elements = input_elements,
        .output_payload_elements = 1u,
        .input_frame_elements = input_elements,
        .output_frame_elements = 1u,
        .operation = footprint.operation,
        .materialization_hi = program.graph_info.fingerprint.hi,
        .materialization_lo = program.graph_info.fingerprint.lo,
    };
  }
  std::uint64_t symmetric_window = 0u;
  if (footprint.pattern != graph::AccessPattern::Window ||
      footprint.input_elements != input_elements ||
      footprint.output_elements != output_elements ||
      input_elements != output_elements || footprint.stride != 1u ||
      footprint.pad_left == 0u ||
      !kernel::checked::mul(footprint.pad_left, 2u, symmetric_window) ||
      !kernel::checked::add(symmetric_window, 1u, symmetric_window) ||
      footprint.window_size != symmetric_window ||
      input_elements <= footprint.pad_left * 2u ||
      (footprint.boundary !=
           static_cast<std::uint32_t>(kernel::WindowBoundary::Clamp) &&
       footprint.boundary !=
           static_cast<std::uint32_t>(kernel::WindowBoundary::Clip))) {
    return std::nullopt;
  }
  return VirtualGeometry{
      .route = VirtualRoute::Window,
      .input_payload_elements = input_elements - footprint.pad_left * 2u,
      .output_payload_elements = output_elements - footprint.pad_left * 2u,
      .input_frame_elements = input_elements,
      .output_frame_elements = output_elements,
      .input_prefix_elements = footprint.pad_left,
      .output_prefix_elements = footprint.pad_left,
      .operation = footprint.operation,
      .boundary = footprint.boundary,
      .materialization_hi = program.graph_info.fingerprint.hi,
      .materialization_lo = program.graph_info.fingerprint.lo,
  };
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
      program.input_types[0] != input.type ||
      program.input_formats[0] != input.format ||
      program.output_types[0] != output.type ||
      program.output_formats[0] != output.format || input.count == 0u) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  const auto geometry = virtual_geometry(program);
  if (!geometry || (geometry->route == VirtualRoute::Reduction
                        ? output.count != 1u
                        : input.count != output.count)) {
    return Status::fail(Reason::PrimitiveUnsupported);
  }
  return Status::success();
}

Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_pipeline(const std::shared_ptr<ProgramState> &program,
                         const std::shared_ptr<VirtualBufferState> &input,
                         const std::shared_ptr<VirtualBufferState> &output,
                         const ResidencyConfig config) noexcept {
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
  if (input->backing->size_bytes() != input->bytes ||
      output->backing->size_bytes() != output->bytes) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  const Status capability = virtual_pipeline_capability(*program->device);
  if (!capability) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        capability.reason());
  }

  const auto geometry = virtual_geometry(*program);
  if (!geometry) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PrimitiveUnsupported);
  }
  const std::uint64_t page_elements = geometry->input_payload_elements;
  const std::uint64_t page_count =
      input->count / page_elements +
      static_cast<std::uint64_t>(input->count % page_elements != 0u);
  std::uint64_t input_frame_bytes = 0u;
  std::uint64_t output_frame_bytes = 0u;
  std::uint64_t frame_pair_bytes = 0u;
  std::uint64_t device_frame_bytes = 0u;
  std::uint64_t host_frame_bytes = 0u;
  std::uint64_t logical_bytes = 0u;
  if (!kernel::checked::mul(geometry->input_frame_elements,
                            input->element_bytes, input_frame_bytes) ||
      !kernel::checked::mul(geometry->output_frame_elements,
                            output->element_bytes, output_frame_bytes) ||
      !kernel::checked::add(input_frame_bytes, output_frame_bytes,
                            frame_pair_bytes) ||
      !kernel::checked::mul(frame_pair_bytes, 2u, device_frame_bytes) ||
      !kernel::checked::mul(input_frame_bytes, 2u, host_frame_bytes) ||
      !kernel::checked::add(host_frame_bytes, frame_pair_bytes,
                            host_frame_bytes) ||
      !kernel::checked::add(input->bytes, output->bytes, logical_bytes) ||
      input_frame_bytes > std::numeric_limits<std::size_t>::max() ||
      output_frame_bytes > std::numeric_limits<std::size_t>::max()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }

  std::uint64_t default_device_budget = 0u;
  std::uint64_t default_host_budget = 0u;
  if (!kernel::checked::mul(device_frame_bytes, 2u, default_device_budget) ||
      !kernel::checked::mul(host_frame_bytes, 2u, default_host_budget)) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  const std::uint64_t device_budget = config.device_resident_bytes == 0u
                                          ? default_device_budget
                                          : config.device_resident_bytes;
  const std::uint64_t host_budget = config.host_staging_bytes == 0u
                                        ? default_host_budget
                                        : config.host_staging_bytes;
  const std::uint64_t requested_frames =
      geometry->route == VirtualRoute::Scan
          ? std::min<std::uint64_t>(1u,
                                    std::min(device_budget / device_frame_bytes,
                                             host_budget / host_frame_bytes))
          : std::min(device_budget / device_frame_bytes,
                     host_budget / host_frame_bytes);
  if (requested_frames == 0u || requested_frames > PipelineIterationCapacity ||
      requested_frames > PipelineLeafCapacity) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineMemoryBudget);
  }

  residency::StreamPlanInput page_input{
      // One residency page is an input/output transform pair. This makes the
      // planner's page bytes, resident frame bytes, and admitted peak share one
      // physical unit instead of mixing a one-sided logical page with a paired
      // working-set charge.
      .page_bytes = frame_pair_bytes,
      .page_count = page_count,
      .requested_frames = requested_frames,
      .max_frames = PipelineIterationCapacity,
  };
  auto planned = residency::PlanResidency(page_input);
  if (!planned) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        planned.failure == residency::Failure::Infeasible
            ? Reason::PipelineMemoryBudget
            : Reason::PipelineCapacity);
  }
  const std::uint64_t frames = planned.plan.stream().frame_capacity();
  std::uint64_t staging_bytes = 0u;
  std::uint64_t resident_bytes = 0u;
  if (frames == 0u ||
      !kernel::checked::mul(frames, frame_pair_bytes, staging_bytes) ||
      !kernel::checked::mul(frames, device_frame_bytes, resident_bytes) ||
      staging_bytes > std::numeric_limits<std::size_t>::max()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }

  try {
    auto pages =
        std::make_shared<residency::ResidencyPlan>(std::move(planned.plan));
    if (program->device->residency == nullptr) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::DeviceInvalid);
    }
    auto pool = program->device->residency->acquire(
        program->device,
        residency::PoolLayout{
            .input_type = program->input_types[0],
            .input_format = program->input_formats[0],
            .output_type = program->output_types[0],
            .output_format = program->output_formats[0],
            .input_page_bytes = input_frame_bytes,
            .output_page_bytes = output_frame_bytes,
            .frame_capacity = static_cast<std::uint32_t>(frames),
        });
    if (pool == nullptr || pool->staging_bytes != staging_bytes) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineMemoryBudget);
    }
    auto build = make_pipeline(program->device);
    if (build == nullptr) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineCapacity);
    }
    append_pipeline_residency(
        build, program, pages, pool, logical_bytes, input_frame_bytes,
        output_frame_bytes, resident_bytes, static_cast<std::uint32_t>(frames));
    auto physical = prepare_pipeline(std::move(build));
    if (!physical) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          physical.reason(), physical.location());
    }
    std::shared_ptr<PipelineState> pipeline = std::move(physical).value();
    if (pipeline->residency != pages || pipeline->residency_pool != pool ||
        pool->staging == nullptr ||
        pipeline->residency_input_page_bytes != input_frame_bytes ||
        pipeline->residency_output_page_bytes != output_frame_bytes ||
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
    state->geometry = *geometry;
    state->stats = pipeline_stats(state->pipeline);
    state->stats.pipeline.residency = ResidencyStats{
        .logical_bytes = logical_bytes,
        .page_bytes = frame_pair_bytes,
        .page_count = page_count,
        .frame_capacity = frames,
        .resident_frames_peak = 0u,
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
