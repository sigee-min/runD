#include "residency/local.hpp"

#include "../../backend.hpp"
#include "../../device/residency/pool.hpp"
#include "../../graph/compile/slice.hpp"
#include "../../graph/compile/slice/semantic.hpp"
#include "../../pipeline/residency/integration.hpp"
#include "../../pipeline/residency/planner.hpp"
#include "../host_ring.hpp"
#include "../stats.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/capacity.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace rund::compute::detail::virtual_prepare_detail {

Result<std::shared_ptr<VirtualPipelineState>> prepare_residency_pipeline(
    const std::shared_ptr<ProgramState> &program,
    const std::shared_ptr<VirtualBufferState> &input,
    const std::shared_ptr<VirtualBufferState> &output,
    const ResidencyConfig config, const VirtualGeometry &geometry_value,
    const std::shared_ptr<ProgramState> &pointwise_fused) noexcept {
  const VirtualGeometry *const geometry = &geometry_value;
  const std::uint64_t page_elements = geometry->input_payload_elements;
  const std::uint64_t page_count =
      input->count / page_elements +
      static_cast<std::uint64_t>(input->count % page_elements != 0u);
  std::uint64_t input_frame_bytes = 0u;
  std::uint64_t output_frame_bytes = 0u;
  std::uint64_t output_payload_bytes = 0u;
  std::uint64_t output_prefix_bytes = 0u;
  std::uint64_t dirty_frame_offset = 0u;
  std::uint64_t dirty_bytes = 0u;
  std::uint64_t frame_pair_bytes = 0u;
  std::uint64_t device_frame_bytes = 0u;
  std::uint64_t host_frame_bytes = 0u;
  std::uint64_t logical_bytes = 0u;
  if (!kernel::checked::mul(geometry->input_frame_elements,
                            input->element_bytes, input_frame_bytes) ||
      !kernel::checked::mul(geometry->output_frame_elements,
                            output->element_bytes, output_frame_bytes) ||
      !kernel::checked::mul(geometry->output_payload_elements,
                            output->element_bytes, output_payload_bytes) ||
      !kernel::checked::mul(geometry->output_prefix_elements,
                            output->element_bytes, output_prefix_bytes) ||
      !kernel::checked::add(input_frame_bytes, output_prefix_bytes,
                            dirty_frame_offset) ||
      !kernel::checked::add(input_frame_bytes, output_frame_bytes,
                            frame_pair_bytes) ||
      !kernel::checked::mul(frame_pair_bytes, 2u, device_frame_bytes) ||
      !kernel::checked::mul(frame_pair_bytes, 2u, host_frame_bytes) ||
      !kernel::checked::add(input->bytes, output->bytes, logical_bytes) ||
      input_frame_bytes > std::numeric_limits<std::size_t>::max() ||
      output_frame_bytes > std::numeric_limits<std::size_t>::max()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }

  const VirtualWindowPreflight window =
      virtual_prepare_detail::preflight_window(*program, *geometry, *input,
                                               *output, page_count, config);
  const bool window_ring =
      window.mode == VirtualWindowPreflightMode::WindowRing;

  std::uint64_t default_device_budget = 0u;
  std::uint64_t default_host_budget = 0u;
  if (!kernel::checked::mul(device_frame_bytes, 2u, default_device_budget) ||
      !kernel::checked::mul(host_frame_bytes, 2u, default_host_budget)) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  if (geometry->route == VirtualRoute::Window && !window_ring &&
      program->device->backend != Backend::Cpu) {
    // A centered Window epoch consumes the K core pages plus at most one
    // canonical neighbor on either side. Reserve those two source rows in
    // each Host bank; project_virtual_host_ring_capacities keeps the output
    // ring at executable depth and spends this remainder on input first.
    std::uint64_t window_source_rows = 0u;
    std::uint64_t window_source_bytes = 0u;
    if (!kernel::checked::mul(residency::Pool::BankCount, 2u,
                              window_source_rows) ||
        !kernel::checked::mul(window_source_rows, input_frame_bytes,
                              window_source_bytes) ||
        !kernel::checked::add(default_host_budget, window_source_bytes,
                              default_host_budget)) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineCapacity);
    }
  }
  const std::uint64_t device_budget =
      window_ring
          ? window.device_storage_bytes
          : (config.device_resident_bytes == 0u ? default_device_budget
                                                : config.device_resident_bytes);
  const std::uint64_t host_budget =
      window_ring
          ? window.host.storage_bytes
          : (config.host_resident_bytes == 0u ? default_host_budget
                                              : config.host_resident_bytes);
  const std::uint64_t device_requested_frames =
      device_budget / device_frame_bytes;
  const std::uint64_t host_requested_frames =
      window_ring ? window.host.input : host_budget / host_frame_bytes;
  const std::uint64_t requested_frames =
      window_ring
          ? window.frame_capacity
          : (program->device->backend == Backend::Cpu
                 ? host_requested_frames
                 : std::min(device_requested_frames, host_requested_frames));
  if (requested_frames == 0u || requested_frames > PipelineIterationCapacity ||
      requested_frames > PipelineLeafCapacity ||
      (!window_ring && host_requested_frames > PipelineIterationCapacity)) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineMemoryBudget);
  }

  if (geometry->route == VirtualRoute::Reduction) {
    if (!kernel::checked::mul(page_count, output_payload_bytes, dirty_bytes)) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineCapacity);
    }
  } else {
    dirty_bytes = output->bytes;
  }
  const std::uint64_t prefetch_distance =
      program->device->backend != Backend::Cpu &&
              input->backing->tier() == VirtualBackingTier::Persistent &&
              input->backing->max_parallel_reads() >= 2u
          ? 2u
          : 1u;

  residency::StreamPlanInput page_input{
      // One residency page is an input/output transform pair. This makes the
      // planner's page bytes, resident frame bytes, and admitted peak share one
      // physical unit instead of mixing a one-sided logical page with a paired
      // working-set charge.
      .page_bytes = frame_pair_bytes,
      .page_count = page_count,
      .requested_frames = requested_frames,
      .max_frames = PipelineIterationCapacity,
      .dirty = residency::DirtyRange{.offset = dirty_frame_offset,
                                     .bytes = output_payload_bytes},
      .dirty_bytes = dirty_bytes,
      .prefetch_distance = prefetch_distance,
  };
  auto planned = residency::PlanResidency(page_input);
  if (!planned) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        planned.failure == residency::Failure::Infeasible
            ? Reason::PipelineMemoryBudget
            : Reason::PipelineCapacity);
  }
  const std::uint64_t frames = planned.plan.stream().frame_capacity();
  std::uint64_t host_frames = frames;
  std::uint64_t host_output_frames = frames;
  std::uint64_t host_storage_bytes = 0u;
  std::uint64_t resident_bytes = 0u;
  if (program->device->backend != Backend::Cpu) {
    VirtualHostRingCapacities rings{};
    if (window_ring) {
      rings = window.host;
    } else if (!project_virtual_host_ring_capacities(
                   page_count, frames, input_frame_bytes, output_frame_bytes,
                   host_budget, PipelineIterationCapacity, rings)) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineMemoryBudget);
    }
    host_frames = rings.input;
    host_output_frames = rings.output;
  }
  if (frames == 0u || host_frames < frames || host_output_frames < frames ||
      (!window_ring && host_frames > PipelineIterationCapacity) ||
      !kernel::checked::mul(frames, device_frame_bytes, resident_bytes) ||
      frames > std::numeric_limits<std::uint32_t>::max() ||
      host_frames > std::numeric_limits<std::uint32_t>::max() ||
      host_output_frames > std::numeric_limits<std::uint32_t>::max()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  const residency::PoolLayout pool_layout{
      .input_type = program->input_types[0],
      .input_format = program->input_formats[0],
      .output_type = program->output_types[0],
      .output_format = program->output_formats[0],
      .input_page_bytes = input_frame_bytes,
      .output_page_bytes = output_frame_bytes,
      .frame_capacity = static_cast<std::uint32_t>(frames),
      .host_frame_capacity = static_cast<std::uint32_t>(host_frames),
      .host_output_frame_capacity =
          static_cast<std::uint32_t>(host_output_frames),
  };
  residency::PoolFootprint footprint{};
  if (!residency::project_pool_footprint(pool_layout, program->device->backend,
                                         footprint) ||
      footprint.host_storage_bytes > std::numeric_limits<std::size_t>::max()) {
    return Result<std::shared_ptr<VirtualPipelineState>>::fail(
        Reason::PipelineCapacity);
  }
  host_storage_bytes = footprint.host_storage_bytes;

  try {
    auto pages =
        std::make_shared<residency::ResidencyPlan>(std::move(planned.plan));
    if (program->device->residency == nullptr) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::DeviceInvalid);
    }
    auto pool =
        program->device->residency->acquire(program->device, pool_layout);
    if (pool == nullptr || pool->host_storage_bytes != host_storage_bytes) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          Reason::PipelineMemoryBudget);
    }
    std::shared_ptr<ProgramState> residency_program = program;
    if (geometry->route == VirtualRoute::Pointwise &&
        program->device->backend != Backend::Cpu &&
        program->graph_info.nodes.size() > 1u) {
      // This is an optional physical schedule for the exact pure-Map subset.
      // A valid Program outside that subset retains its original Host route.
      if (pointwise_fused != nullptr) {
        residency_program = pointwise_fused;
      }
    } else if (geometry->route == VirtualRoute::Scan &&
               geometry->device_vsm_required) {
      auto fused = graph_compile::compile_service_free_map_scan(program);
      if (!fused) {
        return Result<std::shared_ptr<VirtualPipelineState>>::fail(
            fused.reason(), fused.location());
      }
      residency_program = std::move(fused).value();
    }
    const auto prepare_bank = [&](const std::uint32_t bank) noexcept {
      return prepare_virtual_bank(program, residency_program, pages, pool,
                                  logical_bytes, input_frame_bytes,
                                  output_frame_bytes, resident_bytes,
                                  static_cast<std::uint32_t>(frames), bank);
    };

    auto primary = prepare_bank(0u);
    auto alternate = primary ? prepare_bank(1u)
                             : Result<std::shared_ptr<PipelineState>>::fail(
                                   primary.reason(), primary.location());
    if (!primary || !alternate) {
      const auto &failed = primary ? alternate : primary;
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          failed.reason(), failed.location());
    }

    auto state = std::make_shared<VirtualPipelineState>();
    state->inputs[0u] = input;
    state->input_count = 1u;
    state->output = output;
    state->pipeline = std::move(primary).value();
    state->alternate_pipeline = std::move(alternate).value();
    state->geometry = *geometry;
    state->window_preflight = window;
    state->stats = pipeline_stats(state->pipeline);
    const Status accumulated = accumulate_virtual_epoch(
        state->stats, pipeline_stats(state->alternate_pipeline));
    if (!accumulated) {
      return Result<std::shared_ptr<VirtualPipelineState>>::fail(
          accumulated.reason());
    }
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

} // namespace rund::compute::detail::virtual_prepare_detail
