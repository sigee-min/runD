#include "projection.hpp"

#include "../../device/residency_pool.hpp"
#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/window/model.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail {

bool project_virtual_run(VirtualPipelineState &state,
                         const std::uint64_t active_count,
                         VirtualRunProjection &projection) noexcept {
  projection = {};
  PipelineState &pipeline = *state.pipeline;
  const residency::StreamPlan &stream = pipeline.residency->stream();
  const std::uint64_t pair_bytes = pipeline.plan.residency.page_bytes;
  const std::uint64_t input_page_bytes = pipeline.residency_input_page_bytes;
  const std::uint64_t output_page_bytes = pipeline.residency_output_page_bytes;
  const std::uint64_t capacity = pipeline.plan.residency.frame_capacity;
  const VirtualGeometry &geometry = state.geometry;
  std::uint64_t expected_pair_bytes = 0u;
  std::uint64_t input_payload_bytes = 0u;
  std::uint64_t output_payload_bytes = 0u;
  std::uint64_t input_prefix_bytes = 0u;
  std::uint64_t output_prefix_bytes = 0u;
  std::uint64_t input_arena_bytes = 0u;
  std::uint64_t output_arena_bytes = 0u;
  std::uint64_t expected_staging = 0u;
  if (input_page_bytes == 0u || output_page_bytes == 0u ||
      state.input->element_bytes == 0u || state.output->element_bytes == 0u ||
      !kernel::checked::add(input_page_bytes, output_page_bytes,
                            expected_pair_bytes) ||
      geometry.input_payload_elements == 0u ||
      geometry.output_payload_elements == 0u ||
      geometry.input_frame_elements == 0u ||
      geometry.output_frame_elements == 0u ||
      !kernel::checked::mul(geometry.input_payload_elements,
                            state.input->element_bytes, input_payload_bytes) ||
      !kernel::checked::mul(geometry.output_payload_elements,
                            state.output->element_bytes,
                            output_payload_bytes) ||
      !kernel::checked::mul(geometry.input_prefix_elements,
                            state.input->element_bytes, input_prefix_bytes) ||
      !kernel::checked::mul(geometry.output_prefix_elements,
                            state.output->element_bytes, output_prefix_bytes) ||
      geometry.input_frame_elements !=
          input_page_bytes / state.input->element_bytes ||
      geometry.output_frame_elements !=
          output_page_bytes / state.output->element_bytes ||
      pair_bytes != expected_pair_bytes || capacity == 0u ||
      capacity > PipelineLeafCapacity ||
      !kernel::checked::mul(input_page_bytes, capacity, input_arena_bytes) ||
      !kernel::checked::mul(output_page_bytes, capacity, output_arena_bytes) ||
      !kernel::checked::add(input_arena_bytes, output_arena_bytes,
                            expected_staging) ||
      input_arena_bytes > std::numeric_limits<std::size_t>::max() ||
      output_arena_bytes > std::numeric_limits<std::size_t>::max() ||
      pipeline.residency_pool == nullptr ||
      pipeline.residency_pool->staging == nullptr ||
      pipeline.residency_pool->staging_bytes != expected_staging ||
      pipeline.residency_input >= pipeline.resources.size() ||
      pipeline.residency_output >= pipeline.resources.size() ||
      pipeline.residency_input == pipeline.residency_output ||
      state.input->backing->size_bytes() != state.input->bytes ||
      state.output->backing->size_bytes() != state.output->bytes) {
    return false;
  }

  VirtualActiveProjection active{};
  const std::uint64_t active_output_count =
      geometry.route == VirtualRoute::Reduction ? static_cast<std::uint64_t>(1u)
                                                : active_count;
  if (stream.frame_capacity() != capacity ||
      !project_virtual_active(stream, active_count, state.input->count,
                              geometry.input_payload_elements,
                              active_output_count, state.input->element_bytes,
                              state.output->element_bytes, active)) {
    return false;
  }

  std::byte *const input_stage = pipeline.residency_pool->staging.get();
  std::byte *const output_stage =
      input_stage + static_cast<std::size_t>(input_arena_bytes);
  projection = VirtualRunProjection{
      .active = active,
      .input_capacity_bytes = state.input->bytes,
      // A cached page is the exact materialization for this invocation's
      // visible prefix.  Keying every route by that extent prevents a page
      // filled for a larger run from exposing its inactive tail to a smaller
      // Map, Reduce or Scan invocation.
      .input_visible_bytes = active.input_bytes,
      .cache_extent = active.active_count,
      .cache_identity_hi = geometry.materialization_hi,
      .cache_identity_lo = geometry.materialization_lo,
      .input_page_bytes = input_page_bytes,
      .output_page_bytes = output_page_bytes,
      .input_payload_bytes = input_payload_bytes,
      .output_payload_bytes = output_payload_bytes,
      .input_prefix_bytes = input_prefix_bytes,
      .output_prefix_bytes = output_prefix_bytes,
      .input_frame_elements = geometry.input_frame_elements,
      .clamp_window = geometry.route == VirtualRoute::Window &&
                      geometry.boundary == static_cast<std::uint32_t>(
                                               kernel::WindowBoundary::Clamp),
      .clip_window = geometry.route == VirtualRoute::Window &&
                     geometry.boundary == static_cast<std::uint32_t>(
                                              kernel::WindowBoundary::Clip),
      .reduction = geometry.route == VirtualRoute::Reduction,
      .scan = geometry.route == VirtualRoute::Scan,
      .inclusive_scan = geometry.route == VirtualRoute::Scan &&
                        geometry.operation == static_cast<std::uint32_t>(
                                                  kernel::ScanOp::InclusiveSum),
      .input_type = state.input->type,
      .output_type = state.output->type,
      .operation = geometry.operation,
      .frame_capacity = capacity,
      .input_arena_bytes = input_arena_bytes,
      .output_arena_bytes = output_arena_bytes,
      .input_stage = input_stage,
      .output_stage = output_stage,
  };
  return true;
}

bool bind_virtual_run_transfer(VirtualPipelineState &state,
                               VirtualRunProjection &projection) noexcept {
  PipelineState &pipeline = *state.pipeline;
  PipelineResource &input_resource =
      pipeline.resources[pipeline.residency_input];
  PipelineResource &output_resource =
      pipeline.resources[pipeline.residency_output];
  if (!input_resource.owned || !output_resource.owned ||
      input_resource.buffer == nullptr || output_resource.buffer == nullptr ||
      input_resource.bytes != projection.input_arena_bytes ||
      output_resource.bytes != projection.output_arena_bytes ||
      output_resource.output == PipelineResource::no_output) {
    return false;
  }
  projection.upload = PipelineFrameUpload{
      .buffer = input_resource.buffer.get(),
      .data = projection.input_stage,
      .bytes = static_cast<std::size_t>(projection.input_arena_bytes),
  };
  projection.download = PipelineFrameDownload{
      .buffer = output_resource.buffer.get(),
      .data = projection.output_stage,
      .bytes = static_cast<std::size_t>(projection.output_arena_bytes),
      .output = output_resource.output,
  };
  return true;
}

bool project_virtual_epoch(const VirtualRunProjection &run,
                           const std::uint64_t epoch,
                           VirtualEpochProjection &projection) noexcept {
  projection = {};
  residency::PageRun pages{};
  if (!run.active.stream.epoch(epoch, pages) || pages.page_count == 0u ||
      pages.page_count > run.frame_capacity) {
    return false;
  }
  projection.failed_page = pages.first_page;
  projection.page_count = pages.page_count;
  std::uint64_t active_input_bytes = 0u;
  std::uint64_t active_output_bytes = 0u;
  if (!kernel::checked::mul(pages.first_page, run.input_payload_bytes,
                            projection.input_offset) ||
      (!run.reduction &&
       !kernel::checked::mul(pages.first_page, run.output_payload_bytes,
                             projection.output_offset)) ||
      !kernel::checked::mul(pages.page_count, run.input_payload_bytes,
                            active_input_bytes) ||
      !kernel::checked::mul(pages.page_count, run.output_payload_bytes,
                            active_output_bytes) ||
      projection.input_offset >= run.active.input_bytes ||
      (!run.reduction && projection.output_offset >= run.active.output_bytes) ||
      active_input_bytes > std::numeric_limits<std::size_t>::max() ||
      active_output_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  projection.logical_input_bytes = static_cast<std::size_t>(std::min(
      active_input_bytes, run.active.input_bytes - projection.input_offset));
  projection.logical_output_bytes = static_cast<std::size_t>(
      run.reduction
          ? active_output_bytes
          : std::min(active_output_bytes,
                     run.active.output_bytes - projection.output_offset));
  return true;
}

bool project_virtual_input_page(
    const VirtualRunProjection &run, const std::uint64_t page,
    VirtualInputPageProjection &projection) noexcept {
  projection = {};
  if (run.input_frame_elements == 0u || run.input_page_bytes == 0u ||
      run.input_page_bytes % run.input_frame_elements != 0u ||
      run.input_payload_bytes == 0u) {
    return false;
  }
  const std::uint64_t element_bytes =
      run.input_page_bytes / run.input_frame_elements;
  const std::uint64_t payload_elements =
      run.input_payload_bytes / element_bytes;
  const std::uint64_t prefix_elements = run.input_prefix_bytes / element_bytes;
  const std::uint64_t active_elements = run.input_visible_bytes / element_bytes;
  std::uint64_t core_first = 0u;
  std::uint64_t core_end = 0u;
  if (payload_elements == 0u || prefix_elements > run.input_frame_elements ||
      payload_elements > run.input_frame_elements - prefix_elements ||
      !kernel::checked::mul(page, payload_elements, core_first) ||
      core_first >= active_elements ||
      !kernel::checked::add(core_first, payload_elements, core_end)) {
    return false;
  }
  const std::uint64_t suffix_elements =
      run.input_frame_elements - prefix_elements - payload_elements;
  if (run.scan) {
    std::uint64_t logical_offset = 0u;
    std::uint64_t target_offset = 0u;
    std::uint64_t transfer_elements =
        std::min(payload_elements, active_elements - core_first);
    std::uint64_t transfer_bytes = 0u;
    if (!kernel::checked::mul(core_first, element_bytes, logical_offset) ||
        !kernel::checked::mul(prefix_elements, element_bytes, target_offset) ||
        !kernel::checked::mul(transfer_elements, element_bytes,
                              transfer_bytes) ||
        logical_offset > std::numeric_limits<std::size_t>::max() ||
        target_offset > std::numeric_limits<std::size_t>::max() ||
        transfer_bytes > std::numeric_limits<std::size_t>::max() ||
        target_offset + transfer_bytes > run.input_page_bytes) {
      return false;
    }
    projection = VirtualInputPageProjection{
        .logical_offset = logical_offset,
        .target_offset = static_cast<std::size_t>(target_offset),
        .transfer_bytes = static_cast<std::size_t>(transfer_bytes),
        .leading_fill_bytes = static_cast<std::size_t>(target_offset),
        .trailing_fill_offset =
            static_cast<std::size_t>(target_offset + transfer_bytes),
    };
    return true;
  }
  const std::uint64_t first =
      core_first > prefix_elements ? core_first - prefix_elements : 0u;
  std::uint64_t wanted_end = 0u;
  if (!kernel::checked::add(core_end, suffix_elements, wanted_end)) {
    return false;
  }
  const std::uint64_t end = std::min(wanted_end, active_elements);
  const std::uint64_t target_elements =
      core_first >= prefix_elements ? 0u : prefix_elements - core_first;
  const std::uint64_t transfer_elements = end - first;
  std::uint64_t logical_offset = 0u;
  std::uint64_t target_offset = 0u;
  std::uint64_t transfer_bytes = 0u;
  if (end <= first ||
      !kernel::checked::mul(first, element_bytes, logical_offset) ||
      !kernel::checked::mul(target_elements, element_bytes, target_offset) ||
      !kernel::checked::mul(transfer_elements, element_bytes, transfer_bytes) ||
      logical_offset > std::numeric_limits<std::size_t>::max() ||
      target_offset > std::numeric_limits<std::size_t>::max() ||
      transfer_bytes > std::numeric_limits<std::size_t>::max() ||
      target_offset + transfer_bytes > run.input_page_bytes) {
    return false;
  }
  projection = VirtualInputPageProjection{
      .logical_offset = logical_offset,
      .target_offset = static_cast<std::size_t>(target_offset),
      .transfer_bytes = static_cast<std::size_t>(transfer_bytes),
      .leading_fill_bytes = static_cast<std::size_t>(target_offset),
      .trailing_fill_offset =
          static_cast<std::size_t>(target_offset + transfer_bytes),
  };
  return true;
}

} // namespace rund::compute::detail
