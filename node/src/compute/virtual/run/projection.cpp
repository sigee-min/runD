#include "projection.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail {

bool project_virtual_run(VirtualPipelineState &state,
                         const std::uint64_t active_count,
                         VirtualRunProjection &projection) noexcept {
  projection = {};
  PipelineState &pipeline = *state.pipeline;
  const residency::LinearPlan &linear = pipeline.residency->linear();
  const std::uint64_t pair_bytes = pipeline.plan.residency.page_bytes;
  const std::uint64_t input_page_bytes = pipeline.residency_input_page_bytes;
  const std::uint64_t output_page_bytes = pipeline.residency_output_page_bytes;
  const std::uint64_t capacity = pipeline.plan.residency.slot_capacity;
  std::uint64_t expected_pair_bytes = 0u;
  std::uint64_t input_arena_bytes = 0u;
  std::uint64_t output_arena_bytes = 0u;
  std::uint64_t expected_staging = 0u;
  if (input_page_bytes == 0u || output_page_bytes == 0u ||
      !kernel::checked::add(input_page_bytes, output_page_bytes,
                            expected_pair_bytes) ||
      pair_bytes != expected_pair_bytes || capacity == 0u ||
      capacity > PipelineLeafCapacity ||
      !kernel::checked::mul(input_page_bytes, capacity, input_arena_bytes) ||
      !kernel::checked::mul(output_page_bytes, capacity, output_arena_bytes) ||
      !kernel::checked::add(input_arena_bytes, output_arena_bytes,
                            expected_staging) ||
      input_arena_bytes > std::numeric_limits<std::size_t>::max() ||
      output_arena_bytes > std::numeric_limits<std::size_t>::max() ||
      pipeline.residency_staging == nullptr ||
      pipeline.residency_staging_bytes != expected_staging ||
      pipeline.residency_input >= pipeline.resources.size() ||
      pipeline.residency_output >= pipeline.resources.size() ||
      pipeline.residency_input == pipeline.residency_output ||
      state.input->backing->size_bytes() != state.input->bytes ||
      state.output->backing->size_bytes() != state.output->bytes) {
    return false;
  }

  VirtualActiveProjection active{};
  if (linear.slot_capacity() != capacity ||
      !project_virtual_active(linear, active_count, state.input->count,
                              input_page_bytes, output_page_bytes,
                              state.input->element_bytes,
                              state.output->element_bytes, active)) {
    return false;
  }

  std::byte *const input_stage = pipeline.residency_staging.get();
  std::byte *const output_stage =
      input_stage + static_cast<std::size_t>(input_arena_bytes);
  projection = VirtualRunProjection{
      .active = active,
      .input_page_bytes = input_page_bytes,
      .output_page_bytes = output_page_bytes,
      .slot_capacity = capacity,
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
  projection.upload = PipelineSlotUpload{
      .buffer = input_resource.buffer.get(),
      .data = projection.input_stage,
      .bytes = static_cast<std::size_t>(projection.input_arena_bytes),
  };
  projection.download = PipelineSlotDownload{
      .buffer = output_resource.buffer.get(),
      .data = projection.output_stage,
      .bytes = static_cast<std::size_t>(projection.output_arena_bytes),
      .output = output_resource.output,
  };
  return true;
}

bool project_virtual_wave(const VirtualRunProjection &run,
                          const std::uint64_t wave,
                          VirtualWaveProjection &projection) noexcept {
  projection = {};
  residency::PageRun pages{};
  if (!run.active.linear.wave(wave, pages) || pages.page_count == 0u ||
      pages.page_count > run.slot_capacity) {
    return false;
  }
  projection.failed_page = pages.first_page;
  projection.page_count = pages.page_count;
  std::uint64_t active_input_bytes = 0u;
  std::uint64_t active_output_bytes = 0u;
  if (!kernel::checked::mul(pages.first_page, run.input_page_bytes,
                            projection.input_offset) ||
      !kernel::checked::mul(pages.first_page, run.output_page_bytes,
                            projection.output_offset) ||
      !kernel::checked::mul(pages.page_count, run.input_page_bytes,
                            active_input_bytes) ||
      !kernel::checked::mul(pages.page_count, run.output_page_bytes,
                            active_output_bytes) ||
      projection.input_offset >= run.active.input_bytes ||
      projection.output_offset >= run.active.output_bytes ||
      active_input_bytes > std::numeric_limits<std::size_t>::max() ||
      active_output_bytes > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  projection.logical_input_bytes = static_cast<std::size_t>(std::min(
      active_input_bytes, run.active.input_bytes - projection.input_offset));
  projection.logical_output_bytes = static_cast<std::size_t>(std::min(
      active_output_bytes, run.active.output_bytes - projection.output_offset));
  return true;
}

} // namespace rund::compute::detail
