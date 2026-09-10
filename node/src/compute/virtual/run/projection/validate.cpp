#include "ordinary.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../device/state.hpp"
#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <array>
#include <limits>

namespace rund::compute::detail {

[[nodiscard]] bool virtual_valid_regions(
    const residency::Pool &pool,
    const std::array<residency::FrameRegion, residency::Pool::BankCount>
        &regions,
    residency::FrameTier tier, residency::FrameRole role,
    std::uint32_t capacity) noexcept;

bool validate_virtual_ordinary_projection(
    VirtualPipelineState &state, OrdinaryProjectionInputs &inputs) noexcept {
  VirtualBufferState *const input = virtual_input(state);
  if (input == nullptr) {
    return false;
  }
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
  std::uint64_t dirty_frame_offset = 0u;
  std::uint64_t capacity_dirty_bytes = 0u;
  std::uint64_t input_arena_bytes = 0u;
  std::uint64_t output_arena_bytes = 0u;
  std::uint64_t host_input_group_bytes = 0u;
  std::uint64_t host_input_arena_bytes = 0u;
  std::uint64_t host_output_arena_bytes = 0u;
  std::uint64_t expected_host_storage = 0u;
  if (input_page_bytes == 0u || output_page_bytes == 0u ||
      input->element_bytes == 0u || state.output->element_bytes == 0u ||
      !kernel::checked::add(input_page_bytes, output_page_bytes,
                            expected_pair_bytes) ||
      geometry.input_payload_elements == 0u ||
      geometry.output_payload_elements == 0u ||
      geometry.input_frame_elements == 0u ||
      geometry.output_frame_elements == 0u ||
      !kernel::checked::mul(geometry.input_payload_elements,
                            input->element_bytes, input_payload_bytes) ||
      !kernel::checked::mul(geometry.output_payload_elements,
                            state.output->element_bytes,
                            output_payload_bytes) ||
      !kernel::checked::mul(geometry.input_prefix_elements,
                            input->element_bytes, input_prefix_bytes) ||
      !kernel::checked::mul(geometry.output_prefix_elements,
                            state.output->element_bytes, output_prefix_bytes) ||
      !kernel::checked::add(input_page_bytes, output_prefix_bytes,
                            dirty_frame_offset) ||
      (geometry.route == VirtualRoute::Reduction
           ? !kernel::checked::mul(stream.page_count(), output_payload_bytes,
                                   capacity_dirty_bytes)
           : (capacity_dirty_bytes = state.output->bytes, false)) ||
      geometry.input_frame_elements !=
          input_page_bytes / input->element_bytes ||
      geometry.output_frame_elements !=
          output_page_bytes / state.output->element_bytes ||
      stream.dirty_extent() !=
          residency::DirtyRange{.offset = dirty_frame_offset,
                                .bytes = output_payload_bytes} ||
      stream.dirty_bytes() != capacity_dirty_bytes ||
      pair_bytes != expected_pair_bytes || capacity == 0u ||
      capacity > PipelineLeafCapacity ||
      !kernel::checked::mul(input_page_bytes, capacity, input_arena_bytes) ||
      !kernel::checked::mul(output_page_bytes, capacity, output_arena_bytes) ||
      !kernel::checked::mul(
          input_page_bytes,
          pipeline.residency_pool == nullptr
              ? 0u
              : pipeline.residency_pool->layout.host_frame_capacity,
          host_input_group_bytes) ||
      !kernel::checked::mul(
          host_input_group_bytes,
          pipeline.residency_pool == nullptr
              ? 0u
              : pipeline.residency_pool->layout.graph_host_input_count,
          host_input_arena_bytes) ||
      !kernel::checked::mul(
          output_page_bytes,
          pipeline.residency_pool == nullptr
              ? 0u
              : pipeline.residency_pool->layout.host_output_frame_capacity,
          host_output_arena_bytes) ||
      (pipeline.device->backend != Backend::Cpu &&
       (!kernel::checked::add(host_input_arena_bytes, host_output_arena_bytes,
                              expected_host_storage) ||
        !kernel::checked::mul(expected_host_storage, residency::Pool::BankCount,
                              expected_host_storage))) ||
      input_arena_bytes > std::numeric_limits<std::size_t>::max() ||
      output_arena_bytes > std::numeric_limits<std::size_t>::max() ||
      pipeline.residency_pool == nullptr ||
      pipeline.residency_pool->host_storage_bytes != expected_host_storage ||
      (pipeline.device->backend != Backend::Cpu &&
       pipeline.residency_pool->host_storage == nullptr) ||
      !virtual_valid_regions(
          *pipeline.residency_pool, pipeline.residency_pool->input_regions,
          pipeline.device->backend == Backend::Cpu
              ? residency::FrameTier::Host
              : residency::FrameTier::Device,
          residency::FrameRole::Input, static_cast<std::uint32_t>(capacity)) ||
      pipeline.residency_pool->output_frame_count !=
          capacity * residency::Pool::BankCount ||
      pipeline.residency_pool->layout.host_frame_capacity < capacity ||
      (pipeline.device->backend == Backend::Cpu
           ? pipeline.residency_pool->host_input_frame_count != 0u
           : pipeline.residency_pool->host_input_frame_count !=
                 pipeline.residency_pool->layout.host_frame_capacity *
                     pipeline.residency_pool->layout.graph_host_input_count *
                     residency::Pool::BankCount) ||
      pipeline.residency_pool->first_output_frame >
          pipeline.residency_pool->authority().frame_capacity() ||
      pipeline.residency_pool->output_frame_count >
          pipeline.residency_pool->authority().frame_capacity() -
              pipeline.residency_pool->first_output_frame ||
      (pipeline.residency_pool->host_input_frame_count != 0u &&
       (pipeline.residency_pool->first_host_input_frame >
            pipeline.residency_pool->authority().frame_capacity() ||
        pipeline.residency_pool->host_input_frame_count >
            pipeline.residency_pool->authority().frame_capacity() -
                pipeline.residency_pool->first_host_input_frame)) ||
      (pipeline.device->backend == Backend::Cpu
           ? pipeline.residency_pool->host_output_frame_count != 0u
           : pipeline.residency_pool->host_output_frame_count !=
                 pipeline.residency_pool->layout.host_output_frame_capacity *
                     residency::Pool::BankCount) ||
      (pipeline.residency_pool->host_output_frame_count != 0u &&
       (pipeline.residency_pool->first_host_output_frame >
            pipeline.residency_pool->authority().frame_capacity() ||
        pipeline.residency_pool->host_output_frame_count >
            pipeline.residency_pool->authority().frame_capacity() -
                pipeline.residency_pool->first_host_output_frame)) ||
      pipeline.residency_input >= pipeline.resources.size() ||
      pipeline.residency_output >= pipeline.resources.size() ||
      pipeline.residency_input == pipeline.residency_output ||
      input->backing->size_bytes() != input->bytes ||
      state.output->backing->size_bytes() != state.output->bytes) {
    return false;
  }

  inputs = OrdinaryProjectionInputs{
      .input = input,
      .pipeline = &pipeline,
      .stream = &stream,
      .input_page_bytes = input_page_bytes,
      .output_page_bytes = output_page_bytes,
      .capacity = capacity,
      .input_payload_bytes = input_payload_bytes,
      .output_payload_bytes = output_payload_bytes,
      .input_prefix_bytes = input_prefix_bytes,
      .output_prefix_bytes = output_prefix_bytes,
      .input_arena_bytes = input_arena_bytes,
      .output_arena_bytes = output_arena_bytes,
      .host_input_arena_bytes = host_input_arena_bytes,
      .host_output_arena_bytes = host_output_arena_bytes,
  };
  return true;
}

} // namespace rund::compute::detail
