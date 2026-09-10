#include "ordinary.hpp"
#include "identity.hpp"
#include "internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../device/state.hpp"
#include "../../backing.hpp"
#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/model.hpp>
#include <kernel/program/compute/window/model.hpp>

#include <array>

namespace rund::compute::detail {

bool project_virtual_ordinary_run(VirtualPipelineState &state,
                                  const std::uint64_t active_count,
                                  VirtualRunProjection &projection) noexcept {
  OrdinaryProjectionInputs inputs{};
  if (!validate_virtual_ordinary_projection(state, inputs)) {
    return false;
  }
  VirtualBufferState &input = *inputs.input;
  PipelineState &pipeline = *inputs.pipeline;
  const VirtualGeometry &geometry = state.geometry;

  VirtualActiveProjection active{};
  const std::uint64_t active_output_count =
      geometry.route == VirtualRoute::Reduction ? static_cast<std::uint64_t>(1u)
                                                : active_count;
  if (inputs.stream->frame_capacity() != inputs.capacity ||
      !project_virtual_active(
          *inputs.stream, active_count, input.count,
          geometry.input_payload_elements, active_output_count,
          input.element_bytes, state.output->element_bytes,
          geometry.route == VirtualRoute::Reduction, active)) {
    return false;
  }

  std::array<std::byte *, residency::Pool::BankCount> input_host_banks{};
  std::array<std::byte *, residency::Pool::BankCount> output_host_banks{};
  if (pipeline.device->backend == Backend::Cpu) {
    for (std::size_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
      CpuBufferState *const input_buffer =
          cpu_buffer(*pipeline.residency_pool->input[bank]);
      CpuBufferState *const output_buffer =
          cpu_buffer(*pipeline.residency_pool->output[bank]);
      if (input_buffer == nullptr || output_buffer == nullptr ||
          input_buffer->data == nullptr || output_buffer->data == nullptr ||
          input_buffer->bytes < inputs.input_arena_bytes ||
          output_buffer->bytes != inputs.output_arena_bytes) {
        return false;
      }
      input_host_banks[bank] = input_buffer->data.get();
      output_host_banks[bank] = output_buffer->data.get();
    }
  } else {
    std::byte *const host_storage = pipeline.residency_pool->host_storage.get();
    std::uint64_t bank_bytes = 0u;
    if (host_storage == nullptr ||
        !kernel::checked::add(inputs.host_input_arena_bytes,
                              inputs.host_output_arena_bytes, bank_bytes)) {
      return false;
    }
    for (std::size_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
      const std::size_t bank_offset =
          static_cast<std::size_t>(bank * bank_bytes);
      input_host_banks[bank] = host_storage + bank_offset;
      output_host_banks[bank] =
          host_storage + bank_offset +
          static_cast<std::size_t>(inputs.host_input_arena_bytes);
    }
  }

  const residency::Identity input_identity = input_materialization_identity(
      input.type, input.format, inputs.input_page_bytes,
      inputs.input_payload_bytes, inputs.input_prefix_bytes,
      geometry.input_frame_elements, geometry.boundary);
  const residency::Identity canonical_input_identity =
      input_materialization_identity(
          input.type, input.format, inputs.input_payload_bytes,
          inputs.input_payload_bytes, 0u, geometry.input_payload_elements, 0u);
  projection = VirtualRunProjection{
      .active = active,
      .inputs = {VirtualRunInputProjection{
          .capacity_bytes = input.bytes,
          .visible_bytes = active.input_bytes,
          .identity_hi = input_identity.hi,
          .identity_lo = input_identity.lo,
          .backing = VirtualBackingAccess::id(*input.backing),
          .version = VirtualBackingAccess::version(*input.backing),
          .type = input.type,
      }},
      .input_count = 1u,
      .input_capacity_bytes = input.bytes,
      // Boundary pages bind to this invocation's exact visible prefix. Full
      // pages remain extent-independent; cache_extent distinguishes the two.
      .input_visible_bytes = active.input_bytes,
      .cache_extent = active.active_count,
      .input_identity_hi = input_identity.hi,
      .input_identity_lo = input_identity.lo,
      .canonical_input_identity_hi = canonical_input_identity.hi,
      .canonical_input_identity_lo = canonical_input_identity.lo,
      .result_identity_hi = geometry.materialization_hi,
      .result_identity_lo = geometry.materialization_lo,
      .input_backing = VirtualBackingAccess::id(*input.backing),
      .input_version = VirtualBackingAccess::version(*input.backing),
      .output_backing = VirtualBackingAccess::id(*state.output->backing),
      .output_version = VirtualBackingAccess::version(*state.output->backing),
      .input_page_bytes = inputs.input_page_bytes,
      .output_page_bytes = inputs.output_page_bytes,
      .input_payload_bytes = inputs.input_payload_bytes,
      .output_payload_bytes = inputs.output_payload_bytes,
      .input_prefix_bytes = inputs.input_prefix_bytes,
      .output_prefix_bytes = inputs.output_prefix_bytes,
      .input_frame_elements = geometry.input_frame_elements,
      .clamp_window = geometry.route == VirtualRoute::Window &&
                      geometry.boundary == static_cast<std::uint32_t>(
                                               kernel::WindowBoundary::Clamp),
      .clip_window = geometry.route == VirtualRoute::Window &&
                     geometry.boundary == static_cast<std::uint32_t>(
                                              kernel::WindowBoundary::Clip),
      .device_vsm_required = geometry.device_vsm_required,
      .topology = geometry.route == VirtualRoute::Reduction
                      ? VirtualRunTopology::Reduction
                  : geometry.route == VirtualRoute::Scan
                      ? VirtualRunTopology::Scan
                  : geometry.route == VirtualRoute::Window
                      ? VirtualRunTopology::LocalWindow
                      : VirtualRunTopology::Direct,
      .input_type = input.type,
      .output_type = state.output->type,
      .operation = geometry.operation,
      .frame_capacity = inputs.capacity,
      .input_arena_bytes = inputs.input_arena_bytes,
      .output_arena_bytes = inputs.output_arena_bytes,
      .input_regions = pipeline.residency_pool->input_regions,
      .output_regions =
          {residency::FrameRegion{
               .tier = pipeline.device->backend == Backend::Cpu
                           ? residency::FrameTier::Host
                           : residency::FrameTier::Device,
               .role = residency::FrameRole::Output,
               .first = pipeline.residency_pool->first_output_frame,
               .count = static_cast<std::uint32_t>(inputs.capacity),
           },
           residency::FrameRegion{
               .tier = pipeline.device->backend == Backend::Cpu
                           ? residency::FrameTier::Host
                           : residency::FrameTier::Device,
               .role = residency::FrameRole::Output,
               .first = pipeline.residency_pool->first_output_frame +
                        static_cast<std::uint32_t>(inputs.capacity),
               .count = static_cast<std::uint32_t>(inputs.capacity),
           }},
      .first_output_frame = pipeline.residency_pool->first_output_frame,
      .first_host_input_frame = pipeline.residency_pool->first_host_input_frame,
      .host_input_count =
          pipeline.residency_pool->layout.graph_host_input_count,
      .host_frame_capacity =
          pipeline.residency_pool->layout.host_frame_capacity,
      .first_host_output_frame =
          pipeline.residency_pool->first_host_output_frame,
      .host_output_frame_capacity =
          pipeline.device->backend == Backend::Cpu
              ? 0u
              : pipeline.residency_pool->layout.host_output_frame_capacity,
      .input_host_banks = input_host_banks,
      .output_host_banks = output_host_banks,
  };
  return true;
}

} // namespace rund::compute::detail
