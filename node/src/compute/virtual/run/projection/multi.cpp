#include "identity.hpp"
#include "internal.hpp"

#include "../../../device/state.hpp"
#include "../../backing.hpp"
#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/model.hpp>

#include <array>

namespace rund::compute::detail {

bool project_virtual_multi_run(VirtualPipelineState &state,
                               const std::uint64_t active_count,
                               VirtualRunProjection &projection) noexcept {
  const VirtualBufferState *const first = virtual_input(state);
  const VirtualGeometry &geometry = state.geometry;
  const bool pointwise = geometry.route == VirtualRoute::MultiPointwise;
  const bool scan = geometry.route == VirtualRoute::Scan;
  if ((!pointwise && !scan) || !geometry.device_vsm_required ||
      first == nullptr || state.input_count < 2u ||
      state.input_count > VirtualPipelineState::InputCapacity ||
      state.output == nullptr || state.pipeline == nullptr ||
      state.alternate_pipeline == nullptr ||
      state.pipeline->device == nullptr ||
      state.pipeline->device->backend == Backend::Cpu ||
      geometry.input_payload_elements == 0u ||
      geometry.output_payload_elements != geometry.input_payload_elements ||
      geometry.input_frame_elements < geometry.input_payload_elements ||
      geometry.output_frame_elements < geometry.output_payload_elements ||
      geometry.input_prefix_elements + geometry.input_payload_elements !=
          geometry.input_frame_elements ||
      geometry.output_prefix_elements + geometry.output_payload_elements !=
          geometry.output_frame_elements ||
      (scan && first->type != Type::U64) || first->element_bytes == 0u ||
      state.output->element_bytes != first->element_bytes ||
      active_count > first->count || active_count > state.output->count) {
    return false;
  }

  std::uint64_t input_page_bytes = 0u;
  std::uint64_t input_payload_bytes = 0u;
  std::uint64_t output_page_bytes = 0u;
  std::uint64_t output_payload_bytes = 0u;
  std::uint64_t input_prefix_bytes = 0u;
  std::uint64_t output_prefix_bytes = 0u;
  std::uint64_t dirty_offset = 0u;
  if (!kernel::checked::mul(geometry.input_frame_elements, first->element_bytes,
                            input_page_bytes) ||
      !kernel::checked::mul(geometry.input_payload_elements,
                            first->element_bytes, input_payload_bytes) ||
      !kernel::checked::mul(geometry.output_frame_elements,
                            state.output->element_bytes, output_page_bytes) ||
      !kernel::checked::mul(geometry.output_payload_elements,
                            state.output->element_bytes,
                            output_payload_bytes) ||
      !kernel::checked::mul(geometry.input_prefix_elements,
                            first->element_bytes, input_prefix_bytes) ||
      !kernel::checked::mul(geometry.output_prefix_elements,
                            state.output->element_bytes, output_prefix_bytes) ||
      !kernel::checked::mul(input_page_bytes, state.input_count,
                            dirty_offset) ||
      input_page_bytes == 0u || input_payload_bytes == 0u ||
      output_page_bytes != input_page_bytes ||
      output_payload_bytes != input_payload_bytes ||
      state.output->backing == nullptr ||
      state.output->backing->size_bytes() != state.output->bytes) {
    return false;
  }

  const std::uint64_t full_pages =
      first->count / geometry.input_payload_elements +
      static_cast<std::uint64_t>(
          first->count % geometry.input_payload_elements != 0u);
  const std::uint64_t pipeline_width = scan ? 2u : 1u;
  const residency::StreamPlan capacity{
      full_pages, pipeline_width,
      residency::DirtyRange{.offset = dirty_offset, .bytes = output_page_bytes},
      state.output->bytes, 0u};
  VirtualActiveProjection active{};
  if (!project_virtual_active(capacity, active_count, first->count,
                              geometry.input_payload_elements, active_count,
                              first->element_bytes, state.output->element_bytes,
                              false, active)) {
    return false;
  }

  const residency::Identity identity = input_materialization_identity(
      first->type, first->format, input_page_bytes, input_payload_bytes,
      input_prefix_bytes, geometry.input_frame_elements, geometry.boundary);
  std::array<VirtualRunInputProjection, VirtualPipelineState::InputCapacity>
      inputs{};
  for (std::size_t index = 0u; index < state.input_count; ++index) {
    const VirtualBufferState *const input = virtual_input(state, index);
    if (input == nullptr || input->backing == nullptr ||
        input->type != first->type || input->format != first->format ||
        input->count != first->count ||
        input->element_bytes != first->element_bytes ||
        input->bytes != first->bytes ||
        input->backing->size_bytes() != input->bytes) {
      return false;
    }
    inputs[index] = VirtualRunInputProjection{
        .capacity_bytes = input->bytes,
        .visible_bytes = active.input_bytes,
        .identity_hi = identity.hi,
        .identity_lo = identity.lo,
        .backing = VirtualBackingAccess::id(*input->backing),
        .version = VirtualBackingAccess::version(*input->backing),
        .type = input->type,
    };
  }

  projection = VirtualRunProjection{
      .active = active,
      .inputs = inputs,
      .input_count = state.input_count,
      .input_capacity_bytes = first->bytes,
      .input_visible_bytes = active.input_bytes,
      .cache_extent = active.active_count,
      .input_identity_hi = identity.hi,
      .input_identity_lo = identity.lo,
      .result_identity_hi = geometry.materialization_hi,
      .result_identity_lo = geometry.materialization_lo,
      .output_backing = VirtualBackingAccess::id(*state.output->backing),
      .output_version = VirtualBackingAccess::version(*state.output->backing),
      .input_page_bytes = input_page_bytes,
      .output_page_bytes = output_page_bytes,
      .input_payload_bytes = input_payload_bytes,
      .output_payload_bytes = output_payload_bytes,
      .input_prefix_bytes = input_prefix_bytes,
      .output_prefix_bytes = output_prefix_bytes,
      .input_frame_elements = geometry.input_frame_elements,
      .device_vsm_required = true,
      .topology = scan ? VirtualRunTopology::MultiScan
                       : VirtualRunTopology::MultiPointwise,
      .input_type = first->type,
      .output_type = state.output->type,
      .operation = geometry.operation,
      .frame_capacity = pipeline_width,
  };
  return true;
}

} // namespace rund::compute::detail
