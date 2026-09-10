#include "internal.hpp"

#include "../backing.hpp"
#include "../../device/residency/pool.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/window/model.hpp>
#include <rund/compute/pipeline/capacity.hpp>

#include <cstdint>
#include <limits>

namespace rund::compute::detail::virtual_prepare_detail {

VirtualWindowPreflight preflight_window(
    const ProgramState &program, const VirtualGeometry &geometry,
    const VirtualBufferState &input, const VirtualBufferState &output,
    const std::uint64_t page_count, const ResidencyConfig config) noexcept {
  VirtualWindowPreflight result{};
  if (program.device == nullptr || program.device->backend == Backend::Cpu ||
      geometry.route != VirtualRoute::Window ||
      geometry.boundary !=
          static_cast<std::uint32_t>(kernel::WindowBoundary::Clamp) ||
      geometry.device_vsm_required || page_count < 2u ||
      page_count > std::numeric_limits<std::uint32_t>::max() ||
      input.backing == nullptr || output.backing == nullptr ||
      input.backing == output.backing ||
      (input.type != Type::I32 && input.type != Type::U32) ||
      output.type != input.type ||
      (geometry.operation != static_cast<std::uint32_t>(kernel::WindowOp::Sum) &&
       geometry.operation != static_cast<std::uint32_t>(kernel::WindowOp::Min) &&
       geometry.operation != static_cast<std::uint32_t>(kernel::WindowOp::Max)) ||
      geometry.input_prefix_elements == 0u ||
      geometry.input_prefix_elements != geometry.output_prefix_elements ||
      geometry.input_payload_elements == 0u ||
      geometry.input_payload_elements != geometry.output_payload_elements) {
    return result;
  }

  const auto &input_owner = VirtualBackingAccess::resident(*input.backing);
  const auto &output_owner = VirtualBackingAccess::resident(*output.backing);
  if ((input_owner == nullptr) != (output_owner == nullptr)) {
    return result;
  }
  const auto valid_owner = [&](const std::shared_ptr<BufferState> &owner,
                               const VirtualBufferState &buffer) noexcept {
    return owner == nullptr ||
           (owner->device == program.device && owner->type == buffer.type &&
            owner->bytes == buffer.bytes);
  };
  if (!valid_owner(input_owner, input) || !valid_owner(output_owner, output) ||
      (input_owner != nullptr && input_owner == output_owner)) {
    return result;
  }

  std::uint64_t input_bytes = 0u;
  std::uint64_t output_bytes = 0u;
  std::uint64_t input_frame_bytes = 0u;
  std::uint64_t output_frame_bytes = 0u;
  std::uint64_t frame_bytes = 0u;
  std::uint64_t twice_prefix = 0u;
  std::uint64_t input_shape = 0u;
  std::uint64_t output_shape = 0u;
  if (!kernel::checked::mul(input.count, input.element_bytes, input_bytes) ||
      !kernel::checked::mul(output.count, output.element_bytes, output_bytes) ||
      input_bytes != input.bytes || output_bytes != output.bytes ||
      !kernel::checked::mul(geometry.input_frame_elements,
                            input.element_bytes, input_frame_bytes) ||
      !kernel::checked::mul(geometry.output_frame_elements,
                            output.element_bytes, output_frame_bytes) ||
      !kernel::checked::add(input_frame_bytes, output_frame_bytes,
                            frame_bytes) ||
      !kernel::checked::mul(geometry.input_prefix_elements, 2u,
                            twice_prefix) ||
      !kernel::checked::add(twice_prefix, geometry.input_payload_elements,
                            input_shape) ||
      !kernel::checked::add(twice_prefix, geometry.output_payload_elements,
                            output_shape) ||
      input_shape != geometry.input_frame_elements ||
      output_shape != geometry.output_frame_elements ||
      input_frame_bytes > std::numeric_limits<std::size_t>::max() ||
      output_frame_bytes > std::numeric_limits<std::size_t>::max()) {
    return result;
  }

  constexpr std::uint64_t k = 2u;
  const std::uint64_t banks =
      static_cast<std::uint64_t>(residency::Pool::BankCount);
  const residency::PoolLayout layout{
      .input_type = input.type,
      .input_format = input.format,
      .output_type = output.type,
      .output_format = output.format,
      .input_page_bytes = input_frame_bytes,
      .output_page_bytes = output_frame_bytes,
      .frame_capacity = static_cast<std::uint32_t>(k),
      .host_frame_capacity = static_cast<std::uint32_t>(page_count),
      .host_output_frame_capacity = static_cast<std::uint32_t>(k),
  };
  residency::PoolFootprint footprint{};
  std::uint64_t device_storage = 0u;
  if (banks == 0u ||
      !residency::project_pool_footprint(layout, program.device->backend,
                                         footprint) ||
      footprint.host_storage_bytes == 0u ||
      !kernel::checked::mul(frame_bytes, k, device_storage) ||
      !kernel::checked::mul(device_storage, banks, device_storage)) {
    return result;
  }
  const std::uint64_t host_storage = footprint.host_storage_bytes;
  if ((config.device_resident_bytes != 0u &&
       config.device_resident_bytes < device_storage) ||
      (config.host_resident_bytes != 0u &&
       config.host_resident_bytes < host_storage)) {
    result.mode = VirtualWindowPreflightMode::RollingPool;
  } else {
    VirtualHostRingCapacities host{};
    if (!project_virtual_host_ring_capacities(
            page_count, k, input_frame_bytes, output_frame_bytes, host_storage,
            std::numeric_limits<std::uint32_t>::max(), host) ||
        host.input != page_count || host.output != k ||
        host.storage_bytes != host_storage) {
      return result;
    }
    result.mode = VirtualWindowPreflightMode::WindowRing;
    result.host = host;
  }
  result.endpoint = input_owner == nullptr
                        ? VirtualWindowPreflightEndpoint::Staged
                        : VirtualWindowPreflightEndpoint::Resident;
  result.page_count = page_count;
  result.frame_capacity = k;
  result.input_bytes = input_bytes;
  result.output_bytes = output_bytes;
  result.input_frame_bytes = input_frame_bytes;
  result.output_frame_bytes = output_frame_bytes;
  result.frame_bytes = frame_bytes;
  result.device_storage_bytes = device_storage;
  return result;
}

} // namespace rund::compute::detail::virtual_prepare_detail
