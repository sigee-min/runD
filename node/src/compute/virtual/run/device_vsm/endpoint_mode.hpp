#pragma once

#include "../projection.hpp"

#include "../../../device/state.hpp"
#include "../../../type.hpp"
#include "../../backing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace rund::compute::detail::device_vsm_product_detail {

enum class EndpointMode : std::uint8_t {
  Invalid,
  Resident,
  Staged,
};

struct EndpointSet final {
  EndpointMode mode{EndpointMode::Invalid};
  std::array<std::shared_ptr<BufferState>, VirtualPipelineState::InputCapacity>
      inputs{};
  std::shared_ptr<BufferState> output{};
};

namespace endpoint_mode_detail {

[[nodiscard]] inline bool
resident_matches(const std::shared_ptr<BufferState> &owner,
                 const std::shared_ptr<DeviceState> &device, const Type type,
                 const std::uint64_t active_bytes) noexcept {
  if (owner == nullptr || device == nullptr || active_bytes == 0u ||
      type_bytes(type) == 0u) {
    return false;
  }
  const AccelBufferState *const storage = accel_buffer(*owner);
  return owner->device == device && owner->type == type &&
         active_bytes <= owner->bytes && storage != nullptr &&
         storage->buffer && owner->bytes == storage->buffer.byte_extent;
}

[[nodiscard]] inline bool
check(const std::shared_ptr<VirtualBufferState> &buffer,
      const std::shared_ptr<DeviceState> &device, const Type type,
      const std::uint64_t bytes, bool &resident,
      std::shared_ptr<BufferState> &owner) noexcept {
  resident = false;
  owner.reset();
  if (buffer == nullptr || buffer->backing == nullptr || device == nullptr ||
      bytes == 0u || type_bytes(type) == 0u) {
    return false;
  }
  owner = VirtualBackingAccess::resident(*buffer->backing);
  if (owner == nullptr) {
    return true;
  }
  resident = true;
  return resident_matches(owner, device, type, bytes);
}

} // namespace endpoint_mode_detail

[[nodiscard]] inline EndpointMode
classify_endpoints(const VirtualPipelineState &state,
                   const VirtualRunProjection &run,
                   EndpointSet &result) noexcept {
  EndpointSet found{};
  if (state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.output == nullptr || run.input_count == 0u ||
      run.input_count != state.input_count ||
      run.input_count > VirtualPipelineState::InputCapacity) {
    result = {};
    return EndpointMode::Invalid;
  }

  bool any_resident = false;
  bool any_staged = false;
  const auto visit = [&](const std::shared_ptr<VirtualBufferState> &buffer,
                         const Type type, const std::uint64_t bytes,
                         std::shared_ptr<BufferState> &owner) noexcept {
    bool resident = false;
    if (!endpoint_mode_detail::check(buffer, state.pipeline->device, type,
                                     bytes, resident, owner)) {
      return false;
    }
    any_resident = any_resident || resident;
    any_staged = any_staged || !resident;
    return true;
  };

  for (std::size_t index = 0u; index < run.input_count; ++index) {
    if (!visit(state.inputs[index], run.input_type, run.active.input_bytes,
               found.inputs[index])) {
      result = {};
      return EndpointMode::Invalid;
    }
  }
  if (!visit(state.output, run.output_type, run.active.output_bytes,
             found.output) ||
      any_resident == any_staged) {
    result = {};
    return EndpointMode::Invalid;
  }

  found.mode = any_resident ? EndpointMode::Resident : EndpointMode::Staged;
  result = std::move(found);
  return result.mode;
}

} // namespace rund::compute::detail::device_vsm_product_detail
