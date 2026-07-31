#pragma once

#include "../../buffer/resident/model.hpp"
#include "../../descriptor.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] inline VulkanStorageBinding
GatherResidentBinding(const VulkanResidentBufferResult &resident) noexcept {
  return VulkanStorageBindingFor(resident.device_buffer, resident.ref);
}

#endif

} // namespace rund::node::accel::detail
