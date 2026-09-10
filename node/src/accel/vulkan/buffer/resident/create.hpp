#pragma once

#include "../../../backend/result.hpp"
#include "model.hpp"

#include <accel/device.hpp>

namespace rund::node::accel::detail {

struct ResidentDesc;

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] VulkanResidentBufferResult CreateVulkanResidentBuffer(
    const rund::AccelDevice &pick, const ResidentDesc &desc,
    bool zero_initialize = false,
    BackendBufferMemory memory = BackendBufferMemory::DeviceLocal,
    std::uint64_t exact_storage_bytes = 0u);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
