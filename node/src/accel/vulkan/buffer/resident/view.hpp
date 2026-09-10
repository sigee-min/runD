#pragma once

#include "model.hpp"

#include <accel/device.hpp>

#include "../../../backend/result.hpp"

#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] BackendHostView ReadVulkanResidentBuffer(
    const rund::AccelDevice &pick,
    const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle) noexcept;

[[nodiscard]] BackendHostWriteView WriteVulkanResidentBuffer(
    const rund::AccelDevice &pick,
    const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle) noexcept;

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
