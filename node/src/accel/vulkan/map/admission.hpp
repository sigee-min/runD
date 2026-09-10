#pragma once

#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool PrepareVulkanMapControl(
    const rund::AccelDevice &, const BoundControl &,
    const std::vector<rund::kernel::ComputeDispatchWindow> &,
    VulkanMapEncodeResources &);

[[nodiscard]] bool BindVulkanGeneratedMapAdmission(
    VulkanMapEncodeResources &, const VulkanBuffer &, const VulkanBuffer &)
    noexcept;

#endif

} // namespace rund::node::accel::detail
