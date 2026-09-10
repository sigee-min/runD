#pragma once

#include "../local.hpp"

#include <span>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// One owner for terminal lifecycle mutation.  A non-empty pending mask is the
// pre-submit transaction being rejected; an empty mask preserves the already
// submitted frontier used by terminal classification.
void quarantine(
    VulkanPipeline &pipeline, VulkanResidencySelection &selection,
    const char *reason = nullptr,
    std::span<const bool> pending = {}) noexcept;

[[nodiscard]] bool add_totals(
    VulkanResidencySelection &selection,
    const VulkanResidencyGraphGeneratedDiagnostics &diagnostics) noexcept;

void close_known(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
                 const char *reason) noexcept;

[[nodiscard]] bool rearm(VulkanPipeline &pipeline,
                         VulkanResidencySelection &selection) noexcept;

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
