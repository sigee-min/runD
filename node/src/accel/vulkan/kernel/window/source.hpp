#pragma once

#include <kernel/program/compute/artifact.hpp>

#include <cstdint>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] std::string_view VulkanWindowSourceText() noexcept;
[[nodiscard]] bool VulkanWindowSourceBytes(std::uint64_t &bytes) noexcept;
[[nodiscard]] std::string_view VulkanGateSourceText() noexcept;

[[nodiscard]] rund::kernel::LoweringArtifact VulkanWindowArtifact();
[[nodiscard]] rund::kernel::LoweringArtifact VulkanGateArtifact();

#endif

} // namespace rund::node::accel::detail
