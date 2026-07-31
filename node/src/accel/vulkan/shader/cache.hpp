#pragma once

#include "../adapter/shader.hpp"

#include <cstddef>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::size_t kVulkanSpirvCacheCapacity = 256u;
inline constexpr std::size_t kVulkanSpirvCacheByteCapacity =
    16u * 1024u * 1024u;

[[nodiscard]] bool FindValidatedVulkanSpirv(
    std::string_view compiler, std::string_view validator,
    std::string_view source, VulkanShader& shader);

void CacheValidatedVulkanSpirv(std::string_view compiler,
                               std::string_view validator,
                               std::string_view source,
                               const VulkanShader& shader);

[[nodiscard]] std::size_t ValidatedVulkanSpirvCacheSize();
[[nodiscard]] std::size_t ValidatedVulkanSpirvCacheBytes();
void ClearValidatedVulkanSpirvCache();

#endif

} // namespace rund::node::accel::detail
