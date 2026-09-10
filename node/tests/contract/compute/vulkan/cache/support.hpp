#pragma once

#include <cstdint>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <type_traits>
#endif

namespace rund::node::vulkan_cache_contract {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
template <typename Handle>
[[nodiscard]] Handle FakeHandle(const std::uintptr_t value) noexcept {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(value);
  } else {
    return static_cast<Handle>(value);
  }
}
#endif

[[nodiscard]] int DescriptorRange();
[[nodiscard]] int MapBias();
[[nodiscard]] int VulkanCollectiveSourceIdentity();
[[nodiscard]] int VulkanShaderCache();

} // namespace rund::node::vulkan_cache_contract
