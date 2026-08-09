#pragma once

#include "../command/model.hpp"

#include <accel/check.hpp>
#include <accel/runtime.hpp>

#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanDispatchTrace final {
  VulkanCommand command{};
  VkQueryPool queries{VK_NULL_HANDLE};
  std::vector<std::uint64_t> values{};
  std::uint32_t query_count{};
  bool ready{};
};

struct VulkanAdapter;
struct VulkanKernelResources;
class PreparedMemoryMeter;

void DestroyVulkanDispatchTrace(VulkanAdapter &adapter,
                                VulkanDispatchTrace &trace) noexcept;
[[nodiscard]] rund::AccelCheck
EnsureVulkanDispatchTrace(VulkanAdapter &adapter,
                          VulkanKernelResources &resources,
                          PreparedMemoryMeter *memory) noexcept;
[[nodiscard]] rund::AccelCheck
EncodeVulkanDispatchTrace(VulkanAdapter &adapter,
                          VulkanKernelResources &resources) noexcept;
[[nodiscard]] rund::AccelCheck
FoldVulkanDispatchTrace(VulkanAdapter &adapter, VulkanDispatchTrace &trace,
                        rund::RuntimeStats &stats) noexcept;

#endif

} // namespace rund::node::accel::detail
