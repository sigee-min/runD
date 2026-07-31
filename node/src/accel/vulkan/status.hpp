#pragma once

#include "adapter/buffer.hpp"
#include "../kernel/preparation.hpp"

#include <kernel/core/model.hpp>

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

struct VulkanAdapter;
struct VulkanStorageBinding;

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanStatus final {
  VulkanBuffer device{};
  VulkanBuffer readback{};
  VkDeviceSize read_bytes{};
  bool pipeline{};
};

[[nodiscard]] bool CreateVulkanStatus(VulkanAdapter &adapter,
                                      VkDeviceSize bytes,
                                      VulkanStatus &status,
                                      KernelPreparationMode mode =
                                          CurrentKernelPreparationMode());

void ReleaseVulkanStatus(VulkanAdapter &adapter,
                         VulkanStatus &status) noexcept;

[[nodiscard]] bool ResetVulkanStatus(VkCommandBuffer command,
                                     const VulkanBuffer &status,
                                     VkDeviceSize bytes,
                                     std::uint32_t value = 0u) noexcept;

[[nodiscard]] bool ResetVulkanStatus(
    VkCommandBuffer command, const VulkanStorageBinding &status,
    VkDeviceSize bytes, std::uint32_t value = 0u) noexcept;

[[nodiscard]] bool ResetVulkanStatus(VkCommandBuffer command,
                                     const VulkanStatus &status,
                                     VkDeviceSize bytes,
                                     std::uint32_t value = 0u) noexcept;

[[nodiscard]] bool FinishVulkanStatus(
    VkCommandBuffer command, const VulkanStatus &status,
    std::span<const VulkanBuffer *const> outputs) noexcept;

[[nodiscard]] const rund::kernel::u32 *
VulkanStatusValue(const VulkanStatus &status) noexcept;
#endif

} // namespace rund::node::accel::detail
