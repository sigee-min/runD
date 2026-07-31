#pragma once

#include "../kernel/scratch.hpp"
#include "adapter/api.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

class VulkanScratch final {
public:
  VulkanScratch(const rund::AccelDevice &pick,
                const KernelScratchLayout &layout,
                const RunBinds &binds) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool used() const noexcept;
  [[nodiscard]] bool active() const noexcept;
  void reset() noexcept;
  [[nodiscard]] bool acquire(VkDeviceSize bytes, VkBufferUsageFlags usage,
                             VulkanBuffer &buffer) noexcept;

private:
  struct Page final {
    const VulkanBuffer *buffer{};
    VkDeviceSize base{};
    VkDeviceSize bytes{};
    VkDeviceSize used{};
  };

  VulkanAdapter *adapter_{};
  std::vector<Page> pages_{};
  bool valid_{};
  bool used_{};
};

class VulkanScratchScope final {
public:
  explicit VulkanScratchScope(VulkanScratch *scratch) noexcept;
  ~VulkanScratchScope();

  VulkanScratchScope(const VulkanScratchScope &) = delete;
  VulkanScratchScope &operator=(const VulkanScratchScope &) = delete;

private:
  VulkanScratch *prior_{};
};

[[nodiscard]] VulkanScratch *ActiveVulkanScratch() noexcept;

#endif

} // namespace rund::node::accel::detail
