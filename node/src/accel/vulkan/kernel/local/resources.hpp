#pragma once

struct VulkanKernelEntry final {
  std::shared_ptr<void> resource{};
  std::shared_ptr<VulkanViewLowering> view{};
  VulkanKernelOps ops{};
  ResetSpan resets{};
  bool barrier_before{};
};

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanReset final {
  VulkanResidentBufferResult resident{};
  reset::Range range{};
  VkDeviceSize binding_offset{};
  VkDescriptorSet descriptor{VK_NULL_HANDLE};
};
#endif

struct VulkanKernelResources final {
  InlineIndexedStorage<VulkanKernelEntry, kInlineBoundStepCapacity> entries{};
  std::uint64_t dispatch_count = 0u;
  std::uint64_t reset_count = 0u;
  std::uint64_t reset_bytes = 0u;
  std::uint64_t traffic = 0u;
  KernelPreparationMode mode{KernelPreparationMode::Standalone};
  bool shared_scratch{};
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *adapter = nullptr;
  VulkanCommand command{};
  VulkanCollectivePipeline *reset_pipeline{};
  std::vector<VulkanCollectiveDescriptorLease> descriptor_leases{};
  std::vector<VulkanReset> resets{};
  submission::State<VulkanKernelResources> submission{};
#endif

  [[nodiscard]] bool reserve(const std::size_t step_count) {
    entries.resize(step_count);
    return entries.valid();
  }

  [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }

  [[nodiscard]] VulkanKernelEntry *entry(const std::size_t index) noexcept {
    return entries.get(index);
  }

  void release() {
    VulkanKernelEntry *const values = entries.data();
    if (values == nullptr) {
      return;
    }
    for (std::size_t index = 0u; index < entries.size(); ++index) {
      values[index].resource.reset();
      values[index].view.reset();
    }
  }
};
