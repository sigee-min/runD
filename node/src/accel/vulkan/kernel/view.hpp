#pragma once

#include "../../kernel/backend/run.hpp"
#include "../../kernel/grid.hpp"
#include "../../kernel/memory.hpp"
#include "../../kernel/preparation.hpp"
#include "../buffer/resident/model.hpp"

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanCollectivePipeline;
struct VulkanAdapter;

struct ViewPage final {
  VkDescriptorSet descriptor = VK_NULL_HANDLE;
  std::uint64_t begin{};
  std::uint64_t count{};
  std::uint64_t base_bytes{};
  std::uint64_t span_bytes{};
  std::uint64_t external_words{};
  std::uint64_t dense_words{};
  Grid grid{};
};

struct VulkanViewTransfer final {
  std::uint64_t binding{};
  VulkanResidentBufferResult external{};
  VulkanResidentBufferResult dense{};
  std::vector<ViewPage> pages{};
  std::uint64_t count{};
  std::uint64_t element_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t stride_bytes{};
  bool input{};
  bool planned{};
};

struct VulkanViewLowering final {
  VulkanAdapter *adapter{};
  RunBinds binds{};
  BoundStep step{};
  std::vector<VulkanViewTransfer> transfers{};
  std::vector<std::uint32_t> transfer_by_binding{};
  VulkanCollectivePipeline *pipeline{};
  bool has_input{};

  VulkanViewLowering() = default;
  VulkanViewLowering(const VulkanViewLowering &) = delete;
  VulkanViewLowering &operator=(const VulkanViewLowering &) = delete;
};

[[nodiscard]] rund::AccelCheck PrepareVulkanViewLowering(
    const rund::AccelDevice &pick, const BoundStep &source,
    KernelPreparationMode mode, const KernelViewLayout *views,
    const RunBinds *view_binds, std::shared_ptr<VulkanViewLowering> &out);

// Called with the adapter preparation lock held after descriptor leasing is
// active. Resident dense storage was already allocated outside that lock.
[[nodiscard]] rund::AccelCheck
PrepareVulkanViewCommands(VulkanAdapter &adapter,
                          const std::shared_ptr<VulkanViewLowering> &view);

[[nodiscard]] rund::AccelCheck
EncodeVulkanViewInputs(const std::shared_ptr<VulkanViewLowering> &view,
                       VkCommandBuffer command);

[[nodiscard]] rund::AccelCheck
EncodeVulkanViewOutputs(const std::shared_ptr<VulkanViewLowering> &view,
                        VkCommandBuffer command);

[[nodiscard]] PreparedMemory
VulkanViewMemory(const std::shared_ptr<VulkanViewLowering> &view,
                 std::uint64_t budget, std::uint64_t &traffic) noexcept;

[[nodiscard]] std::uint64_t VulkanViewDispatchCount(
    const std::shared_ptr<VulkanViewLowering> &view) noexcept;

#endif

} // namespace rund::node::accel::detail
