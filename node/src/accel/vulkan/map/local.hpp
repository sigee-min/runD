#pragma once

#include "../../plan/validation.hpp"
#include "../../resident/window/admission/runtime/windows.hpp"
#include "../../sequence/input/window.hpp"
#include "../adapter/api.hpp"
#include "../buffer/resident/model.hpp"
#include "../cached/pipeline.hpp"
#include "../kernel.hpp"
#include "../resident/bindings.hpp"
#include "../scope.hpp"
#include "../status.hpp"
#include <cstdint>
#include <kernel/program/compute/graph/schema.hpp>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kVulkanMapInlineDescriptorCount = 4u;

struct VulkanMapCheck final {
  std::uint32_t binding{};
  std::uint64_t limit{};
  std::uint64_t base{};
  std::uint64_t offset{};
  std::uint64_t stride{};
};

struct VulkanMapEncodeResources {
  VulkanAdapter *adapter = nullptr;
  std::uint32_t iterations{1u};
  bool history_recurrence{};
  rund::kernel::ComputePlan plan{};
  rund::kernel::BindingSet bindings{};
  // BindingSet is a non-owning projection. Recurrence history may project
  // proof-owned refs, so retain that owner through the last encode/finish.
  std::shared_ptr<const void> binding_owner{};
  std::vector<InputWindowPlan> input_plans{};
  std::vector<VulkanMapCheck> checks{};
  std::vector<rund::kernel::ComputeDispatchWindow> windows{};
  VulkanResidentBindings resident{};
  ScopedBuffer param{};
  VulkanCachedPipeline *pipeline = nullptr;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> descriptor_sets{};
  rund::kernel::GraphControl control{};
  VulkanResidentBufferResult control_count{};
  VulkanResidentBufferResult control_predicate{};
  std::uint64_t count_base{};
  std::uint64_t predicate_base{};
  ScopedBuffer control_args{};
  VulkanCollectivePipeline *control_pipeline = nullptr;
  VkDescriptorSet control_descriptor = VK_NULL_HANDLE;
  VulkanCollectivePipeline *check_pipeline = nullptr;
  VkDescriptorSet check_descriptor = VK_NULL_HANDLE;
  VulkanStatus control_status{};

  [[nodiscard]] bool controlled() const noexcept {
    return control.has_count() || control.has_predicate() || !checks.empty();
  }
};

void DestroyVulkanMapEncodeResources(void *raw);
[[nodiscard]] bool AllocateVulkanMapDescriptorSets(
    VulkanAdapter &adapter, const VulkanCachedPipeline &pipeline,
    rund::kernel::u64 set_count, VkDescriptorPool &descriptor_pool,
    std::vector<VkDescriptorSet> &descriptor_sets);
[[nodiscard]] bool MakeVulkanMapHostBuffer(VulkanAdapter &adapter,
                                           const void *bytes,
                                           rund::kernel::u64 byte_count,
                                           ScopedBuffer &out);
[[nodiscard]] bool VulkanMapResidentWindowSpan(
    const VulkanAdapter &adapter, const rund::kernel::ResidentBufferRef &ref,
    const rund::kernel::ComputeDispatchWindow &window, VkDeviceSize &offset,
    VkDeviceSize &range, const char *&reason) noexcept;
[[nodiscard]] bool VulkanMapResidentOutputWindowSpan(
    const VulkanMapEncodeResources &map,
    const rund::kernel::ResidentBufferRef &ref,
    const rund::kernel::ComputeDispatchWindow &window, VkDeviceSize &offset,
    VkDeviceSize &range, const char *&reason) noexcept;
[[nodiscard]] bool ValidateVulkanMapHistoryOutputs(
    const VulkanMapEncodeResources &map, const char *&reason) noexcept;
#endif

} // namespace rund::node::accel::detail
