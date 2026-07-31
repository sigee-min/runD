#pragma once

#include "../../backend/number.hpp"
#include "../../plan/validation.hpp"
#include "../../resident/window/admission/runtime/windows.hpp"
#include "../../sequence/input/pack.hpp"
#include "../../sequence/input/window.hpp"
#include "../../sequence/output.hpp"
#include "../adapter/api.hpp"
#include "../cached/pipeline.hpp"
#include "../command.hpp"
#include "../resident/bindings.hpp"
#include "../scope.hpp"
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr std::uint32_t kInlineDescriptorCount = 4u;

struct VulkanWindowBuffers {
  bool resident = false;
  StagedProof staged{};
  ScopedBuffer input{};
  ScopedBuffer output{};
  std::size_t output_size = 0u;
};

[[nodiscard]] VkDescriptorSet
DescriptorSetForPipeline(VulkanAdapter &adapter,
                         VulkanCachedPipeline &pipeline);
[[nodiscard]] bool MakeHostBuffer(VulkanAdapter &adapter, const void *bytes,
                                  rund::kernel::u64 byte_count,
                                  ScopedBuffer &out);
[[nodiscard]] bool
ResidentWindowSpan(const rund::kernel::ResidentBufferRef &ref,
                   const rund::kernel::ComputeDispatchWindow &window,
                   VkDeviceSize &offset, VkDeviceSize &range,
                   const char *&reason) noexcept;
[[nodiscard]] bool PrepareVulkanWindowBuffers(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings,
    std::span<const InputWindowPlan> input_plans,
    const VulkanResidentBindings *resident_bindings, VulkanWindowBuffers &out);
[[nodiscard]] bool UpdateVulkanWindowDescriptorSet(
    VulkanAdapter &adapter, VulkanCachedPipeline &pipeline,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings,
    std::span<const InputWindowPlan> input_plans,
    const ScopedBuffer &param_buffer,
    const VulkanResidentBindings *resident_bindings,
    const VulkanWindowBuffers &buffers, VkDescriptorSet &descriptor_set);
[[nodiscard]] bool
EncodeSubmitVulkanWindow(VulkanAdapter &adapter, VulkanCachedPipeline &pipeline,
                         const rund::kernel::ComputeDispatchWindow &window,
                         VkDescriptorSet descriptor_set,
                         const VulkanWindowBuffers &buffers);
[[nodiscard]] bool
FinishVulkanWindowReadback(VulkanAdapter &adapter,
                           const rund::kernel::ComputeDispatchWindow &window,
                           const rund::kernel::BindingSet &bindings,
                           const VulkanWindowBuffers &buffers);
[[nodiscard]] bool
ExecuteWindow(VulkanAdapter &adapter, VulkanCachedPipeline &pipeline,
              const rund::kernel::ComputePlan &plan,
              const rund::kernel::ComputeDispatchWindow &window,
              const rund::kernel::BindingSet &bindings,
              std::span<const InputWindowPlan> input_plans,
              const ScopedBuffer &param_buffer,
              const VulkanResidentBindings *resident_bindings);
#endif

} // namespace rund::node::accel::detail
