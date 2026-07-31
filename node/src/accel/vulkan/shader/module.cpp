#include "module.hpp"

#include "../adapter/api.hpp"
#include "../adapter/state.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

VulkanModule::~VulkanModule() { reset(); }

VulkanModule::VulkanModule(VulkanModule &&other) noexcept {
  *this = std::move(other);
}

VulkanModule &VulkanModule::operator=(VulkanModule &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  adapter_ = std::exchange(other.adapter_, nullptr);
  handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
  return *this;
}

void VulkanModule::reset() noexcept {
  if (adapter_ != nullptr && handle_ != VK_NULL_HANDLE) {
    vkDestroyShaderModule(adapter_->device, handle_, nullptr);
    ::rund::detail::counter::Release(adapter_->shader_module_current, 1u);
  }
  adapter_ = nullptr;
  handle_ = VK_NULL_HANDLE;
}

rund::AccelCheck CreateVulkanModule(VulkanAdapter &adapter,
                                    const VulkanShader &shader,
                                    VulkanModule &module) noexcept {
  module.reset();
  if (shader.words == nullptr || shader.words->empty() ||
      shader.words->size() >
          std::numeric_limits<std::uint32_t>::max() / sizeof(std::uint32_t)) {
    SetVulkanLastError(adapter, "accel_vulkan_spirv_invalid");
    return rund::AccelCheck{false, "accel_vulkan_spirv_invalid"};
  }
  VkShaderModuleCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = shader.words->size() * sizeof(std::uint32_t);
  info.pCode = shader.words->data();
  VkShaderModule handle = VK_NULL_HANDLE;
  if (vkCreateShaderModule(adapter.device, &info, nullptr, &handle) !=
          VK_SUCCESS ||
      handle == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_shader_module_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_shader_module_unavailable"};
  }
  module.adapter_ = &adapter;
  module.handle_ = handle;
  ::rund::detail::counter::Accumulate(adapter.shader_module_current, 1u);
  ::rund::detail::counter::Accumulate(adapter.shader_module_create_count, 1u);
  adapter.shader_module_peak =
      std::max(adapter.shader_module_peak, adapter.shader_module_current);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
