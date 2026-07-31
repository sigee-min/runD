#include "adapter/api.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CompileVulkanShader(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact, VulkanShader &shader) {
  shader = VulkanShader{};
  return CompileVulkanSourceWithTools(adapter, plan, artifact, shader);
}

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
