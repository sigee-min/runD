#pragma once

#include "../adapter/shader.hpp"
#include "../adapter/state.hpp"

namespace rund::kernel {
struct ComputePlan;
struct LoweringArtifact;
} // namespace rund::kernel

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool CompileVulkanShader(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact, VulkanShader &shader);

[[nodiscard]] bool CompileVulkanSourceWithTools(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact, VulkanShader &shader);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
