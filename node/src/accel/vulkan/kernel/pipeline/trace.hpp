#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck
EnsureVulkanPipelineDispatchTrace(VulkanPipeline &pipeline) noexcept;
[[nodiscard]] rund::AccelCheck
EncodeVulkanPipelineDispatchTrace(VulkanPipeline &pipeline) noexcept;
[[nodiscard]] rund::AccelCheck
FoldVulkanPipelineDispatchTrace(VulkanPipeline &pipeline,
                                rund::RuntimeStats &stats) noexcept;

#endif

} // namespace rund::node::accel::detail
