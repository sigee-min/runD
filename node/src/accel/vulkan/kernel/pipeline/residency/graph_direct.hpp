#pragma once

#include "../state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
BuildGraphDirectProof(const VulkanPipeline &pipeline,
                      VulkanResidencyGraphStageDirectProof &proof) noexcept;

[[nodiscard]] bool
GraphDirectProofMatches(const VulkanPipeline &pipeline,
                        const VulkanResidencySelection &selection) noexcept;

#endif

} // namespace rund::node::accel::detail
