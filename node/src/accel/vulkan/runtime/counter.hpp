#pragma once

#include <rund/counter.hpp>
#include "../adapter/api.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline void RecordVulkanCommandSubmitWaitNs(VulkanAdapter &adapter,
                                            const std::uint64_t elapsed_ns) {
  ::rund::detail::counter::Accumulate(adapter.command_submit_count, 1u);
  ::rund::detail::counter::Accumulate(adapter.command_submit_wait_ns,
                                      elapsed_ns);
}

inline void RecordVulkanPipelineCreateNs(VulkanAdapter &adapter,
                                         const std::uint64_t elapsed_ns) {
  ::rund::detail::counter::Accumulate(adapter.pipeline_create_ns, elapsed_ns);
}

inline void RecordVulkanDescriptorSetupNs(VulkanAdapter &adapter,
                                          const std::uint64_t elapsed_ns) {
  ::rund::detail::counter::Accumulate(adapter.descriptor_setup_ns, elapsed_ns);
}

inline void RecordVulkanSpirvCompileNs(VulkanAdapter &adapter,
                                       const std::uint64_t elapsed_ns) {
  ::rund::detail::counter::Accumulate(adapter.spirv_compile_ns, elapsed_ns);
}

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
