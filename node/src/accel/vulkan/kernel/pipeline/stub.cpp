#include "../../../kernel/backend/execute.hpp"

namespace rund::node::accel::detail {

#if !defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanPipeline(const std::span<const BackendBatchEntry>,
                                       const std::span<const BackendBatchEntry>,
                                       const std::span<const std::uint8_t>,
                                       const std::span<const TileTransducer>,
                                       const std::span<const BackendPublish>,
                                       PreparedPipelineStatusLayout &,
                                       const bool,
                                       std::shared_ptr<void> &prepared,
                                       PreparedPipelineMemory &memory) {
  prepared.reset();
  memory = {};
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
SeedPreparedVulkanPipelineGeneration(const std::shared_ptr<void> &,
                                     const std::uint32_t) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck SubmitPreparedVulkanPipeline(const std::shared_ptr<void> &,
                                              KernelCompletion,
                                              void *) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
