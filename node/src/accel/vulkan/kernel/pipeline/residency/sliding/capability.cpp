#include "../mode.hpp"
#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

BackendResidencySlidingCapability
VulkanResidencySlidingCapability(const std::shared_ptr<void> &prepared,
                                 const ResidencySlidingMemory memory) noexcept {
  bool ready = false;
  const rund::AccelCheck checked =
      VulkanPipelineResidencyReady(prepared, ready);
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  const bool valid = checked.ok && ready && ValidVulkanPipeline(pipeline) &&
                     pipeline->residency != nullptr &&
                     pipeline->residency->ready;
  if (valid && vulkan_residency_detail::owns(pipeline->residency->mode)) {
    return BackendResidencySlidingCapability{
        .check = {false, "accel_vulkan_pipeline_selection_unavailable"}};
  }
  const bool physical = valid &&
                        pipeline->residency->sliding.ready_for_submit &&
                        !pipeline->residency->sliding.quarantined.load(
                            std::memory_order_acquire) &&
                        !pipeline->adapter->residency_quarantined.load(
                            std::memory_order_acquire) &&
                        vulkan_sliding_detail::coherent_storage(
                            pipeline->residency->sliding.descriptor) &&
                        pipeline->residency->sliding.ready != VK_NULL_HANDLE;
  if (!valid || memory != ResidencySlidingMemory::HostCoherent) {
    return BackendResidencySlidingCapability{
        .check = {false, valid ? "accel_vulkan_pipeline_selection_unavailable"
                               : checked.reason}};
  }
  if (!physical) {
    return BackendResidencySlidingCapability{
        .check = {false, "accel_vulkan_pipeline_selection_unavailable"}};
  }
  return BackendResidencySlidingCapability{
      .check = {true, "ok"},
      .memory = ResidencySlidingMemory::HostCoherent,
      .retained_bytes = pipeline->residency->sliding.retained_bytes,
      .transient_bytes = sizeof(VulkanResidencySlidingSubmissionStorage),
      .max_slots = static_cast<std::uint8_t>(ResidencySlidingCapacity),
      .callbacks_async = true,
      .descriptor_release_acquire = false,
      .flush_before_frontier = false,
      .integrated_copy = false,
      .distinct_transfer_queue = false,
  };
}

#endif

} // namespace rund::node::accel::detail
