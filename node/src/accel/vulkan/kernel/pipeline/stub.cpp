#include "../../../kernel/backend/execute.hpp"
#include "transfer.hpp"

namespace rund::node::accel::detail {

#if !defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanPipeline(
    const std::span<const BackendBatchEntry>,
    const std::span<const BackendBatchEntry>,
    const std::span<const std::uint8_t>, const std::span<const TileTransducer>,
    const std::span<const NestedAggregate>,
    const std::span<const BackendPublish>, PreparedKernelTemplateRegistry &,
    PreparedPipelineStatusLayout &, const bool, std::shared_ptr<void> &prepared,
    PreparedPipelineMemory &memory, PreparedPipelineMemoryMeter *,
    rund::AccelRunFacts &preparation, PreparedPipelineFailure &failure) {
  prepared.reset();
  memory = {};
  preparation = {};
  failure = PreparedPipelineFailure{
      .stage = PreparedPipelineFailureStage::BackendAdmission,
      .native_reason_key = "accel_vulkan_unavailable",
  };
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
SeedPreparedVulkanPipelineGeneration(const std::shared_ptr<void> &,
                                     const std::uint32_t) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck PrepareVulkanPipelineTransfer(const std::shared_ptr<void> &,
                                               const UploadRoute &,
                                               const DownloadRoute &,
                                               const std::uint64_t) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

BackendUpload UploadPreparedVulkanPipeline(const std::shared_ptr<void> &,
                                           const void *,
                                           const std::uint64_t) noexcept {
  return BackendUpload{.check = {false, "accel_vulkan_unavailable"}};
}

BackendDownload DownloadPreparedVulkanPipeline(const std::shared_ptr<void> &,
                                               void *, const std::uint64_t,
                                               std::uint64_t *) noexcept {
  return BackendDownload{.check = {false, "accel_vulkan_unavailable"}};
}

rund::AccelCheck SubmitPreparedVulkanPipeline(const std::shared_ptr<void> &,
                                              KernelCompletion, void *,
                                              KernelTiming,
                                              PipelineSubmitMode) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
