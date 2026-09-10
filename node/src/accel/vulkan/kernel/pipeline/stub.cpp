#include "../../../backend/result.hpp"

#include "../../../kernel/backend/execute.hpp"
#include "residency/local.hpp"
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

rund::AccelCheck QueryVulkanPipelineResidency(const std::shared_ptr<void> &,
                                              bool &supported) noexcept {
  supported = false;
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck VulkanPipelineResidencyReady(const std::shared_ptr<void> &,
                                              bool &ready) noexcept {
  ready = false;
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
StageVulkanPipelineResidency(const std::shared_ptr<void> &,
                             std::shared_ptr<void> &candidate,
                             std::uint64_t &retained_bytes) noexcept {
  candidate.reset();
  retained_bytes = 0u;
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

void CommitVulkanPipelineResidency(const std::shared_ptr<void> &,
                                   std::shared_ptr<void>) noexcept {}

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

rund::AccelCheck
SubmitPreparedVulkanPipeline(const std::shared_ptr<void> &, KernelCompletion,
                             void *, KernelTiming, PipelineSubmitMode,
                             std::span<const std::uint32_t>) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

BackendResidencySlidingCapability
VulkanResidencySlidingCapability(const std::shared_ptr<void> &,
                                 const ResidencySlidingMemory memory) noexcept {
  BackendResidencySlidingCapability result{};
  result.check = rund::AccelCheck{false, "accel_vulkan_unavailable"};
  result.memory = memory;
  return result;
}

rund::AccelCheck SubmitVulkanResidencySliding(
    const std::shared_ptr<void> &, const BackendResidencySlidingDescriptor &,
    KernelCompletion, void *, KernelTiming, PipelineSubmitMode,
    std::span<const std::uint32_t>) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
SubmitVulkanResidencyWindow(const BackendResidencyWindowRequest &) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
SignalVulkanResidencyWindow(const std::shared_ptr<void> &,
                            const BackendResidencyWindowSignal &) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
AbortVulkanResidencyWindow(const std::shared_ptr<void> &,
                           const BackendResidencyWindowAbort &) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

BackendResidencySchedulePreparation
PrepareVulkanResidencySchedule(std::span<const BackendResidencyScheduleRole>,
                               const std::uint64_t,
                               const std::size_t) noexcept {
  BackendResidencySchedulePreparation result{};
  result.check = rund::AccelCheck{false, "accel_vulkan_unavailable"};
  return result;
}

rund::AccelCheck SubmitVulkanResidencySchedule(
    const BackendResidencyScheduleRequest &) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
SignalVulkanResidencySchedule(const std::shared_ptr<void> &,
                              const BackendResidencyWindowSignal &) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

rund::AccelCheck
AbortVulkanResidencySchedule(const std::shared_ptr<void> &,
                             const BackendResidencyWindowAbort &) noexcept {
  return rund::AccelCheck{false, "accel_vulkan_unavailable"};
}

PersistentResidencySlidingPreparation PrepareVulkanResidencyPersistent(
    const PersistentResidencySlidingRequest &) noexcept {
  PersistentResidencySlidingPreparation result{};
  result.capability.check = rund::AccelCheck{false, "accel_vulkan_unavailable"};
  return result;
}

#endif

} // namespace rund::node::accel::detail
