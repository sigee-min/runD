#pragma once

#include "../../../../kernel/residency/persistent_sliding.hpp"
#include "../../../../kernel/residency/sliding.hpp"
#include "../state.hpp"

#include <span>

namespace rund::node::accel::detail {

struct PreparedKernelPipeline;
struct PreparedResidencyPersistentSlidingRole;

[[nodiscard]] PersistentResidencySlidingCapability
QueryVulkanPreparedPersistentSlidingCapability(
    std::span<const PreparedResidencyPersistentSlidingRole>, std::uint64_t,
    ResidencySlidingMemory, PersistentResidencySlidingMode) noexcept;

// BackendOps-visible entry points exist in both SDK and fail-closed stub
// builds. SDK-private helpers remain below the feature boundary.
[[nodiscard]] rund::AccelCheck
VulkanPipelineResidencyReady(const std::shared_ptr<void> &prepared,
                             bool &ready) noexcept;
[[nodiscard]] rund::AccelCheck
StageVulkanPipelineResidency(const std::shared_ptr<void> &prepared,
                             std::shared_ptr<void> &candidate,
                             std::uint64_t &retained_bytes) noexcept;
void CommitVulkanPipelineResidency(const std::shared_ptr<void> &prepared,
                                   std::shared_ptr<void> candidate) noexcept;
[[nodiscard]] BackendResidencySlidingCapability
VulkanResidencySlidingCapability(const std::shared_ptr<void> &prepared,
                                 ResidencySlidingMemory memory) noexcept;
[[nodiscard]] rund::AccelCheck SubmitVulkanResidencySliding(
    const std::shared_ptr<void> &prepared,
    const BackendResidencySlidingDescriptor &, KernelCompletion, void *,
    KernelTiming, PipelineSubmitMode, std::span<const std::uint32_t>) noexcept;
[[nodiscard]] rund::AccelCheck
SubmitVulkanResidencyWindow(const BackendResidencyWindowRequest &) noexcept;
[[nodiscard]] rund::AccelCheck
SignalVulkanResidencyWindow(const std::shared_ptr<void> &,
                            const BackendResidencyWindowSignal &) noexcept;
[[nodiscard]] rund::AccelCheck
AbortVulkanResidencyWindow(const std::shared_ptr<void> &,
                           const BackendResidencyWindowAbort &) noexcept;
[[nodiscard]] BackendResidencySchedulePreparation
PrepareVulkanResidencySchedule(std::span<const BackendResidencyScheduleRole>,
                               std::uint64_t epoch_count,
                               std::size_t tail_local_count) noexcept;
[[nodiscard]] rund::AccelCheck
SubmitVulkanResidencySchedule(const BackendResidencyScheduleRequest &) noexcept;
[[nodiscard]] rund::AccelCheck
SignalVulkanResidencySchedule(const std::shared_ptr<void> &,
                              const BackendResidencyWindowSignal &) noexcept;
[[nodiscard]] rund::AccelCheck
AbortVulkanResidencySchedule(const std::shared_ptr<void> &,
                             const BackendResidencyWindowAbort &) noexcept;
[[nodiscard]] PersistentResidencySlidingPreparation
PrepareVulkanResidencyPersistent(
    const PersistentResidencySlidingRequest &) noexcept;

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool InspectVulkanResidencyAdmission(
    const PreparedKernelPipeline &prepared,
    VulkanResidencyAdmissionSnapshot &snapshot) noexcept;

[[nodiscard]] bool
InspectVulkanResidencyStatus(const PreparedKernelPipeline &prepared,
                             VulkanResidencyStatus &status) noexcept;

[[nodiscard]] bool InspectVulkanGraphGeneratedDiagnostics(
    const PreparedKernelPipeline &prepared,
    VulkanResidencyGraphGeneratedDiagnostics &diagnostics) noexcept;

void CompleteVulkanPipeline(void *, KernelResult) noexcept;

[[nodiscard]] bool
EvaluateVulkanResidencyStatus(const VulkanPipeline &pipeline,
                              VulkanResidencyStatus &status) noexcept;

[[nodiscard]] rund::AccelCheck
PrepareVulkanPipelineResidency(VulkanPipeline &pipeline) noexcept;

[[nodiscard]] rund::AccelCheck
PrepareVulkanResidencySlidingGate(VulkanPipeline &pipeline,
                                  VulkanResidencySelection &selection) noexcept;

void DestroyVulkanResidencySlidingGate(
    VulkanResidencySelection &selection) noexcept;

// Source-private physical fault used only by the MoltenVK descriptor test. It
// preserves the previously published row for one submission so the GPU, not
// the Host, must reject the stale descriptor.
[[nodiscard]] bool InjectVulkanResidencySlidingStaleDescriptorOnce(
    const std::shared_ptr<void> &prepared) noexcept;

struct VulkanResidencySlidingDiagnostics final {
  std::uint64_t submit_frontier_count{};
  std::uint64_t terminal_frontier_count{};
  std::uint64_t gpu_result_read_count{};
  bool terminal_frontier_released{};
  bool gate_quarantined{};
  bool adapter_quarantined{};
};

[[nodiscard]] bool InspectVulkanResidencySliding(
    const std::shared_ptr<void> &prepared,
    VulkanResidencySlidingDiagnostics &diagnostics) noexcept;
[[nodiscard]] bool InjectVulkanResidencySlidingTerminalFrontierPauseOnce(
    const std::shared_ptr<void> &prepared) noexcept;
[[nodiscard]] bool ReleaseVulkanResidencySlidingTerminalFrontier(
    const std::shared_ptr<void> &prepared) noexcept;

[[nodiscard]] rund::AccelCheck BuildVulkanResidencySubmission(
    const VulkanPipeline &pipeline, std::span<const std::uint32_t> locals,
    std::span<VkCommandBuffer> storage, std::size_t &command_count,
    std::uint64_t &dispatch_count, std::uint64_t &control_count,
    std::uint64_t &reset_count, std::uint64_t &reset_bytes) noexcept;

[[nodiscard]] bool
SelectVulkanResidencyArguments(VulkanPipeline &,
                               std::span<const std::uint32_t> locals,
                               bool execute) noexcept;

[[nodiscard]] bool
SeedVulkanResidencyControl(VulkanPipeline &,
                           const BackendResidencyWindowSignal &,
                           bool execute) noexcept;

[[nodiscard]] KernelResult ObserveVulkanResidency(
    VulkanPipeline &, rund::AccelCheck admission, std::uint64_t dispatch_count,
    std::uint64_t control_count, std::uint64_t reset_count,
    std::uint64_t reset_bytes, std::uint32_t control_generation) noexcept;

[[nodiscard]] PersistentResidencySlidingCapability
VulkanResidencyPersistentCapability(
    std::span<const PersistentResidencySlidingRole>, std::uint64_t,
    ResidencySlidingMemory) noexcept;
[[nodiscard]] const PersistentResidencySlidingServiceOps &
VulkanResidencyPersistentServiceOps() noexcept;

[[nodiscard]] rund::AccelCheck
SubmitVulkanResidencyPersistent(const PersistentResidencySlidingRequest &,
                                PersistentResidencySlidingControl &) noexcept;

[[nodiscard]] rund::AccelCheck WaitVulkanResidencyPersistentDone(
    PersistentResidencySlidingControl &,
    const PersistentResidencySlidingDoneWait &,
    PersistentResidencySlidingDoneObservation &) noexcept;

[[nodiscard]] rund::AccelCheck SignalVulkanResidencyPersistentReady(
    PersistentResidencySlidingControl &,
    const PersistentResidencySlidingReadySignal &) noexcept;

[[nodiscard]] rund::AccelCheck AcknowledgeVulkanResidencyPersistentDone(
    PersistentResidencySlidingControl &,
    const PersistentResidencySlidingAcknowledgeDone &) noexcept;

[[nodiscard]] rund::AccelCheck FailVulkanResidencyPersistentService(
    PersistentResidencySlidingControl &,
    const PersistentResidencySlidingServiceFailure &) noexcept;
void QuarantineVulkanResidencyPersistentUnknown(
    PersistentResidencySlidingControl &) noexcept;

#endif

} // namespace rund::node::accel::detail
