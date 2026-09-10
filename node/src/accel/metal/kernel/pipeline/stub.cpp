#include "../../../kernel/backend/execute.hpp"
#include "../../kernel.hpp"

namespace rund::node::accel::detail {

#if !defined(__APPLE__) || !defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck PrepareMetalPipeline(
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
      .native_reason_key = "accel_metal_unavailable",
  };
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

rund::AccelCheck
SeedPreparedMetalPipelineGeneration(const std::shared_ptr<void> &,
                                    const std::uint32_t) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

rund::AccelCheck
StageMetalPipelineResidency(const std::shared_ptr<void> &,
                            std::shared_ptr<void> &candidate,
                            std::uint64_t &retained_bytes) noexcept {
  candidate.reset();
  retained_bytes = 0u;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

void CommitMetalPipelineResidency(const std::shared_ptr<void> &,
                                  std::shared_ptr<void>) noexcept {}

rund::AccelCheck QueryMetalPipelineResidency(const std::shared_ptr<void> &,
                                             bool &supported) noexcept {
  supported = false;
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck MetalPipelineResidencyReady(const std::shared_ptr<void> &,
                                            bool &ready) noexcept {
  ready = false;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

BackendResidencySlidingCapability
MetalResidencySlidingCapability(const std::shared_ptr<void> &,
                                const ResidencySlidingMemory memory) noexcept {
  BackendResidencySlidingCapability result{};
  result.check = rund::AccelCheck{false, "accel_metal_unavailable"};
  result.memory = memory;
  return result;
}

rund::AccelCheck SubmitMetalResidencySliding(
    const std::shared_ptr<void> &, const BackendResidencySlidingDescriptor &,
    KernelCompletion, void *, KernelTiming, PipelineSubmitMode,
    std::span<const std::uint32_t>) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

bool InspectMetalResidencySliding(
    const std::shared_ptr<void> &,
    MetalResidencySlidingDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  return false;
}

bool InjectMetalResidencySlidingStaleDescriptorOnce(
    const std::shared_ptr<void> &) noexcept {
  return false;
}

MetalPersistentResidencySlidingPreparation
PrepareMetalPersistentResidencySliding(
    const PersistentResidencySlidingRequest &) noexcept {
  MetalPersistentResidencySlidingPreparation result{};
  result.capability.check = rund::AccelCheck{false, "accel_metal_unavailable"};
  return result;
}

const PersistentResidencySlidingServiceOps &
MetalPersistentResidencySlidingServiceOps() noexcept {
  static constexpr PersistentResidencySlidingServiceOps unavailable{};
  return unavailable;
}

bool InspectMetalPersistentResidencySliding(
    const std::shared_ptr<void> &,
    MetalPersistentResidencySlidingDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  return false;
}

bool InspectMetalFusedDirectRecurrence(
    const std::shared_ptr<void> &,
    MetalFusedDirectRecurrenceDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  return false;
}

rund::AccelCheck
SubmitPreparedMetalPipeline(const std::shared_ptr<void> &, KernelCompletion,
                            void *, KernelTiming, PipelineSubmitMode,
                            std::span<const std::uint32_t>) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

rund::AccelCheck SubmitMetalResidencyWindow(
    const BackendResidencyWindowRequest &) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

rund::AccelCheck SignalMetalResidencyWindow(
    const std::shared_ptr<void> &,
    const BackendResidencyWindowSignal &) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

rund::AccelCheck AbortMetalResidencyWindow(
    const std::shared_ptr<void> &,
    const BackendResidencyWindowAbort &) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
