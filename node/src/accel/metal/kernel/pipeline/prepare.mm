#include "build.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
PrepareMetalPipeline(const std::span<const BackendBatchEntry> templates,
                     const std::span<const BackendBatchEntry> entries,
                     const std::span<const std::uint8_t> barriers,
                     const std::span<const TileTransducer> transducers,
                     const std::span<const NestedAggregate> aggregates,
                     const std::span<const BackendPublish> publications,
                     PreparedKernelTemplateRegistry &template_registry,
                     PreparedPipelineStatusLayout &status,
                     const bool profile_steps, std::shared_ptr<void> &prepared,
                     PreparedPipelineMemory &memory,
                     PreparedPipelineFailure &failure) {
  prepared.reset();
  memory = {};
  failure = {};
  @autoreleasepool {
    const KernelPreparationScope preparation{
        KernelPreparationMode::PipelinePrivate};
    MetalPipelineBuild build{
        .templates = templates,
        .entries = entries,
        .barriers = barriers,
        .transducers = transducers,
        .aggregates = aggregates,
        .publications = publications,
        .template_registry = template_registry,
        .status = status,
        .profile_steps = profile_steps,
    };
    build.failure_context.stage(PreparedPipelineFailureStage::BackendAdmission);
    const rund::AccelCheck admitted = build.Admit();
    if (!admitted.ok) {
      failure = build.failure_context.failure(admitted.reason);
      return admitted;
    }
    build.failure_context.stage(
        PreparedPipelineFailureStage::BackendDescription);
    const rund::AccelCheck described = build.Describe();
    if (!described.ok) {
      failure = build.failure_context.failure(described.reason);
      return described;
    }
    build.failure_context.stage(
        PreparedPipelineFailureStage::BackendAllocation);
    const rund::AccelCheck allocated = build.Allocate(prepared, memory);
    if (!allocated.ok) {
      failure = build.failure_context.failure(allocated.reason);
      return allocated;
    }
    build.failure_context.stage(PreparedPipelineFailureStage::BackendCapture);
    const rund::AccelCheck captured = build.Capture();
    if (!captured.ok) {
      failure = build.failure_context.failure(captured.reason);
      return captured;
    }
    // Finalization freezes the complete backend stream and has no single
    // Program owner. Do not leak Capture's last route into a global native or
    // cold-workspace failure coordinate.
    build.failure_context.clear_route();
    build.failure_context.stage(
        PreparedPipelineFailureStage::BackendFinalization);
    const rund::AccelCheck finalized = build.Finalize(prepared, memory);
    if (!finalized.ok) {
      failure = build.failure_context.failure(finalized.reason);
    }
    return finalized;
  }
}

#endif

} // namespace rund::node::accel::detail
