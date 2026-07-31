#include "build.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
PrepareMetalPipeline(const std::span<const BackendBatchEntry> templates,
                     const std::span<const BackendBatchEntry> entries,
                     const std::span<const std::uint8_t> barriers,
                     const std::span<const TileTransducer> transducers,
                     const std::span<const BackendPublish> publications,
                     PreparedPipelineStatusLayout &status,
                     const bool profile_steps, std::shared_ptr<void> &prepared,
                     PreparedPipelineMemory &memory) {
  prepared.reset();
  memory = {};
  @autoreleasepool {
    const KernelPreparationScope preparation{
        KernelPreparationMode::PipelinePrivate};
    MetalPipelineBuild build{
        .templates = templates,
        .entries = entries,
        .barriers = barriers,
        .transducers = transducers,
        .publications = publications,
        .status = status,
        .profile_steps = profile_steps,
    };
    for (const auto stage :
         {&MetalPipelineBuild::Admit, &MetalPipelineBuild::Describe}) {
      const rund::AccelCheck result = (build.*stage)();
      if (!result.ok) {
        return result;
      }
    }
    const rund::AccelCheck allocated = build.Allocate(prepared, memory);
    if (!allocated.ok) {
      return allocated;
    }
    const rund::AccelCheck captured = build.Capture();
    if (!captured.ok) {
      return captured;
    }
    return build.Finalize(prepared, memory);
  }
}

#endif

} // namespace rund::node::accel::detail
