#include "pipeline/artifact/cache.hpp"
#include "pipeline/artifact/compile.hpp"
#include "pipeline/guard.hpp"
#include "state.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::shared_ptr<void>
MetalPipelineForArtifact(MetalAdapter &adapter,
                         rund::kernel::LoweringArtifact artifact) {
  SetMetalLastError(adapter, "compute_artifact_pipeline_preparing");
  if (!artifact.ok) {
    SetMetalLastError(adapter, artifact.reason == nullptr
                                   ? "compute_pipeline_capacity"
                                   : artifact.reason);
    return {};
  }
  std::uint64_t source_upper = 0u;
  const bool pipeline_private =
      IsPipelinePrivatePreparation(CurrentKernelPreparationMode());
  if (!PipelinePrivateMetalSourceUpperBytes(
          std::max<std::uint64_t>(artifact.source_text.size(),
                                  artifact.source_text_upper_bytes),
          1u, pipeline_private, source_upper)) {
    SetMetalLastError(adapter, "compute_artifact_source_capacity");
    return {};
  }
  artifact.source_text =
      PipelinePrivateMetalSource(std::move(artifact.source_text), source_upper);
  if (artifact.source_text.empty()) {
    SetMetalLastError(adapter, "compute_artifact_pipeline_guard_invalid");
    return {};
  }
  artifact.source_text_upper_bytes = source_upper;
  std::shared_ptr<void> cached =
      FindMetalMapArtifactPipeline(adapter, artifact);
  if (cached != nullptr) {
    return cached;
  }
  std::shared_ptr<void> pipeline =
      CompileMetalMapArtifactPipeline(adapter, artifact);
  if (pipeline == nullptr) {
    return {};
  }
  std::shared_ptr<void> stored =
      StoreMetalMapArtifactPipeline(adapter, std::move(artifact), pipeline);
  if (stored == nullptr) {
    SetMetalLastError(adapter, "compute_artifact_pipeline_publish_failed");
  }
  return stored;
}

#else

std::shared_ptr<void> MetalPipelineForArtifact(MetalAdapter &,
                                               rund::kernel::LoweringArtifact) {
  return {};
}

#endif

} // namespace rund::node::accel::detail
