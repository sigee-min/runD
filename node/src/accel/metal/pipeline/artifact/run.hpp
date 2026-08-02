#pragma once

#include "../guard.hpp"
#include "cache.hpp"
#include "compile.hpp"
#include "store.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::shared_ptr<void>
MetalPipelineForArtifact(MetalAdapter &adapter,
                         rund::kernel::LoweringArtifact artifact) {
  if (!artifact.ok) {
    SetMetalLastError(
        adapter, artifact.reason == nullptr ? "compute_pipeline_capacity"
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
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return {};
  }
  artifact.source_text =
      PipelinePrivateMetalSource(std::move(artifact.source_text), source_upper);
  if (artifact.source_text.empty()) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return {};
  }
  artifact.source_text_upper_bytes = source_upper;
  std::shared_ptr<void> cached = FindMetalMapArtifactPipeline(adapter, artifact);
  if (cached != nullptr) {
    return cached;
  }
  std::shared_ptr<void> pipeline =
      CompileMetalMapArtifactPipeline(adapter, artifact);
  if (pipeline == nullptr) {
    return {};
  }
  return StoreMetalMapArtifactPipeline(adapter, std::move(artifact), pipeline);
}

#else

std::shared_ptr<void>
MetalPipelineForArtifact(MetalAdapter &,
                         rund::kernel::LoweringArtifact) {
  return {};
}

#endif

} // namespace rund::node::accel::detail
