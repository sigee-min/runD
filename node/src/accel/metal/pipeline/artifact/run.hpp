#pragma once

#include "../guard.hpp"
#include "cache.hpp"
#include "compile.hpp"
#include "store.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::shared_ptr<void>
MetalPipelineForArtifact(MetalAdapter &adapter,
                         const rund::kernel::LoweringArtifact &artifact) {
  rund::kernel::LoweringArtifact guarded = artifact;
  guarded.source_text =
      PipelinePrivateMetalSource(std::move(guarded.source_text));
  if (guarded.source_text.empty()) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return {};
  }
  std::shared_ptr<void> cached = FindMetalMapArtifactPipeline(adapter, guarded);
  if (cached != nullptr) {
    return cached;
  }
  std::shared_ptr<void> pipeline =
      CompileMetalMapArtifactPipeline(adapter, guarded);
  if (pipeline == nullptr) {
    return {};
  }
  return StoreMetalMapArtifactPipeline(adapter, guarded, pipeline);
}

#else

std::shared_ptr<void>
MetalPipelineForArtifact(MetalAdapter &,
                         const rund::kernel::LoweringArtifact &) {
  return {};
}

#endif

} // namespace rund::node::accel::detail
