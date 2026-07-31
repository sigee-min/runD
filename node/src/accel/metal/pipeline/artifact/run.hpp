#pragma once

#include "cache.hpp"
#include "compile.hpp"
#include "store.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

std::shared_ptr<void> MetalPipelineForArtifact(
    MetalAdapter& adapter,
    const rund::kernel::LoweringArtifact& artifact) {
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
  return StoreMetalMapArtifactPipeline(adapter, artifact, pipeline);
}

#else

std::shared_ptr<void> MetalPipelineForArtifact(
    MetalAdapter&,
    const rund::kernel::LoweringArtifact&) {
  return {};
}

#endif

}  // namespace rund::node::accel::detail
