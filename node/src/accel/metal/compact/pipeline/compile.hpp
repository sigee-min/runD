#pragma once

#include "cache.hpp"
#include "../../pipeline/named.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool CompileMissingMetalCompactPipelines(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    MetalCompactPipelines& pipelines,
    const bool status_required,
    const bool block_offsets) {
  if (block_offsets && pipelines.count_blocks == nullptr &&
      !MakeNamedMetalPipeline(device, library,
                              "rund_compute_compact_count_blocks",
                              pipelines.count_blocks)) {
    return false;
  }
  if (block_offsets && pipelines.scatter_blocks == nullptr &&
      !MakeNamedMetalPipeline(
          device, library, "rund_compute_compact_scatter_block_offsets",
          pipelines.scatter_blocks)) {
    return false;
  }
  if (!block_offsets && pipelines.scatter == nullptr &&
      !MakeNamedMetalPipeline(device, library, "rund_compute_compact_scatter",
                              pipelines.scatter)) {
    return false;
  }
  if (status_required && pipelines.status == nullptr &&
      !MakeNamedMetalPipeline(device, library, "rund_compute_compact_status",
                              pipelines.status)) {
    return false;
  }
  return true;
}

[[nodiscard]] inline bool CompileMetalCompactPipelineLibrary(
    MetalAdapter& adapter,
    MetalCompactPipelines& pipelines,
    const bool status_required,
    const bool block_offsets) {
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalCompactSource());
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  return library != nil &&
         CompileMissingMetalCompactPipelines(device, library, pipelines,
                                             status_required, block_offsets);
}
#endif

}  // namespace rund::node::accel::detail
