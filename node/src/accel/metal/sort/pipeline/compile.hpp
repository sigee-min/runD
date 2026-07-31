#pragma once

#include "cache.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool MetalSortPipelineMatchesShape(
    const std::shared_ptr<void> &owner) {
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)owner.get();
  return pipeline != nil &&
         pipeline.threadExecutionWidth == kMetalSortSimdWidth &&
         pipeline.maxTotalThreadsPerThreadgroup >= kMetalSortThreadCount;
}

[[nodiscard]] inline bool MetalSortPipelinesMatchShape(
    const MetalSortPipelines &pipelines) {
  return MetalSortPipelineMatchesShape(pipelines.dispatch) &&
         MetalSortPipelineMatchesShape(pipelines.histogram) &&
         MetalSortPipelineMatchesShape(pipelines.prefix) &&
         MetalSortPipelineMatchesShape(pipelines.base) &&
         MetalSortPipelineMatchesShape(pipelines.scatter);
}

[[nodiscard]] inline bool MakeMetalSortPipeline(
    id<MTLDevice> device, id<MTLLibrary> library,
    NSString *const function_name, std::shared_ptr<void> &out) {
  if (!MakeNamedMetalPipeline(device, library, function_name, out) ||
      !MetalSortPipelineMatchesShape(out)) {
    out.reset();
    return false;
  }
  return true;
}

[[nodiscard]] inline bool
CompileMissingMetalSortPipelines(id<MTLDevice> device, id<MTLLibrary> library,
                                 const rund::kernel::SortKey key,
                                 MetalSortPipelines &pipelines) {
  return MakeMetalSortPipeline(device, library, @"rund_compute_sort_dispatch",
                               pipelines.dispatch) &&
         MakeMetalSortPipeline(
             device, library,
             SortFunctionName("rund_compute_sort_histogram", key),
             pipelines.histogram) &&
         MakeMetalSortPipeline(device, library, @"rund_compute_sort_prefix",
                               pipelines.prefix) &&
         MakeMetalSortPipeline(device, library, @"rund_compute_sort_base",
                               pipelines.base) &&
         MakeMetalSortPipeline(
             device, library,
             SortFunctionName("rund_compute_sort_scatter", key),
             pipelines.scatter);
}

[[nodiscard]] inline bool CompileMetalSortPipelineLibrary(
    MetalAdapter &adapter, const rund::kernel::SortKey key,
    const rund::kernel::u32 block_size, MetalSortPipelines &pipelines) {
  if (block_size != kMetalSortBlockSize) {
    return false;
  }
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalSortSource(block_size));
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  return library != nil &&
         CompileMissingMetalSortPipelines(device, library, key, pipelines) &&
         MetalSortPipelinesMatchShape(pipelines);
}
#endif

} // namespace rund::node::accel::detail
