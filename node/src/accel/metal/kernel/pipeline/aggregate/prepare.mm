#include "prepare.hpp"

#include "source.hpp"

#include "../../../pipeline/named.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck PrepareMetalNestedAggregate(MetalAdapter &adapter,
                                             MetalNestedAggregate &aggregate) {
  for (const std::shared_ptr<void> &buffer : aggregate.buffers) {
    if (buffer == nullptr) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
  }
  const KernelPreparationScope unguarded{KernelPreparationMode::Standalone};
  id<MTLDevice> const device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil || aggregate.params.outer_bound == 0u) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  constexpr const char *reduce_key = "pipeline.nested.aggregate.reduce.u32";
  constexpr const char *finalize_key = "pipeline.nested.aggregate.finalize.u32";
  aggregate.reduce_pipeline = LookupMetalNamedPipeline(adapter, reduce_key);
  aggregate.finalize_pipeline = LookupMetalNamedPipeline(adapter, finalize_key);
  if (aggregate.reduce_pipeline == nullptr ||
      aggregate.finalize_pipeline == nullptr) {
    const std::string source{MetalNestedAggregateSource()};
    const std::uint64_t begin = MonotonicNanoseconds();
    const std::shared_ptr<void> library_owner =
        AcquireMetalLibrary(adapter, source);
    id<MTLLibrary> const library = (__bridge id<MTLLibrary>)library_owner.get();
    if (library == nil) {
      return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
    }
    if (aggregate.reduce_pipeline == nullptr &&
        !MakeNamedMetalPipeline(device, library,
                                "rund_pipeline_nested_aggregate_reduce_u32",
                                aggregate.reduce_pipeline)) {
      return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
    }
    if (aggregate.finalize_pipeline == nullptr &&
        !MakeNamedMetalPipeline(device, library,
                                "rund_pipeline_nested_aggregate_finalize_u32",
                                aggregate.finalize_pipeline)) {
      return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
    }
    const std::uint64_t compile_ns = MonotonicNanoseconds() - begin;
    StoreMetalNamedPipeline(adapter, reduce_key, aggregate.reduce_pipeline,
                            compile_ns);
    StoreMetalNamedPipeline(adapter, finalize_key, aggregate.finalize_pipeline,
                            0u);
  }
  id<MTLComputePipelineState> const reduce =
      (__bridge id<MTLComputePipelineState>)aggregate.reduce_pipeline.get();
  id<MTLComputePipelineState> const finalize =
      (__bridge id<MTLComputePipelineState>)aggregate.finalize_pipeline.get();
  if (reduce == nil || finalize == nil || reduce.threadExecutionWidth == 0u ||
      reduce.maxTotalThreadsPerThreadgroup == 0u) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  const NSUInteger simd = reduce.threadExecutionWidth;
  if ((simd & (simd - 1u)) != 0u ||
      simd > reduce.maxTotalThreadsPerThreadgroup) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  // The second reduction is performed by one SIMD group, so the number of
  // first-level partials may not exceed either its lane count or the fixed
  // 32-entry threadgroup arrays. Never launch SIMD groups that cannot own a
  // tile element; small tiles otherwise pay for permanently idle lanes.
  if (simd > std::numeric_limits<std::uint32_t>::max()) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  const std::uint64_t tile_groups = std::max<std::uint64_t>(
      1u,
      (static_cast<std::uint64_t>(aggregate.params.tile) + simd - 1u) / simd);
  const NSUInteger groups =
      std::min<NSUInteger>({reduce.maxTotalThreadsPerThreadgroup / simd, simd,
                            32u, static_cast<NSUInteger>(tile_groups)});
  aggregate.threads = simd * groups;
  if (aggregate.threads == 0u) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  aggregate.workgroup_count =
      static_cast<std::uint64_t>(aggregate.params.outer_bound) + 1u;
  aggregate.work_item_count =
      static_cast<std::uint64_t>(aggregate.params.outer_bound) *
          aggregate.threads +
      1u;
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck EncodeMetalNestedAggregate(
    const MetalNestedAggregate &aggregate, id<MTLBuffer> const control,
    id<MTLBuffer> const step_control, RUNDMetalPipelineCapture *const encoder) {
  id<MTLComputePipelineState> const reduce =
      (__bridge id<MTLComputePipelineState>)aggregate.reduce_pipeline.get();
  id<MTLComputePipelineState> const finalize =
      (__bridge id<MTLComputePipelineState>)aggregate.finalize_pipeline.get();
  if (reduce == nil || finalize == nil || control == nil ||
      step_control == nil || encoder == nil || aggregate.threads == 0u ||
      aggregate.params.outer_bound == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::array<id<MTLBuffer>, 7u> buffers{};
  for (NSUInteger index = 0u; index < aggregate.buffers.size(); ++index) {
    buffers[index] = (__bridge id<MTLBuffer>)aggregate.buffers[index].get();
    if (buffers[index] == nil) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
  }

  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:buffers[0u] offset:0u atIndex:0u];
  [encoder setBuffer:buffers[1u] offset:0u atIndex:1u];
  [encoder setBuffer:buffers[2u] offset:0u atIndex:2u];
  [encoder setBytes:&aggregate.params
             length:sizeof(aggregate.params)
            atIndex:3u];
  [encoder setBuffer:buffers[5u] offset:0u atIndex:4u];
  [encoder setBuffer:buffers[6u] offset:0u atIndex:5u];
  [encoder
       dispatchThreadgroups:MTLSizeMake(aggregate.params.outer_bound, 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(aggregate.threads, 1u, 1u)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

  [encoder setComputePipelineState:finalize];
  [encoder setBuffer:buffers[2u] offset:0u atIndex:0u];
  [encoder setBuffer:buffers[3u] offset:0u atIndex:1u];
  [encoder setBuffer:buffers[4u] offset:0u atIndex:2u];
  [encoder setBuffer:control offset:0u atIndex:3u];
  [encoder setBytes:&aggregate.params
             length:sizeof(aggregate.params)
            atIndex:4u];
  [encoder setBuffer:step_control offset:0u atIndex:5u];
  [encoder setBuffer:buffers[5u] offset:0u atIndex:6u];
  [encoder setBuffer:buffers[6u] offset:0u atIndex:7u];
  [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
