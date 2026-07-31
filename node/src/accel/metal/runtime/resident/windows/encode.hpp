#pragma once

#include "offset.hpp"

#include <algorithm>
#include <span>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool EncodeResidentWindow(
    MetalAdapter& adapter,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const MetalResidentBindings& resident,
    const std::span<const rund::kernel::ReadRoute> read_routes,
    id<MTLBuffer> const indirect_args,
    const NSUInteger indirect_offset) {
  if (encoder == nil || pipeline == nil || resident.bindings != &bindings ||
      plan.output_buffer_count == 0u) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return false;
  }
  const auto fail_encoding = [&adapter]() {
    SetMetalLastError(adapter, "compute_binding_mismatch");
    return false;
  };
  for (rund::kernel::u64 index = 0u; index < plan.input_buffer_count; ++index) {
    NSUInteger input_offset = 0u;
    rund::kernel::u64 input_range = 0u;
    const rund::kernel::ResidentBufferRef *const ref =
        bindings.resident_inputs.ref(index);
    const auto route =
        std::find_if(read_routes.begin(), read_routes.end(),
                     [&](const rund::kernel::ReadRoute value) {
                       return value.source == index;
                     });
    if (ref == nullptr) {
      return fail_encoding();
    }
    if (route == read_routes.end()) {
      if (!ResidentWindowOffset(*ref, window, input_offset, input_range)) {
        return fail_encoding();
      }
    } else {
      const rund::kernel::u64 last = route->count - 1u;
      if (route->count == 0u ||
          !rund::kernel::checked::mul(last, ref->stride_bytes) ||
          !rund::kernel::checked::add(last * ref->stride_bytes,
                                      ref->element_bytes) ||
          last * ref->stride_bytes + ref->element_bytes > ref->bytes ||
          ref->offset_bytes >
              ref->bytes - (last * ref->stride_bytes + ref->element_bytes)) {
        return fail_encoding();
      }
      input_offset = static_cast<NSUInteger>(ref->offset_bytes);
      input_range = last * ref->stride_bytes + ref->element_bytes;
    }
    const MetalResidentBufferResult& input = resident.input(index);
    if (!input.check.ok || input.device_buffer == nullptr) {
      return fail_encoding();
    }
    (void)input_range;
    [encoder setBuffer:(__bridge id<MTLBuffer>)input.device_buffer.get()
                offset:input_offset
               atIndex:static_cast<NSUInteger>(index + 1u)];
  }
  for (rund::kernel::u64 index = 0u; index < plan.output_buffer_count;
       ++index) {
    NSUInteger output_offset = 0u;
    rund::kernel::u64 output_range = 0u;
    const rund::kernel::ResidentBufferRef *const ref =
        bindings.resident_outputs.ref(index);
    if (ref == nullptr ||
        !ResidentWindowOffset(*ref, window, output_offset, output_range)) {
      return fail_encoding();
    }
    const MetalResidentBufferResult& output = resident.output(index);
    if (!output.check.ok || output.device_buffer == nullptr) {
      return fail_encoding();
    }
    (void)output_range;
    [encoder setBuffer:(__bridge id<MTLBuffer>)output.device_buffer.get()
                offset:output_offset
               atIndex:static_cast<NSUInteger>(
                           plan.input_buffer_count + index + 1u)];
  }
  const NSUInteger thread_count = static_cast<NSUInteger>(window.tile_count);
  const NSUInteger width =
      std::max<NSUInteger>(1u,
                           std::min<NSUInteger>(
                               thread_count,
                               [pipeline maxTotalThreadsPerThreadgroup]));
  if (indirect_args != nil) {
    [encoder setBuffer:indirect_args
                offset:indirect_offset + 3u * sizeof(std::uint32_t)
               atIndex:static_cast<NSUInteger>(
                           plan.input_buffer_count +
                           plan.output_buffer_count + 1u)];
    [encoder dispatchThreadgroupsWithIndirectBuffer:indirect_args
                               indirectBufferOffset:indirect_offset
                              threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
  } else {
    [encoder dispatchThreads:MTLSizeMake(thread_count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
  }
  return true;
}
#endif

}  // namespace rund::node::accel::detail
