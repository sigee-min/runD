#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

#include <algorithm>
#include <cstdint>

namespace rund::node::accel::detail {

rund::AccelCheck EncodeMetalMap(MetalAdapter &adapter,
                                const std::shared_ptr<void> &resources,
                                void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const map = static_cast<MetalMapEncodeResources *>(resources.get());
  id<MTLComputeCommandEncoder> encoder =
      (__bridge id<MTLComputeCommandEncoder>)command_encoder;
  if (map == nullptr || map->adapter != &adapter || encoder == nil ||
      map->windows.empty()) {
    SetMetalLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)map->pipeline.get();
  if (pipeline == nil || map->param.buffer == nullptr) {
    SetMetalLastError(adapter, "accel_metal_command_unavailable");
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:(__bridge id<MTLBuffer>)map->param.buffer.get()
              offset:0u
             atIndex:0u];
  if (map->iterations != 1u) {
    [encoder setBytes:&map->iterations
               length:sizeof(map->iterations)
              atIndex:static_cast<NSUInteger>(
                          map->plan.input_buffer_count +
                          map->plan.output_buffer_count + 1u)];
  }
  id<MTLBuffer> indirect_args = nil;
  if (map->controlled()) {
    id<MTLComputePipelineState> const control_pipeline =
        (__bridge id<MTLComputePipelineState>)map->control_pipeline.get();
    indirect_args =
        (__bridge id<MTLBuffer>)map->control_args.buffer.get();
    id<MTLBuffer> const control_params =
        (__bridge id<MTLBuffer>)map->control_params.buffer.get();
    id<MTLBuffer> const control_status =
        (__bridge id<MTLBuffer>)map->control_status.buffer.get();
    id<MTLBuffer> const count =
        map->control.has_count()
            ? (__bridge id<MTLBuffer>)map->control_count.device_buffer.get()
            : indirect_args;
    id<MTLBuffer> const predicate =
        map->control.has_predicate()
            ? (__bridge id<MTLBuffer>)map->control_predicate.device_buffer.get()
            : indirect_args;
    if (control_pipeline == nil || indirect_args == nil ||
        control_params == nil || control_status == nil || count == nil ||
        predicate == nil) {
      SetMetalLastError(adapter, "accel_metal_command_unavailable");
      return rund::AccelCheck{false, "accel_metal_command_unavailable"};
    }
    if (!map->checks.empty()) {
      id<MTLComputePipelineState> const check_pipeline =
          (__bridge id<MTLComputePipelineState>)map->check_pipeline.get();
      if (check_pipeline == nil) {
        SetMetalLastError(adapter, "accel_metal_command_unavailable");
        return rund::AccelCheck{false, "accel_metal_command_unavailable"};
      }
      [encoder setComputePipelineState:check_pipeline];
      [encoder setBuffer:count
                  offset:static_cast<NSUInteger>(
                             map->control_count.ref.offset_bytes +
                             map->control.count_byte_offset)
                 atIndex:0u];
      [encoder setBuffer:predicate
                  offset:static_cast<NSUInteger>(
                             map->control_predicate.ref.offset_bytes +
                             map->control.predicate_byte_offset)
                 atIndex:1u];
      for (std::size_t index = 0u; index < map->checks.size(); ++index) {
        const MetalMapCheck check = map->checks[index];
        const auto *const ref =
            map->bindings.resident_inputs.ref(check.binding);
        const MetalResidentBufferResult &resident =
            map->resident.input(check.binding);
        if (ref == nullptr || !resident.check.ok ||
            resident.device_buffer == nullptr) {
          SetMetalLastError(adapter, "compute_binding_mismatch");
          return rund::AccelCheck{false, "compute_binding_mismatch"};
        }
        [encoder setBuffer:(__bridge id<MTLBuffer>)resident.device_buffer.get()
                    offset:static_cast<NSUInteger>(ref->offset_bytes)
                   atIndex:static_cast<NSUInteger>(index + 2u)];
      }
      [encoder setBuffer:control_params
                  offset:static_cast<NSUInteger>(map->control_config_offset)
                 atIndex:static_cast<NSUInteger>(map->checks.size() + 2u)];
      [encoder setBuffer:control_status
                  offset:0u
                 atIndex:static_cast<NSUInteger>(map->checks.size() + 3u)];
      [encoder dispatchThreads:MTLSizeMake(256u, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder setComputePipelineState:control_pipeline];
    [encoder setBuffer:count
                offset:static_cast<NSUInteger>(
                           map->control_count.ref.offset_bytes +
                           map->control.count_byte_offset)
               atIndex:0u];
    [encoder setBuffer:predicate
                offset:static_cast<NSUInteger>(
                           map->control_predicate.ref.offset_bytes +
                           map->control.predicate_byte_offset)
               atIndex:1u];
    [encoder setBuffer:indirect_args offset:0u atIndex:2u];
    [encoder setBuffer:control_params offset:0u atIndex:3u];
    [encoder setBuffer:control_params
                offset:static_cast<NSUInteger>(map->control_config_offset)
               atIndex:4u];
    [encoder setBuffer:control_status offset:0u atIndex:5u];
    const NSUInteger count_windows = map->windows.size();
    const NSUInteger width = std::max<NSUInteger>(
        1u, std::min<NSUInteger>(
                count_windows,
                [control_pipeline maxTotalThreadsPerThreadgroup]));
    [encoder dispatchThreads:MTLSizeMake(count_windows, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)map->param.buffer.get()
                offset:0u
               atIndex:0u];
  }
  std::size_t window_index = 0u;
  for (const rund::kernel::ComputeDispatchWindow &window : map->windows) {
    if (!EncodeResidentWindow(adapter, encoder, pipeline, map->plan, window,
                              map->bindings, map->resident, map->read_routes,
                              indirect_args,
                              static_cast<NSUInteger>(window_index * 4u *
                                                      sizeof(std::uint32_t)))) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
    ++window_index;
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
