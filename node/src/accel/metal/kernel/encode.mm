#include <accel/check.hpp>

#include "local.hpp"
#include "ops/table.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck EncodeMetalResets(MetalKernelResources &resources,
                                   const std::size_t step,
                                   id<MTLComputeCommandEncoder> encoder) {
  MetalKernelEntry *const entry =
      step < resources.size() ? resources.entry(step) : nullptr;
  if (entry == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
  }
  if (entry->resets.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (entry->resets.begin > resources.resets.size() ||
      entry->resets.count >
          resources.resets.size() - entry->resets.begin) {
    return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
  }
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)resources.reset_pipeline.get();
  if (encoder == nil || pipeline == nil) {
    return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
  }
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [encoder setComputePipelineState:pipeline];
  const std::size_t end = entry->resets.begin + entry->resets.count;
  for (std::size_t index = entry->resets.begin; index < end; ++index) {
    const MetalReset &clear = resources.resets[index];
    id<MTLBuffer> buffer =
        (__bridge id<MTLBuffer>)clear.resident.device_buffer.get();
    if (buffer == nil) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    [encoder setBuffer:buffer offset:0u atIndex:0u];
    constexpr std::uint64_t window = std::numeric_limits<std::uint32_t>::max();
    for (std::uint64_t base = 0u; base < clear.range.count(); base += window) {
      const std::uint64_t portion =
          std::min(window, clear.range.count() - base);
      const reset::Params params = reset::Bind(clear.range, base);
      const NSUInteger count = static_cast<NSUInteger>(portion);
      [encoder setBytes:&params length:sizeof(params) atIndex:1u];
      [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(count, 256u),
                                            1u, 1u)];
    }
  }
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck EncodeMetalStep(MetalAdapter &adapter,
                                 const MetalKernelOps &ops,
                                 const std::shared_ptr<void> &resources,
                                 id<MTLComputeCommandEncoder> encoder) {
  if (ops.encode != nullptr) {
    return ops.encode(adapter, resources, (__bridge void *)encoder);
  }
  return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
}

rund::AccelCheck EncodeMetalSteps(MetalAdapter &adapter,
                                  MetalKernelResources &resources,
                                  CommandRun &command) {
  rund::AccelCheck result{true, "ok"};
  for (std::size_t index = 0u; result.ok && index < resources.size(); ++index) {
    MetalKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr) {
      result = rund::AccelCheck{false, "accel_kernel_run_invalid"};
      break;
    }
    result = EncodeMetalResets(resources, index, command.encoder);
    if (!result.ok) {
      break;
    }
    if (entry->barrier_before && entry->resets.empty()) {
      [command.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    result = EncodeMetalViewInputs(entry->view, command.encoder);
    if (!result.ok) {
      break;
    }
    result =
        EncodeMetalStep(adapter, entry->ops, entry->resource, command.encoder);
    if (!result.ok) {
      break;
    }
    result = EncodeMetalViewOutputs(entry->view, command.encoder);
  }
  CloseCommand(command);
  return result;
}
#endif

} // namespace rund::node::accel::detail
