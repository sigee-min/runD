#include "internal.hpp"

#include "../../../../pipeline/named.hpp"
#include "../../icb.hpp"

#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
PrepareMetalResidencySlidingGate(MetalSequence &sequence,
                                 MetalResidencySlidingGate &gate) noexcept {
#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    id<MTLDevice> const device =
        sequence.adapter == nullptr
            ? nil
            : (__bridge id<MTLDevice>)sequence.adapter->device.get();
    if (device == nil || !device.hasUnifiedMemory || sequence.control == nil ||
        sequence.guard_zero == nil ||
        sequence.control.storageMode != MTLStorageModeShared ||
        sequence.guard_zero.storageMode != MTLStorageModeShared ||
        sequence.control.gpuAddress == 0u ||
        sequence.guard_zero.gpuAddress == 0u ||
        sequence.control_generation_stride != 1u ||
        sequence.residency_steps.empty() || sequence.direct_aggregate ||
        sequence.state_count != 0u) {
      return {false, "accel_metal_command_unavailable"};
    }
    id<MTLBuffer> const descriptor =
        [device newBufferWithLength:sizeof(MetalResidencySlidingPayload)
                            options:MTLResourceStorageModeShared];
    NSString *const source = [[NSString alloc]
        initWithUTF8String:metal_residency_sliding::GateSource()];
    NSError *error = nil;
    id<MTLLibrary> const library = source == nil
                                       ? nil
                                       : [device newLibraryWithSource:source
                                                              options:nil
                                                                error:&error];
    id<MTLFunction> const function =
        [library newFunctionWithName:@"rund_metal_residency_sliding_gate"];
    id<MTLComputePipelineState> const pipeline =
        NewMetalPipeline(device, function, &error);
    id<MTLIndirectCommandBuffer> const command =
        AllocateMetalPipelineIcb(device, 1u);
    id<MTLIndirectComputeCommand> const gate_dispatch =
        [command indirectComputeCommandAtIndex:0u];
    id<MTLSharedEvent> const ready = [device newSharedEvent];
    if (descriptor == nil || descriptor.storageMode != MTLStorageModeShared ||
        descriptor.length < sizeof(MetalResidencySlidingPayload) ||
        descriptor.gpuAddress == 0u || [descriptor contents] == nullptr ||
        pipeline == nil || pipeline.maxTotalThreadsPerThreadgroup == 0u ||
        command == nil || gate_dispatch == nil || command.allocatedSize == 0u ||
        ready == nil) {
      return {false, "accel_metal_command_unavailable"};
    }
    [gate_dispatch setComputePipelineState:pipeline];
    [gate_dispatch setKernelBuffer:descriptor offset:0u atIndex:0u];
    [gate_dispatch setKernelBuffer:sequence.guard_zero offset:0u atIndex:1u];
    [gate_dispatch setKernelBuffer:sequence.control offset:0u atIndex:2u];
    [gate_dispatch concurrentDispatchThreads:MTLSizeMake(1u, 1u, 1u)
                       threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    const std::uint64_t descriptor_bytes =
        static_cast<std::uint64_t>(descriptor.allocatedSize);
    const std::uint64_t pipeline_bytes =
        static_cast<std::uint64_t>(pipeline.allocatedSize);
    const std::uint64_t command_bytes =
        static_cast<std::uint64_t>(command.allocatedSize);
    if (descriptor_bytes == 0u || pipeline_bytes == 0u || command_bytes == 0u ||
        descriptor_bytes >
            std::numeric_limits<std::uint64_t>::max() - pipeline_bytes ||
        descriptor_bytes + pipeline_bytes >
            std::numeric_limits<std::uint64_t>::max() - command_bytes) {
      return {false, "compute_pipeline_capacity"};
    }
    std::memset([descriptor contents], 0, sizeof(MetalResidencySlidingPayload));
    gate.descriptor = descriptor;
    gate.pipeline = pipeline;
    gate.command = command;
    gate.ready = ready;
    gate.retained_bytes = descriptor_bytes + pipeline_bytes + command_bytes;
    gate.ready_for_submit = true;
    return {true, "ok"};
  }
#else
  static_cast<void>(sequence);
  static_cast<void>(gate);
#endif
  return {false, "accel_metal_command_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
