#include "../trace.hpp"

#include <rund/counter.hpp>

#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] id<MTLCounterSet>
TimestampCounterSet(id<MTLDevice> const device) noexcept {
  if (device == nil) {
    return nil;
  }
  for (id<MTLCounterSet> const set in [device counterSets]) {
    if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
      return set;
    }
  }
  return nil;
}

} // namespace

bool MetalDispatchTrace::available() const noexcept {
  return samples != nil && values != nil && sample_count != 0u &&
         [values contents] != nullptr &&
         (sampling == MetalTraceSampling::DispatchBoundary ||
          (sampling == MetalTraceSampling::StageBoundary && pass != nil));
}

std::uint64_t MetalDispatchTrace::retained_bytes() const noexcept {
  if (values == nil || samples == nil) {
    return 0u;
  }
  const std::uint64_t sample_bytes = static_cast<std::uint64_t>(sample_count) *
                                     sizeof(MTLCounterResultTimestamp);
  return ::rund::detail::counter::SaturatingAdd(
      sample_bytes, static_cast<std::uint64_t>([values allocatedSize]));
}

void PrepareMetalDispatchTrace(id<MTLDevice> const device,
                               const std::uint64_t dispatch_count,
                               MetalDispatchTrace &trace) noexcept {
  trace = {};
  if (dispatch_count == 0u) {
    return;
  }
  if (dispatch_count > std::numeric_limits<NSUInteger>::max() / 2u) {
    trace.reason = "compute_pipeline_capacity";
    return;
  }
  if (@available(macOS 11.0, iOS 14.0, *)) {
    if (device != nil &&
        [device supportsCounterSampling:
                    MTLCounterSamplingPointAtDispatchBoundary]) {
      trace.sampling = MetalTraceSampling::DispatchBoundary;
    } else if (device != nil &&
               [device supportsCounterSampling:
                           MTLCounterSamplingPointAtStageBoundary]) {
      trace.sampling = MetalTraceSampling::StageBoundary;
    }
  }
  if (trace.sampling == MetalTraceSampling::None) {
    return;
  }
  id<MTLCounterSet> const set = TimestampCounterSet(device);
  if (set == nil) {
    return;
  }
  trace.sample_count = static_cast<NSUInteger>(2u * dispatch_count);
  if (trace.sample_count > std::numeric_limits<NSUInteger>::max() /
                               sizeof(MTLCounterResultTimestamp)) {
    trace.reason = "compute_pipeline_capacity";
    return;
  }
  MTLCounterSampleBufferDescriptor *const descriptor =
      [[MTLCounterSampleBufferDescriptor alloc] init];
  if (descriptor == nil) {
    trace.reason = "compute_device_capacity";
    return;
  }
  descriptor.counterSet = set;
  descriptor.storageMode = MTLStorageModePrivate;
  descriptor.sampleCount = trace.sample_count;
  NSError *error = nil;
  trace.samples = [device newCounterSampleBufferWithDescriptor:descriptor
                                                         error:&error];
#if !__has_feature(objc_arc)
  [descriptor release];
#endif
  if (trace.samples == nil || error != nil) {
    trace = {};
    trace.reason = "compute_device_capacity";
    return;
  }
  const NSUInteger bytes =
      trace.sample_count * sizeof(MTLCounterResultTimestamp);
  trace.values = [device newBufferWithLength:bytes
                                     options:MTLResourceStorageModeShared];
  if (trace.values == nil || [trace.values contents] == nullptr) {
    trace = {};
    trace.reason = "compute_device_capacity";
    return;
  }
  if (trace.sampling == MetalTraceSampling::StageBoundary) {
    trace.pass = [MTLComputePassDescriptor computePassDescriptor];
    MTLComputePassSampleBufferAttachmentDescriptor *const attachment =
        trace.pass == nil ? nil : trace.pass.sampleBufferAttachments[0u];
    if (attachment == nil) {
      trace = {};
      trace.reason = "compute_device_capacity";
      return;
    }
    attachment.sampleBuffer = trace.samples;
  }
  trace.reason = "ok";
}

void BeginMetalDispatchTrace(id<MTLDevice> const device,
                             MetalDispatchTrace &trace) noexcept {
  trace.cursor = 0u;
  trace.cpu_start = 0u;
  trace.gpu_start = 0u;
  trace.failed = !trace.available() || device == nil;
  if (!trace.failed) {
    std::memset([trace.values contents], 0,
                trace.sample_count * sizeof(MTLCounterResultTimestamp));
    [device sampleTimestamps:&trace.cpu_start gpuTimestamp:&trace.gpu_start];
    trace.failed = trace.cpu_start == 0u || trace.gpu_start == 0u;
  }
}

void SampleMetalDispatchTrace(id<MTLComputeCommandEncoder> const encoder,
                              MetalDispatchTrace &trace) noexcept {
  if (encoder == nil || !trace.available() ||
      trace.sampling != MetalTraceSampling::DispatchBoundary ||
      trace.cursor >= trace.sample_count) {
    trace.failed = true;
    return;
  }
  [encoder sampleCountersInBuffer:trace.samples
                    atSampleIndex:trace.cursor
                      withBarrier:YES];
  ++trace.cursor;
}

id<MTLComputeCommandEncoder>
OpenMetalTraceStageEncoder(id<MTLCommandBuffer> const command,
                           MetalDispatchTrace &trace) noexcept {
  if (command == nil || !trace.available() ||
      trace.sampling != MetalTraceSampling::StageBoundary ||
      trace.cursor > trace.sample_count - 2u) {
    trace.failed = true;
    return nil;
  }
  MTLComputePassSampleBufferAttachmentDescriptor *const attachment =
      trace.pass.sampleBufferAttachments[0u];
  if (attachment == nil) {
    trace.failed = true;
    return nil;
  }
  attachment.startOfEncoderSampleIndex = trace.cursor;
  attachment.endOfEncoderSampleIndex = trace.cursor + 1u;
  id<MTLComputeCommandEncoder> const encoder =
      [command computeCommandEncoderWithDescriptor:trace.pass];
  if (encoder == nil) {
    trace.failed = true;
    return nil;
  }
  trace.cursor += 2u;
  return encoder;
}

bool SealMetalDispatchTrace(id<MTLCommandBuffer> const command,
                            MetalDispatchTrace &trace) noexcept {
  if (command == nil || !trace.available() || trace.failed ||
      trace.cursor != trace.sample_count) {
    trace.failed = true;
    return false;
  }
  return true;
}
#endif

} // namespace rund::node::accel::detail
