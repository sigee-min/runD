#include "encode.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

[[nodiscard]] rund::AccelCheck
EncodeMetalResidencyIcb(id<MTL4ComputeCommandEncoder> const encoder,
                        const MetalWarmSubmission warm) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  if (encoder == nil || warm.chunks == nullptr || warm.chunk_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  for (NSUInteger index = 0u; index < warm.chunk_count; ++index) {
    const MetalIcbChunk &chunk = warm.chunks[index];
    if (!chunk.valid() || (index == 0u && chunk.barrier_before())) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    if (chunk.barrier_before()) {
      [encoder barrierAfterEncoderStages:MTLStageDispatch
                     beforeEncoderStages:MTLStageDispatch
                       visibilityOptions:MTL4VisibilityOptionDevice];
    }
    [encoder executeCommandsInBuffer:chunk.commands
                           withRange:NSMakeRange(0u, chunk.command_count)];
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck
EncodeMetalResidencyRange(id<MTL4ComputeCommandEncoder> const encoder,
                          const MetalWarmSubmission warm,
                          const MetalResidencyStepRange range) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  if (range.count == 0u) {
    return rund::AccelCheck{true, "ok"};
  }
  if (encoder == nil || warm.chunks == nullptr || warm.chunk_count == 0u ||
      range.begin > std::numeric_limits<std::uint64_t>::max() - range.count) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  const std::uint64_t range_end = range.begin + range.count;
  std::uint64_t global = 0u;
  std::uint64_t consumed = 0u;
  for (NSUInteger index = 0u; index < warm.chunk_count; ++index) {
    const MetalIcbChunk &chunk = warm.chunks[index];
    if (!chunk.valid() || global > std::numeric_limits<std::uint64_t>::max() -
                                       chunk.command_count) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    const std::uint64_t chunk_end = global + chunk.command_count;
    const std::uint64_t first = std::max(global, range.begin);
    const std::uint64_t last = std::min(chunk_end, range_end);
    if (last > first) {
      const std::uint64_t local = first - global;
      const std::uint64_t count = last - first;
      if ((consumed != 0u || (local == 0u && chunk.barrier_before()))) {
        [encoder barrierAfterEncoderStages:MTLStageDispatch
                       beforeEncoderStages:MTLStageDispatch
                         visibilityOptions:MTL4VisibilityOptionDevice];
      }
      [encoder
          executeCommandsInBuffer:chunk.commands
                        withRange:NSMakeRange(static_cast<NSUInteger>(local),
                                              static_cast<NSUInteger>(count))];
      consumed += count;
    }
    global = chunk_end;
    if (global >= range_end) {
      break;
    }
  }
  return consumed == range.count
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
}

#endif

} // namespace

#if defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

[[nodiscard]] rund::AccelCheck EncodeMetalResidencySelection(
    id<MTL4ComputeCommandEncoder> const encoder, MetalSequence &sequence,
    const std::span<const std::uint32_t> locals, std::uint64_t &dispatch_count,
    std::uint64_t &control_count, std::uint64_t &reset_count,
    std::uint64_t &reset_bytes) noexcept API_AVAILABLE(macos(26.0), ios(26.0)) {
  dispatch_count = 0u;
  control_count = 0u;
  reset_count = 0u;
  reset_bytes = 0u;
  if (locals.empty()) {
    dispatch_count = sequence.dispatch_count;
    control_count = sequence.control_command_count;
    reset_count = sequence.reset_count;
    reset_bytes = sequence.reset_bytes;
    return EncodeMetalResidencyIcb(encoder, sequence.warm);
  }
  if (!sequence.residency_selectable ||
      locals.size() > sequence.residency_steps.size()) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  if (locals.size() == sequence.residency_steps.size()) {
    for (std::size_t index = 0u; index < locals.size(); ++index) {
      if (locals[index] >= sequence.residency_steps.size()) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
      for (std::size_t prior = 0u; prior < index; ++prior) {
        if (locals[prior] == locals[index]) {
          return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
        }
      }
    }
    dispatch_count = sequence.dispatch_count;
    control_count = sequence.control_command_count;
    reset_count = sequence.reset_count;
    reset_bytes = sequence.reset_bytes;
    return EncodeMetalResidencyIcb(encoder, sequence.warm);
  }
  rund::AccelCheck encoded = EncodeMetalResidencyRange(
      encoder, sequence.warm, sequence.residency_prefix);
  control_count = sequence.residency_prefix.control_count;
  for (std::size_t index = 0u; encoded.ok && index < locals.size(); ++index) {
    const std::uint32_t local = locals[index];
    if (local >= sequence.residency_steps.size()) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (locals[prior] == local) {
        return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
      }
    }
    if (sequence.residency_prefix.count != 0u || index != 0u) {
      [encoder barrierAfterEncoderStages:MTLStageDispatch
                     beforeEncoderStages:MTLStageDispatch
                       visibilityOptions:MTL4VisibilityOptionDevice];
    }
    const MetalResidencyStepRange range = sequence.residency_steps[local];
    encoded = EncodeMetalResidencyRange(encoder, sequence.warm, range);
    if (dispatch_count >
        std::numeric_limits<std::uint64_t>::max() - range.dispatch_count) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    dispatch_count += range.dispatch_count;
    control_count = ::rund::detail::counter::SaturatingAdd(control_count,
                                                           range.control_count);
    reset_count =
        ::rund::detail::counter::SaturatingAdd(reset_count, range.reset_count);
    reset_bytes =
        ::rund::detail::counter::SaturatingAdd(reset_bytes, range.reset_bytes);
  }
  if (encoded.ok && sequence.residency_suffix.count != 0u) {
    [encoder barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
    encoded = EncodeMetalResidencyRange(encoder, sequence.warm,
                                        sequence.residency_suffix);
    control_count = ::rund::detail::counter::SaturatingAdd(
        control_count, sequence.residency_suffix.control_count);
  }
  return encoded;
}

[[nodiscard]] id<MTL4CommandQueue>
MetalResidencyQueue(const MetalSequence &sequence) noexcept
    API_AVAILABLE(macos(26.0), ios(26.0)) {
  return sequence.adapter == nullptr ||
                 sequence.adapter->residency_queue == nullptr
             ? nil
             : (__bridge id<MTL4CommandQueue>)
                   sequence.adapter->residency_queue.get();
}

rund::AccelCheck EncodeMetalResidencySlidingCommand(
    MetalSequence &sequence, const std::span<const std::uint32_t> locals,
    std::uint64_t &dispatch_count, std::uint64_t &control_count,
    std::uint64_t &reset_count, std::uint64_t &reset_bytes) noexcept {
  MetalResidencySubmission &submission = sequence.residency_submission;
  MetalResidencySlidingGate &gate = sequence.residency_sliding;
  dispatch_count = 0u;
  control_count = 0u;
  reset_count = 0u;
  reset_bytes = 0u;
  if (!submission.ready() || submission.allocator == nil ||
      submission.command == nil || submission.resources == nil ||
      gate.pipeline == nil || gate.command == nil || locals.empty()) {
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  [submission.command beginCommandBufferWithAllocator:submission.allocator];
  [submission.command useResidencySet:submission.resources];
  id<MTL4ComputeCommandEncoder> const encoder =
      [submission.command computeCommandEncoder];
  if (encoder == nil) {
    [submission.command endCommandBuffer];
    [submission.allocator reset];
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  [encoder executeCommandsInBuffer:gate.command
                          withRange:NSMakeRange(0u, 1u)];
  [encoder barrierAfterEncoderStages:MTLStageDispatch
                 beforeEncoderStages:MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionDevice];
  const rund::AccelCheck encoded = EncodeMetalResidencySelection(
      encoder, sequence, locals, dispatch_count, control_count, reset_count,
      reset_bytes);
  [encoder endEncoding];
  [submission.command endCommandBuffer];
  if (!encoded.ok) {
    [submission.allocator reset];
  }
  return encoded;
}

id<MTL4CommandQueue>
MetalResidencyScheduleQueue(const MetalSequence &sequence) noexcept {
  return MetalResidencyQueue(sequence);
}

rund::AccelCheck EncodeMetalResidencyScheduleCommand(
    MetalSequence &sequence, id<MTL4CommandAllocator> const allocator,
    id<MTL4CommandBuffer> const command,
    const std::span<const std::uint32_t> locals, std::uint64_t &dispatch_count,
    std::uint64_t &control_count, std::uint64_t &reset_count,
    std::uint64_t &reset_bytes) noexcept {
  if (allocator == nil || command == nil ||
      !sequence.residency_submission.ready() || locals.empty()) {
    return rund::AccelCheck{false, "accel_metal_command_unavailable"};
  }
  [command beginCommandBufferWithAllocator:allocator];
  [command useResidencySet:sequence.residency_submission.resources];
  id<MTL4ComputeCommandEncoder> const encoder = [command computeCommandEncoder];
  const rund::AccelCheck encoded =
      EncodeMetalResidencySelection(encoder, sequence, locals, dispatch_count,
                                    control_count, reset_count, reset_bytes);
  if (encoder != nil) {
    [encoder endEncoding];
  }
  [command endCommandBuffer];
  if (!encoded.ok) {
    [allocator reset];
  }
  return encoded;
}

#endif

#endif

} // namespace rund::node::accel::detail
