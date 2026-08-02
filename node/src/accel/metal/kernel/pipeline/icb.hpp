#pragma once

#include <accel/check.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__OBJC__) && defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

struct PreparedKernelPipelineReservation;

// Device calibration and Pipeline materialization share this exact power-of-
// two class family.  A full chunk is deliberately large enough to keep the
// warm path to a handful of execute calls while a short stream still receives
// its smallest fitting native allocation instead of a 64-K-command slab.
inline constexpr std::uint32_t MetalPipelineIcbClassCount = 17u;
inline constexpr std::uint64_t MetalPipelineIcbFullCommandCapacity = 65'536u;
// The SDK header describes this field as a maximum bind index, but native A/B
// execution rejects index 30 when the descriptor is set to 30 and accepts it
// at 31. Preserve the empirically required envelope for Pipeline's 0...30 ABI.
inline constexpr std::uint32_t MetalPipelineIcbBufferBindLimit = 31u;
inline constexpr std::uint64_t MetalPipelineIcbChunkHostBytes = 16u;

static_assert((std::uint64_t{1u} << (MetalPipelineIcbClassCount - 1u)) ==
              MetalPipelineIcbFullCommandCapacity);

struct MetalIcbCalibration final {
  // Row i is the physical allocatedSize reported by this adapter for a
  // 2^i-command ICB made by MakeMetalPipelineIcbDescriptor().
  std::array<std::uint64_t, MetalPipelineIcbClassCount> allocated_bytes{};
};

[[nodiscard]] constexpr bool
ValidMetalIcbCalibration(const MetalIcbCalibration &calibration) noexcept {
  std::uint64_t previous = 0u;
  for (const std::uint64_t bytes : calibration.allocated_bytes) {
    if (bytes == 0u || bytes < previous) {
      return false;
    }
    previous = bytes;
  }
  return true;
}

struct MetalIcbChunkPlan final {
  std::uint64_t command_count{};
  std::uint64_t full_chunk_count{};
  std::uint64_t tail_command_count{};
  std::uint64_t tail_command_capacity{};
  std::uint64_t chunk_count{};
  std::uint64_t allocated_bytes{};
  std::uint64_t retained_chunk_bytes{};
  bool ok{};
};

[[nodiscard]] constexpr std::uint32_t
MetalIcbClassIndex(const std::uint64_t capacity) noexcept {
  if (capacity == 0u || capacity > MetalPipelineIcbFullCommandCapacity ||
      (capacity & (capacity - 1u)) != 0u) {
    return MetalPipelineIcbClassCount;
  }
  std::uint32_t index = 0u;
  std::uint64_t value = capacity;
  while (value > 1u) {
    value >>= 1u;
    ++index;
  }
  return index;
}

[[nodiscard]] constexpr std::uint64_t
MetalIcbTailClassCapacity(const std::uint64_t commands) noexcept {
  if (commands == 0u || commands >= MetalPipelineIcbFullCommandCapacity) {
    return commands == MetalPipelineIcbFullCommandCapacity
               ? MetalPipelineIcbFullCommandCapacity
               : 0u;
  }
  std::uint64_t capacity = 1u;
  while (capacity < commands) {
    capacity <<= 1u;
  }
  return capacity;
}

// The only size-class decomposition authority. Public planning, runtime
// planning, native allocation and contract tests all consume this function.
// The calibrated bytes are measured facts from this exact device rather than
// a coefficient inferred from a different command capacity.
[[nodiscard]] constexpr MetalIcbChunkPlan
PlanMetalIcbChunks(const std::uint64_t command_count,
                   const MetalIcbCalibration &calibration) noexcept {
  if (!ValidMetalIcbCalibration(calibration)) {
    return {};
  }
  if (command_count == 0u) {
    return MetalIcbChunkPlan{.ok = true};
  }
  const std::uint64_t full =
      command_count / MetalPipelineIcbFullCommandCapacity;
  const std::uint64_t tail =
      command_count % MetalPipelineIcbFullCommandCapacity;
  const std::uint64_t tail_capacity = MetalIcbTailClassCapacity(tail);
  const std::uint32_t tail_class = MetalIcbClassIndex(tail_capacity);
  const std::uint64_t full_bytes =
      calibration.allocated_bytes[MetalPipelineIcbClassCount - 1u];
  if (full != 0u &&
      full_bytes > std::numeric_limits<std::uint64_t>::max() / full) {
    return {};
  }
  std::uint64_t bytes = full * full_bytes;
  if (tail != 0u) {
    if (tail_class >= MetalPipelineIcbClassCount ||
        calibration.allocated_bytes[tail_class] >
            std::numeric_limits<std::uint64_t>::max() - bytes) {
      return {};
    }
    bytes += calibration.allocated_bytes[tail_class];
  }
  const std::uint64_t chunks = full + static_cast<std::uint64_t>(tail != 0u);
  if (chunks > std::numeric_limits<std::uint64_t>::max() /
                   MetalPipelineIcbChunkHostBytes) {
    return {};
  }
  return MetalIcbChunkPlan{
      .command_count = command_count,
      .full_chunk_count = full,
      .tail_command_count = tail,
      .tail_command_capacity = tail_capacity,
      .chunk_count = chunks,
      .allocated_bytes = bytes,
      .retained_chunk_bytes = chunks * MetalPipelineIcbChunkHostBytes,
      .ok = true,
  };
}

// Allocation-free Metal structure planner seam used by the backend callback
// and deterministic contracts. The callback supplies the authenticated
// adapter's immutable calibration; tests may supply a measured fixture table.
[[nodiscard]] rund::AccelCheck PlanMetalPipelineStructureForCalibration(
    const MetalIcbCalibration &calibration,
    PreparedKernelPipelineReservation &reservation) noexcept;

#if defined(__OBJC__) && defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// Calibration and materialization must never write descriptor fields along
// separate paths. Keep the complete Pipeline-private descriptor tuple here.
[[nodiscard]] inline MTLIndirectCommandBufferDescriptor *
MakeMetalPipelineIcbDescriptor() {
  MTLIndirectCommandBufferDescriptor *const descriptor =
      [[MTLIndirectCommandBufferDescriptor alloc] init];
  descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch |
                            MTLIndirectCommandTypeConcurrentDispatchThreads;
  descriptor.inheritPipelineState = NO;
  descriptor.inheritBuffers = NO;
  descriptor.maxKernelBufferBindCount = MetalPipelineIcbBufferBindLimit;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    descriptor.maxKernelThreadgroupMemoryBindCount = 0u;
  }
  return descriptor;
}

[[nodiscard]] inline id<MTLIndirectCommandBuffer>
AllocateMetalPipelineIcb(id<MTLDevice> const device,
                         const NSUInteger command_capacity) {
  if (device == nil || command_capacity == 0u) {
    return nil;
  }
  MTLIndirectCommandBufferDescriptor *const descriptor =
      MakeMetalPipelineIcbDescriptor();
  if (descriptor == nil) {
    return nil;
  }
  id<MTLIndirectCommandBuffer> const commands = [device
      newIndirectCommandBufferWithDescriptor:descriptor
                             maxCommandCount:command_capacity
                                     options:MTLResourceStorageModeShared];
#if !__has_feature(objc_arc)
  [descriptor release];
#endif
  return commands;
}

inline constexpr std::uint32_t MetalIcbChunkBarrierBefore = 1u;

// Compact retained warm record. Every native ICB range begins at zero, so a
// count and one boundary flag preserve the complete ordered submission shape
// without retaining a wider NSRange or any cold descriptor metadata.
struct MetalIcbChunk final {
  id<MTLIndirectCommandBuffer> commands = nil;
  std::uint32_t command_count{};
  std::uint32_t flags{};

  [[nodiscard]] bool valid() const noexcept {
    return commands != nil && command_count != 0u &&
           command_count <= MetalPipelineIcbFullCommandCapacity &&
           (flags & ~MetalIcbChunkBarrierBefore) == 0u;
  }

  [[nodiscard]] bool barrier_before() const noexcept {
    return (flags & MetalIcbChunkBarrierBefore) != 0u;
  }
};

static_assert(sizeof(MetalIcbChunk) == MetalPipelineIcbChunkHostBytes);

// The sole warm ICB chunk-loop authority. Production submission and the
// explicit >65,536-command native contract call this exact encoder seam.
[[nodiscard]] inline rund::AccelCheck
EncodeMetalPipelineIcbChunks(id<MTLComputeCommandEncoder> const encoder,
                             const MetalIcbChunk *const chunks,
                             const NSUInteger chunk_count) noexcept {
  if (encoder == nil || chunks == nullptr || chunk_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  for (NSUInteger index = 0u; index < chunk_count; ++index) {
    const MetalIcbChunk &chunk = chunks[index];
    if (!chunk.valid() || (index == 0u && chunk.barrier_before())) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    if (chunk.barrier_before()) {
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder executeCommandsInBuffer:chunk.commands
                           withRange:NSMakeRange(0u, chunk.command_count)];
  }
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
