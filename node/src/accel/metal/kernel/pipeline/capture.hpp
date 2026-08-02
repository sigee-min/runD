#pragma once

#include "../../../kernel/backend/run.hpp"
#include "../../pipeline/guard.hpp"
#include "../local.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

namespace rund::node::accel::detail {

constexpr NSUInteger kMetalArgumentCapacity = kMetalPipelineArgumentCapacity;

struct MetalCaptureRowCapacity final {
  std::uint64_t rows{};
  bool ok{};
};

// Capture rows are preallocated to the frozen producer-derived upper. Runtime
// rejects a std::vector implementation that reserves a different extent, so
// geometric growth cannot silently reintroduce a cold-path RSS multiplier.
[[nodiscard]] inline MetalCaptureRowCapacity
PlanMetalCaptureRowCapacity(const std::uint64_t required) noexcept {
  if (required > std::numeric_limits<std::size_t>::max() ||
      required > static_cast<std::uint64_t>(
                     std::numeric_limits<std::ptrdiff_t>::max())) {
    return {};
  }
  return MetalCaptureRowCapacity{.rows = required, .ok = true};
}

[[nodiscard]] inline MetalCaptureRowCapacity
PlanMetalParameterCapacity(const std::uint64_t current,
                           const std::uint64_t required,
                           const std::uint64_t limit) noexcept {
  if (current > limit || required > limit ||
      limit > std::numeric_limits<std::size_t>::max()) {
    return {};
  }
  if (required <= current) {
    return MetalCaptureRowCapacity{.rows = current, .ok = true};
  }
  constexpr std::uint64_t InitialCapacity = 256u;
  std::uint64_t planned = std::min(limit, InitialCapacity);
  planned = std::max(planned, current);
  while (planned < required) {
    planned = planned > limit / 2u ? limit : planned * 2u;
    if (planned == 0u) {
      return {};
    }
  }
  return MetalCaptureRowCapacity{.rows = planned, .ok = true};
}

enum class MetalGrid : std::uint8_t {
  None,
  Groups,
  Threads,
};

struct MetalBinding final {
  id<MTLBuffer> buffer = nil;
  NSUInteger offset = 0u;
  std::size_t parameter = 0u;
  bool uses_parameter = false;
  bool bound = false;
};

struct MetalPipelineStatusBindingRecord final {
  MetalPipelineStatusBinding binding{};
  std::uint32_t raw_offset{};
  std::uint32_t raw_count{};
};

struct MetalCommandBinding final {
  id<MTLBuffer> buffer = nil;
  NSUInteger offset = 0u;
  std::size_t parameter = 0u;
  NSUInteger index = 0u;
};

struct MetalCommand final {
  id<MTLComputePipelineState> pipeline = nil;
  std::size_t binding_begin = 0u;
  std::size_t binding_count = 0u;
  MTLSize grid = MTLSizeMake(0u, 0u, 0u);
  MTLSize threads = MTLSizeMake(0u, 0u, 0u);
  MetalGrid kind = MetalGrid::None;
  bool barrier = false;
  bool control = false;
  std::uint32_t owner{std::numeric_limits<std::uint32_t>::max()};
};

struct MetalCapture final {
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> guard_zero = nil;
  id<MTLBuffer> guard_states = nil;
  std::array<MetalBinding, kMetalArgumentCapacity> bindings{};
  std::vector<std::byte> parameters;
  std::vector<MetalCommandBinding> command_bindings;
  std::vector<MetalCommand> commands;
  // Non-owning view into the already frozen status-description table. A
  // copied replacement vector carried no independent meaning and introduced
  // geometric cold-path allocation for every Program occurrence.
  std::span<const MetalPipelineStatusBindingRecord> replacements{};
  id<MTLBuffer> replacement_target = nil;
  std::uint32_t binding_mask = 0u;
  std::size_t command_capacity{};
  std::size_t binding_capacity{};
  std::size_t parameter_capacity{};
  NSUInteger producer_binding_slot_upper{};
  std::uint32_t owner{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t guard_state_count{};
  bool unguarded = false;
  bool capacity_failed = false;
  bool failed = false;
};

// Capture callbacks cannot return an error through Metal's encoder-shaped
// interface.  Consume their sticky result at the narrowest call site so the
// preparation cursor still names the command owner that failed.
[[nodiscard]] inline rund::AccelCheck
CheckMetalPipelineCapture(const MetalCapture &capture) noexcept {
  if (capture.capacity_failed) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (capture.failed) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  return rund::AccelCheck{true, "ok"};
}

static_assert(sizeof(MetalBinding) == 32u);
static_assert(sizeof(MetalCommandBinding) == 32u);
static_assert(sizeof(MetalCommand) == 80u);

struct MetalWork final {
  std::uint64_t workgroup_count{};
  std::uint64_t work_item_count{};
  bool exact{};
};

[[nodiscard]] MetalWork
MeasureMetalWork(std::span<const MetalCommand> commands) noexcept;

} // namespace rund::node::accel::detail

@interface RUNDMetalPipelineCapture : NSObject {
@private
  rund::node::accel::detail::MetalCapture *_capture;
}
- (instancetype)initWithCapture:
    (rund::node::accel::detail::MetalCapture *)capture;
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline;
- (void)setBuffer:(id<MTLBuffer>)buffer
           offset:(NSUInteger)offset
          atIndex:(NSUInteger)index;
- (void)setBytes:(const void *)bytes
          length:(NSUInteger)length
         atIndex:(NSUInteger)index;
- (void)dispatchThreadgroups:(MTLSize)groups
       threadsPerThreadgroup:(MTLSize)threads;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group;
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)threads;
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope;
@end

#endif
