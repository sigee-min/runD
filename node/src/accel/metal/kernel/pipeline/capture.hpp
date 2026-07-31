#pragma once

#include "../local.hpp"
#include "../../pipeline/guard.hpp"
#include "../../../kernel/backend/run.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

namespace rund::node::accel::detail {

constexpr NSUInteger kMetalArgumentCapacity =
    kMetalPipelineArgumentCapacity;

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

struct MetalReplacement final {
  id<MTLBuffer> source = nil;
  id<MTLBuffer> target = nil;
  NSUInteger source_offset = 0u;
  NSUInteger source_bytes = 0u;
  NSUInteger target_offset = 0u;
};

struct MetalCommandBinding final {
  id<MTLBuffer> buffer = nil;
  NSUInteger offset = 0u;
  std::size_t parameter = 0u;
  NSUInteger index = 0u;
};

struct MetalThreadgroupBinding final {
  NSUInteger length = 0u;
  NSUInteger index = 0u;
};

struct MetalCommand final {
  id<MTLComputePipelineState> pipeline = nil;
  std::size_t binding_begin = 0u;
  std::size_t binding_count = 0u;
  std::size_t threadgroup_begin = 0u;
  std::size_t threadgroup_count = 0u;
  MTLSize grid = MTLSizeMake(0u, 0u, 0u);
  MTLSize threads = MTLSizeMake(0u, 0u, 0u);
  MetalGrid kind = MetalGrid::None;
  bool barrier = false;
  bool control = false;
  std::uint32_t owner{std::numeric_limits<std::uint32_t>::max()};
};

// ABI proof for the fixed snapshot representation replaced by the sparse
// command rows below. This type is never instantiated.
struct MetalFixedCommandLayoutProof final {
  id<MTLComputePipelineState> pipeline = nil;
  std::array<MetalBinding, kMetalArgumentCapacity> bindings{};
  std::array<NSUInteger, kMetalArgumentCapacity> threadgroup{};
  MTLSize grid = MTLSizeMake(0u, 0u, 0u);
  MTLSize threads = MTLSizeMake(0u, 0u, 0u);
  MetalGrid kind = MetalGrid::None;
  bool barrier = false;
};

struct MetalCapture final {
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> guard_zero = nil;
  id<MTLBuffer> guard_states = nil;
  std::array<MetalBinding, kMetalArgumentCapacity> bindings{};
  std::array<NSUInteger, kMetalArgumentCapacity> threadgroup{};
  std::vector<std::byte> parameters;
  std::vector<MetalCommandBinding> command_bindings;
  std::vector<MetalThreadgroupBinding> command_threadgroups;
  std::vector<MetalCommand> commands;
  std::vector<MetalReplacement> replacements;
  MTLIndirectCommandType types = 0u;
  NSUInteger highest_binding = 0u;
  std::uint32_t binding_mask = 0u;
  std::uint32_t threadgroup_mask = 0u;
  std::uint32_t owner{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t guard_state_count{};
  bool capacity_failed = false;
  bool failed = false;
};

static_assert(sizeof(MetalBinding) == 32u);
static_assert(sizeof(MetalFixedCommandLayoutProof) == 1304u);
static_assert(sizeof(MetalCommandBinding) == 32u);
static_assert(sizeof(MetalThreadgroupBinding) == 16u);
static_assert(sizeof(MetalCommand) == 96u);

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
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreadgroups:(MTLSize)groups
       threadsPerThreadgroup:(MTLSize)threads;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group;
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)threads;
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope;
@end

#endif
