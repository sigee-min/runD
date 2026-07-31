#include "capture.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace rund::node::accel::detail {

[[nodiscard]] std::size_t align_parameter(const std::size_t value) noexcept {
  constexpr std::size_t alignment = 16u;
  return (value + alignment - 1u) & ~(alignment - 1u);
}

[[nodiscard]] bool empty_grid(const MTLSize size) noexcept {
  return size.width == 0u || size.height == 0u || size.depth == 0u;
}

[[nodiscard]] bool work_volume(const MTLSize size,
                               std::uint64_t &out) noexcept {
  static_assert(sizeof(NSUInteger) <= sizeof(std::uint64_t));
  return rund::kernel::checked::mul(static_cast<std::uint64_t>(size.width),
                                    static_cast<std::uint64_t>(size.height),
                                    static_cast<std::uint64_t>(size.depth),
                                    out);
}

template <class Row>
[[nodiscard]] bool grow_rows(std::vector<Row> &rows,
                             const std::size_t additional) {
  if (additional > rows.max_size() - rows.size()) {
    return false;
  }
  const std::size_t required = rows.size() + additional;
  if (required <= rows.capacity()) {
    return true;
  }
  std::size_t capacity = std::max<std::size_t>(rows.capacity(), 8u);
  while (capacity < required) {
    if (capacity > rows.max_size() / 2u) {
      capacity = required;
      break;
    }
    capacity *= 2u;
  }
  rows.reserve(capacity);
  return true;
}

[[nodiscard]] MetalWork
MeasureMetalWork(const std::span<const MetalCommand> commands) noexcept {
  MetalWork evidence{.exact = true};
  for (const MetalCommand &command : commands) {
    if (command.control) {
      continue;
    }
    if (command.kind == MetalGrid::None ||
        command.kind == MetalGrid::IndirectGroups || empty_grid(command.grid) ||
        empty_grid(command.threads)) {
      return {};
    }
    std::uint64_t command_workgroups = 0u;
    std::uint64_t command_work_items = 0u;
    if (command.kind == MetalGrid::Groups) {
      std::uint64_t threads_per_group = 0u;
      if (!work_volume(command.grid, command_workgroups) ||
          !work_volume(command.threads, threads_per_group) ||
          !rund::kernel::checked::mul(command_workgroups, threads_per_group,
                                      command_work_items)) {
        return {};
      }
    } else if (command.kind == MetalGrid::Threads ||
               command.kind == MetalGrid::DirectThreads) {
      const MTLSize groups =
          MTLSizeMake(static_cast<NSUInteger>(rund::kernel::checked::ceil(
                          static_cast<std::uint64_t>(command.grid.width),
                          static_cast<std::uint64_t>(command.threads.width))),
                      static_cast<NSUInteger>(rund::kernel::checked::ceil(
                          static_cast<std::uint64_t>(command.grid.height),
                          static_cast<std::uint64_t>(command.threads.height))),
                      static_cast<NSUInteger>(rund::kernel::checked::ceil(
                          static_cast<std::uint64_t>(command.grid.depth),
                          static_cast<std::uint64_t>(command.threads.depth))));
      if (!work_volume(groups, command_workgroups) ||
          !work_volume(command.grid, command_work_items)) {
        return {};
      }
    } else {
      return {};
    }
    if (!rund::kernel::checked::add(evidence.workgroup_count,
                                    command_workgroups,
                                    evidence.workgroup_count) ||
        !rund::kernel::checked::add(evidence.work_item_count,
                                    command_work_items,
                                    evidence.work_item_count)) {
      return {};
    }
  }
  return evidence;
}

void append_command(MetalCapture &capture, const MetalGrid kind,
                    const MTLSize grid, const MTLSize threads,
                    id<MTLBuffer> const indirect_buffer = nil,
                    const NSUInteger indirect_offset = 0u) {
  if (capture.pipeline == nil || kind == MetalGrid::None ||
      empty_grid(threads) ||
      (kind == MetalGrid::IndirectGroups && indirect_buffer == nil)) {
    capture.failed = true;
    return;
  }
  if (empty_grid(grid)) {
    return;
  }
  const std::size_t binding_begin = capture.command_bindings.size();
  const std::size_t threadgroup_begin = capture.command_threadgroups.size();
  const std::size_t binding_count = std::popcount(capture.binding_mask);
  const std::size_t threadgroup_count = std::popcount(capture.threadgroup_mask);
  if (binding_count > capture.command_bindings.max_size() - binding_begin ||
      threadgroup_count >
          capture.command_threadgroups.max_size() - threadgroup_begin ||
      capture.commands.size() == capture.commands.max_size()) {
    capture.capacity_failed = true;
    capture.failed = true;
    return;
  }
  try {
    if (!grow_rows(capture.command_bindings, binding_count) ||
        !grow_rows(capture.command_threadgroups, threadgroup_count) ||
        !grow_rows(capture.commands, 1u)) {
      capture.capacity_failed = true;
      capture.failed = true;
      return;
    }
    std::uint32_t bindings = capture.binding_mask;
    while (bindings != 0u) {
      const NSUInteger index =
          static_cast<NSUInteger>(std::countr_zero(bindings));
      const MetalBinding &binding = capture.bindings[index];
      if (!binding.bound) {
        capture.failed = true;
        capture.command_bindings.resize(binding_begin);
        capture.command_threadgroups.resize(threadgroup_begin);
        return;
      }
      capture.command_bindings.push_back(MetalCommandBinding{
          .buffer = binding.uses_parameter ? nil : binding.buffer,
          .offset = binding.offset,
          .parameter = binding.parameter,
          .index = index,
      });
      bindings &= bindings - 1u;
    }
    std::uint32_t threadgroups = capture.threadgroup_mask;
    while (threadgroups != 0u) {
      const NSUInteger index =
          static_cast<NSUInteger>(std::countr_zero(threadgroups));
      capture.command_threadgroups.push_back(MetalThreadgroupBinding{
          .length = capture.threadgroup[index],
          .index = index,
      });
      threadgroups &= threadgroups - 1u;
    }
    MetalCommand command{
        .pipeline = capture.pipeline,
        .indirect_buffer = indirect_buffer,
        .indirect_offset = indirect_offset,
        .stream_index = capture.commands.size(),
        .binding_begin = binding_begin,
        .binding_count = binding_count,
        .threadgroup_begin = threadgroup_begin,
        .threadgroup_count = threadgroup_count,
        .grid = grid,
        .threads = threads,
        .kind = kind,
        .owner = capture.owner,
    };
    capture.commands.push_back(std::move(command));
  } catch (const std::bad_alloc &) {
    capture.command_bindings.resize(binding_begin);
    capture.command_threadgroups.resize(threadgroup_begin);
    capture.capacity_failed = true;
    capture.failed = true;
    return;
  }
  if (kind == MetalGrid::Groups) {
    capture.types |= MTLIndirectCommandTypeConcurrentDispatch;
  } else if (kind == MetalGrid::Threads) {
    capture.types |= MTLIndirectCommandTypeConcurrentDispatchThreads;
  }
}

} // namespace rund::node::accel::detail

using rund::node::accel::detail::align_parameter;
using rund::node::accel::detail::append_command;
using rund::node::accel::detail::kMetalArgumentCapacity;
using rund::node::accel::detail::MetalBinding;
using rund::node::accel::detail::MetalCapture;
using rund::node::accel::detail::MetalGrid;
using rund::node::accel::detail::MetalReplacement;

@implementation RUNDMetalPipelineCapture
- (instancetype)initWithCapture:(MetalCapture *)capture {
  self = [super init];
  if (self != nil) {
    _capture = capture;
  }
  return self;
}
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline {
  _capture->pipeline = pipeline;
}
- (void)setBuffer:(id<MTLBuffer>)buffer
           offset:(NSUInteger)offset
          atIndex:(NSUInteger)index {
  if (index >= kMetalArgumentCapacity) {
    _capture->failed = true;
    return;
  }
  for (const MetalReplacement &replacement : _capture->replacements) {
    if (buffer != replacement.source) {
      continue;
    }
    if (offset < replacement.source_offset ||
        offset - replacement.source_offset >= replacement.source_bytes) {
      continue;
    }
    if (replacement.target == nil ||
        offset - replacement.source_offset >
            std::numeric_limits<NSUInteger>::max() -
                replacement.target_offset) {
      _capture->failed = true;
      return;
    }
    buffer = replacement.target;
    offset = offset - replacement.source_offset + replacement.target_offset;
    break;
  }
  _capture->bindings[index] =
      MetalBinding{.buffer = buffer, .offset = offset, .bound = buffer != nil};
  const std::uint32_t bit = std::uint32_t{1u} << index;
  if (buffer == nil) {
    _capture->binding_mask &= ~bit;
  } else {
    _capture->binding_mask |= bit;
  }
  _capture->highest_binding = std::max(_capture->highest_binding, index + 1u);
}
- (void)setBytes:(const void *)bytes
          length:(NSUInteger)length
         atIndex:(NSUInteger)index {
  if (index >= kMetalArgumentCapacity || bytes == nullptr || length == 0u) {
    _capture->failed = true;
    return;
  }
  if (_capture->parameters.size() >
      std::numeric_limits<std::size_t>::max() - 15u) {
    _capture->capacity_failed = true;
    _capture->failed = true;
    return;
  }
  const std::size_t offset = align_parameter(_capture->parameters.size());
  if (length > std::numeric_limits<std::size_t>::max() - offset ||
      offset > _capture->parameters.max_size() ||
      length > _capture->parameters.max_size() - offset) {
    _capture->capacity_failed = true;
    _capture->failed = true;
    return;
  }
  try {
    _capture->parameters.resize(offset);
    const auto *const begin = static_cast<const std::byte *>(bytes);
    _capture->parameters.insert(_capture->parameters.end(), begin,
                                begin + length);
  } catch (const std::bad_alloc &) {
    _capture->capacity_failed = true;
    _capture->failed = true;
    return;
  }
  _capture->bindings[index] =
      MetalBinding{.parameter = offset, .uses_parameter = true, .bound = true};
  _capture->binding_mask |= std::uint32_t{1u} << index;
  _capture->highest_binding = std::max(_capture->highest_binding, index + 1u);
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length
                           atIndex:(NSUInteger)index {
  if (index >= kMetalArgumentCapacity) {
    _capture->failed = true;
    return;
  }
  _capture->threadgroup[index] = length;
  const std::uint32_t bit = std::uint32_t{1u} << index;
  if (length == 0u) {
    _capture->threadgroup_mask &= ~bit;
  } else {
    _capture->threadgroup_mask |= bit;
  }
}
- (void)dispatchThreadgroups:(MTLSize)groups
       threadsPerThreadgroup:(MTLSize)threads {
  append_command(*_capture, MetalGrid::Groups, groups, threads);
}
- (void)dispatchThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)threads {
  append_command(*_capture, MetalGrid::Threads, grid, threads);
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)threads {
  append_command(*_capture, MetalGrid::IndirectGroups, MTLSizeMake(1u, 1u, 1u),
                 threads, buffer, offset);
}
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
  if (scope == 0u) {
    _capture->failed = true;
    return;
  }
  if (!_capture->commands.empty()) {
    _capture->commands.back().barrier = true;
  }
}
@end

#endif
