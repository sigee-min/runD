#include "capture.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

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
[[nodiscard]] bool
append_within_frozen_capacity(const std::vector<Row> &rows,
                              const std::size_t additional,
                              const std::size_t frozen_capacity) noexcept {
  if (additional > rows.max_size() - rows.size()) {
    return false;
  }
  const std::size_t required = rows.size() + additional;
  return required <= frozen_capacity && required <= rows.capacity();
}

[[nodiscard]] bool grow_parameter_rows(MetalCapture &capture,
                                       const std::size_t required) {
  if (required > capture.parameter_capacity) {
    return false;
  }
  if (required <= capture.parameters.capacity()) {
    return true;
  }
  const MetalCaptureRowCapacity planned = PlanMetalParameterCapacity(
      capture.parameters.capacity(), required, capture.parameter_capacity);
  if (!planned.ok) {
    return false;
  }
  capture.parameters.reserve(static_cast<std::size_t>(planned.rows));
  return capture.parameters.capacity() == planned.rows;
}

[[nodiscard]] MetalWork
MeasureMetalWork(const std::span<const MetalCommand> commands) noexcept {
  MetalWork evidence{.exact = true};
  for (const MetalCommand &command : commands) {
    if (command.control) {
      continue;
    }
    if (command.kind == MetalGrid::None || empty_grid(command.grid) ||
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
    } else if (command.kind == MetalGrid::Threads) {
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
                    const MTLSize grid, const MTLSize threads) {
  const bool owned = capture.owner != std::numeric_limits<std::uint32_t>::max();
  if (capture.pipeline == nil || kind == MetalGrid::None ||
      empty_grid(threads) || (capture.unguarded && owned) ||
      (!capture.unguarded &&
       (capture.guard_zero == nil ||
        (capture.binding_mask &
         (std::uint32_t{1u} << kMetalPipelineGuardBinding)) != 0u ||
        (owned && (capture.guard_states == nil ||
                   capture.owner >= capture.guard_state_count))))) {
    capture.failed = true;
    return;
  }
  if (empty_grid(grid)) {
    return;
  }
  const std::size_t binding_begin = capture.command_bindings.size();
  const std::size_t binding_count =
      std::popcount(capture.binding_mask) +
      static_cast<std::size_t>(!capture.unguarded);
  if (binding_count > capture.command_bindings.max_size() - binding_begin ||
      capture.commands.size() == capture.commands.max_size()) {
    capture.capacity_failed = true;
    capture.failed = true;
    return;
  }
  try {
    if (!append_within_frozen_capacity(capture.command_bindings, binding_count,
                                       capture.binding_capacity) ||
        !append_within_frozen_capacity(capture.commands, 1u,
                                       capture.command_capacity)) {
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
    if (!capture.unguarded) {
      capture.command_bindings.push_back(MetalCommandBinding{
          .buffer = owned ? capture.guard_states : capture.guard_zero,
          .offset = owned ? static_cast<NSUInteger>(capture.owner) *
                                    sizeof(ResidentState) +
                                offsetof(ResidentState, stopped)
                          : 0u,
          .index = kMetalPipelineGuardBinding,
      });
    }
    MetalCommand command{
        .pipeline = capture.pipeline,
        .binding_begin = binding_begin,
        .binding_count = binding_count,
        .grid = grid,
        .threads = threads,
        .kind = kind,
        .owner = capture.owner,
    };
    capture.commands.push_back(std::move(command));
  } catch (const std::bad_alloc &) {
    capture.command_bindings.resize(binding_begin);
    capture.capacity_failed = true;
    capture.failed = true;
    return;
  } catch (const std::length_error &) {
    capture.command_bindings.resize(binding_begin);
    capture.capacity_failed = true;
    capture.failed = true;
    return;
  }
}

} // namespace rund::node::accel::detail

using rund::node::accel::detail::align_parameter;
using rund::node::accel::detail::append_command;
using rund::node::accel::detail::kMetalArgumentCapacity;
using rund::node::accel::detail::MetalBinding;
using rund::node::accel::detail::MetalCapture;
using rund::node::accel::detail::MetalGrid;

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
  if (index >= kMetalArgumentCapacity ||
      index >= _capture->producer_binding_slot_upper) {
    _capture->failed = true;
    return;
  }
  for (const auto &replacement : _capture->replacements) {
    const auto &binding = replacement.binding;
    if (!binding.replace || buffer != (__bridge id<MTLBuffer>)binding.buffer) {
      continue;
    }
    if (offset < binding.offset || offset - binding.offset >= binding.bytes) {
      continue;
    }
    static_assert(sizeof(NSUInteger) >= sizeof(std::uint64_t));
    if (_capture->replacement_target == nil ||
        offset - binding.offset >
            std::numeric_limits<NSUInteger>::max() -
                static_cast<NSUInteger>(replacement.raw_offset) *
                    sizeof(std::uint32_t)) {
      _capture->failed = true;
      return;
    }
    buffer = _capture->replacement_target;
    offset =
        offset - binding.offset +
        static_cast<NSUInteger>(replacement.raw_offset) * sizeof(std::uint32_t);
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
}
- (void)setBytes:(const void *)bytes
          length:(NSUInteger)length
         atIndex:(NSUInteger)index {
  if (index >= kMetalArgumentCapacity ||
      index >= _capture->producer_binding_slot_upper || bytes == nullptr ||
      length == 0u) {
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
      length > _capture->parameters.max_size() - offset ||
      offset > _capture->parameter_capacity ||
      length > _capture->parameter_capacity - offset) {
    _capture->capacity_failed = true;
    _capture->failed = true;
    return;
  }
  try {
    if (!grow_parameter_rows(*_capture, offset + length)) {
      _capture->capacity_failed = true;
      _capture->failed = true;
      return;
    }
    _capture->parameters.resize(offset);
    const auto *const begin = static_cast<const std::byte *>(bytes);
    _capture->parameters.insert(_capture->parameters.end(), begin,
                                begin + length);
  } catch (const std::bad_alloc &) {
    _capture->capacity_failed = true;
    _capture->failed = true;
    return;
  } catch (const std::length_error &) {
    _capture->capacity_failed = true;
    _capture->failed = true;
    return;
  }
  _capture->bindings[index] =
      MetalBinding{.parameter = offset, .uses_parameter = true, .bound = true};
  _capture->binding_mask |= std::uint32_t{1u} << index;
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
  (void)buffer;
  (void)offset;
  (void)threads;
  _capture->failed = true;
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
