#include "../build.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_set>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

constexpr NSUInteger kMetalDispatchBytes = 3u * sizeof(std::uint32_t);

rund::AccelCheck MetalPipelineBuild::Finalize(std::shared_ptr<void> &prepared,
                                              PreparedPipelineMemory &memory) {
  if (finished) {
    return rund::AccelCheck{true, "ok"};
  }
  captured.replacements.clear();
  if (captured.commands.size() == reset_command_count) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  // The final captured producer also needs an ICB barrier: command-buffer
  // completion alone does not publish a concurrent indirect dispatch's
  // writes to later readback command buffers.
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  pipeline->control_command_count =
      2u + import_count + static_cast<std::uint32_t>(needs_reset) +
      static_cast<std::uint32_t>(pipeline->telemetry.size()) + fold_count +
      seal_count;
  [encoder setComputePipelineState:complete];
  [encoder setBuffer:pipeline->control offset:0u atIndex:0u];
  [encoder setBytes:&status_params length:sizeof(status_params) atIndex:1u];
  if (profile_steps) {
    [encoder setBuffer:pipeline->step_control offset:0u atIndex:2u];
  }
  [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
  if (!native_publications.empty()) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    for (const MetalPublish &publication : native_publications) {
      if (publication.params.count == 0u) {
        continue;
      }
      id<MTLBuffer> const target =
          (__bridge id<MTLBuffer>)publication.target.get();
      std::array<id<MTLBuffer>, 3u> sources{
          (__bridge id<MTLBuffer>)publication.sources[0].get(),
          (__bridge id<MTLBuffer>)publication.sources[1].get(),
          (__bridge id<MTLBuffer>)publication.sources[2].get()};
      if (target == nil || pipeline->states == nil ||
          std::any_of(sources.begin(), sources.end(),
                      [](const id<MTLBuffer> value) { return value == nil; })) {
        return rund::AccelCheck{false, "accel_metal_buffer_failed"};
      }
      [encoder setComputePipelineState:publish];
      [encoder setBuffer:sources[0] offset:0u atIndex:0u];
      [encoder setBuffer:sources[1] offset:0u atIndex:1u];
      [encoder setBuffer:sources[2] offset:0u atIndex:2u];
      [encoder setBuffer:target offset:0u atIndex:3u];
      [encoder setBuffer:pipeline->control offset:0u atIndex:4u];
      [encoder setBuffer:pipeline->states offset:0u atIndex:5u];
      [encoder setBytes:&publication.params
                 length:sizeof(publication.params)
                atIndex:6u];
      const NSUInteger count =
          static_cast<NSUInteger>(publication.params.count);
      const NSUInteger width =
          std::min(count, [publish maxTotalThreadsPerThreadgroup]);
      [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
      captured.commands.back().control = true;
      ++pipeline->dispatch_count;
      ++pipeline->control_command_count;
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  if (captured.capacity_failed) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (captured.failed || captured.commands.empty()) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  if (!captured.parameters.empty()) {
    pipeline->parameters =
        [device newBufferWithLength:captured.parameters.size()
                            options:MTLResourceStorageModeShared];
    if (pipeline->parameters == nil ||
        [pipeline->parameters contents] == nullptr) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
    std::memcpy([pipeline->parameters contents], captured.parameters.data(),
                captured.parameters.size());
  }
  if (captured.types != 0u) {
    MTLIndirectCommandBufferDescriptor *const descriptor =
        [[MTLIndirectCommandBufferDescriptor alloc] init];
    descriptor.commandTypes = captured.types;
    descriptor.inheritPipelineState = NO;
    descriptor.inheritBuffers = NO;
    descriptor.maxKernelBufferBindCount = captured.highest_binding;
    pipeline->commands = [device
        newIndirectCommandBufferWithDescriptor:descriptor
                               maxCommandCount:captured.commands.size()
                                       options:MTLResourceStorageModePrivate];
    if (pipeline->commands == nil) {
      return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
    }
  }
  pipeline->command_count = captured.commands.size();
  try {
    std::unordered_set<const void *> pipeline_keys;
    std::unordered_set<const void *> resource_keys;
    const auto retain_pipeline = [&](id<MTLComputePipelineState> const value) {
      if (value != nil &&
          pipeline_keys.insert((__bridge const void *)value).second) {
        pipeline->pipelines.push_back(value);
      }
    };
    const auto retain_resource = [&](id<MTLResource> const value) {
      if (value != nil &&
          resource_keys.insert((__bridge const void *)value).second) {
        pipeline->declared.push_back(value);
      }
    };
    for (NSUInteger index = 0u; index < captured.commands.size(); ++index) {
      const MetalCommand &source = captured.commands[index];
      retain_pipeline(source.pipeline);
      const std::size_t binding_end =
          source.binding_begin + source.binding_count;
      if (binding_end < source.binding_begin ||
          binding_end > captured.command_bindings.size()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      for (std::size_t binding = source.binding_begin; binding < binding_end;
           ++binding) {
        const MetalCommandBinding &argument =
            captured.command_bindings[binding];
        const bool uses_parameter = argument.buffer == nil;
        id<MTLBuffer> const buffer =
            uses_parameter ? pipeline->parameters : argument.buffer;
        retain_resource(buffer);
      }
      if (source.kind == MetalGrid::IndirectGroups ||
          source.kind == MetalGrid::DirectThreads) {
        id<MTLResource> const indirect_resource = source.indirect_buffer;
        retain_resource(indirect_resource);
        MetalCommand direct = source;
        if (source.kind == MetalGrid::IndirectGroups &&
            source.owner != std::numeric_limits<std::uint32_t>::max()) {
          if (pipeline->gate_count ==
              std::numeric_limits<std::uint32_t>::max()) {
            return rund::AccelCheck{false, "compute_pipeline_capacity"};
          }
          direct.gate_offset = static_cast<NSUInteger>(pipeline->gate_count) *
                               kMetalDispatchBytes;
          ++pipeline->gate_count;
        }
        pipeline->direct.push_back(std::move(direct));
        continue;
      }
      const bool extend =
          !pipeline->ranges.empty() &&
          pipeline->ranges.back().owner == source.owner &&
          static_cast<std::uint64_t>(pipeline->ranges.back().range.location) +
                  pipeline->ranges.back().range.length ==
              index;
      if (extend) {
        if (pipeline->ranges.back().range.length ==
            std::numeric_limits<std::uint32_t>::max()) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        ++pipeline->ranges.back().range.length;
        pipeline->ranges.back().barrier = source.barrier;
      } else {
        if (index > std::numeric_limits<std::uint32_t>::max()) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        pipeline->ranges.push_back(MetalRangePlan{
            .range = MetalRange{.location = static_cast<std::uint32_t>(index),
                                .length = 1u},
            .owner = source.owner,
            .barrier = source.barrier,
        });
      }
      id<MTLIndirectComputeCommand> const command =
          [pipeline->commands indirectComputeCommandAtIndex:index];
      if (extend && captured.commands[index - 1u].barrier) {
        [command setBarrier];
      }
      [command setComputePipelineState:source.pipeline];
      for (std::size_t binding = source.binding_begin; binding < binding_end;
           ++binding) {
        const MetalCommandBinding &argument =
            captured.command_bindings[binding];
        const bool uses_parameter = argument.buffer == nil;
        id<MTLBuffer> const buffer =
            uses_parameter ? pipeline->parameters : argument.buffer;
        const NSUInteger offset =
            uses_parameter ? argument.parameter : argument.offset;
        [command setKernelBuffer:buffer offset:offset atIndex:argument.index];
      }
      const std::size_t threadgroup_end =
          source.threadgroup_begin + source.threadgroup_count;
      if (threadgroup_end < source.threadgroup_begin ||
          threadgroup_end > captured.command_threadgroups.size()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      for (std::size_t binding = source.threadgroup_begin;
           binding < threadgroup_end; ++binding) {
        const MetalThreadgroupBinding &argument =
            captured.command_threadgroups[binding];
        [command setThreadgroupMemoryLength:argument.length
                                    atIndex:argument.index];
      }
      if (source.kind == MetalGrid::Groups) {
        [command concurrentDispatchThreadgroups:source.grid
                          threadsPerThreadgroup:source.threads];
      } else {
        [command concurrentDispatchThreads:source.grid
                     threadsPerThreadgroup:source.threads];
      }
    }
    const std::size_t range_count = pipeline->ranges.size();
    const std::size_t range_storage = std::max(
        range_count, window_params.empty() ? std::size_t{0u} : std::size_t{1u});
    if (range_storage > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    id<MTLBuffer> const old_ranges = pipeline->range_buffer;
    id<MTLBuffer> const old_owners = pipeline->range_owners;
    if (range_storage != pipeline->range_capacity) {
      id<MTLBuffer> ranges = nil;
      id<MTLBuffer> owners = nil;
      if (range_storage != 0u) {
        ranges =
            [device newBufferWithLength:static_cast<NSUInteger>(range_storage) *
                                        sizeof(MetalRange)
                                options:MTLResourceStorageModeShared];
        owners =
            [device newBufferWithLength:static_cast<NSUInteger>(range_storage) *
                                        sizeof(std::uint32_t)
                                options:MTLResourceStorageModeShared];
        if (ranges == nil || [ranges contents] == nullptr || owners == nil ||
            [owners contents] == nullptr) {
          return rund::AccelCheck{false, "accel_metal_buffer_failed"};
        }
      }
      for (MetalCommandBinding &binding : captured.command_bindings) {
        if (old_ranges != nil && binding.buffer == old_ranges) {
          binding.buffer = ranges;
        } else if (old_owners != nil && binding.buffer == old_owners) {
          binding.buffer = owners;
        }
      }
      for (std::size_t index = 0u; index < pipeline->declared.size(); ++index) {
        if (old_ranges != nil && pipeline->declared[index] == old_ranges) {
          pipeline->declared[index] = ranges;
        } else if (old_owners != nil &&
                   pipeline->declared[index] == old_owners) {
          pipeline->declared[index] = owners;
        }
      }
      pipeline->range_buffer = ranges;
      pipeline->range_owners = owners;
      pipeline->range_capacity = static_cast<std::uint32_t>(range_storage);
    }
    auto *const range_values =
        static_cast<MetalRange *>([pipeline->range_buffer contents]);
    auto *const range_owners =
        static_cast<std::uint32_t *>([pipeline->range_owners contents]);
    if (range_storage != 0u &&
        (range_values == nullptr || range_owners == nullptr)) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
    if (range_storage != 0u) {
      std::memset(range_values, 0, range_storage * sizeof(MetalRange));
      std::fill_n(range_owners, range_storage,
                  std::numeric_limits<std::uint32_t>::max());
    }
    if (!window_params.empty()) {
      auto *const parameters =
          static_cast<std::byte *>([pipeline->parameters contents]);
      if (parameters == nullptr) {
        return rund::AccelCheck{false, "accel_metal_buffer_failed"};
      }
      for (const std::size_t offset : window_params) {
        if (offset > captured.parameters.size() ||
            sizeof(MetalWindowParams) > captured.parameters.size() - offset) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        MetalWindowParams params{};
        std::memcpy(&params, parameters + offset, sizeof(params));
        params.range_count = static_cast<std::uint32_t>(range_count);
        std::memcpy(parameters + offset, &params, sizeof(params));
      }
    }
    pipeline->original_ranges.reserve(pipeline->ranges.size());
    for (std::size_t index = 0u; index < pipeline->ranges.size(); ++index) {
      const MetalRangePlan &range = pipeline->ranges[index];
      pipeline->original_ranges.push_back(range.range);
      range_values[index] = range.range;
      range_owners[index] = range.owner;
    }
    if (!pipeline->direct.empty()) {
      pipeline->indirect_bindings = captured.command_bindings;
      pipeline->indirect_threadgroups = captured.command_threadgroups;
    }
    if (pipeline->gate_count != 0u) {
      const std::uint64_t gate_bytes =
          static_cast<std::uint64_t>(pipeline->gate_count) *
          kMetalDispatchBytes;
      if (gate_bytes > std::numeric_limits<NSUInteger>::max() ||
          pipeline->gate == nil) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      pipeline->gate_buffer =
          [device newBufferWithLength:static_cast<NSUInteger>(gate_bytes)
                              options:MTLResourceStorageModePrivate];
      if (pipeline->gate_buffer == nil) {
        return rund::AccelCheck{false, "accel_metal_buffer_failed"};
      }
      retain_resource(pipeline->gate_buffer);
      retain_pipeline(pipeline->gate);
      if (pipeline->gate_count > std::numeric_limits<std::uint64_t>::max() -
                                     pipeline->dispatch_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      pipeline->dispatch_count += pipeline->gate_count;
      if (pipeline->gate_count > std::numeric_limits<std::uint32_t>::max() -
                                     pipeline->control_command_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      pipeline->control_command_count += pipeline->gate_count;
    }
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const auto *const recurrence_map =
      static_cast<const MetalMapEncodeResources *>(pipeline->recurrence.get());
  const std::uint64_t recurrence_host_bytes =
      recurrence_map == nullptr
          ? 0u
          : sizeof(MetalMapEncodeResources) +
                recurrence_map->windows.capacity() *
                    sizeof(rund::kernel::ComputeDispatchWindow) +
                recurrence_map->resident.overflow_inputs.capacity() *
                    sizeof(MetalResidentBufferResult) +
                recurrence_map->resident.overflow_outputs.capacity() *
                    sizeof(MetalResidentBufferResult);
  const std::uint64_t recurrence_device_bytes =
      recurrence_map == nullptr ? 0u : recurrence_map->param.bytes;
  const std::uint64_t host_bytes =
      sizeof(MetalSequence) +
      pipeline->declared.capacity() * sizeof(id<MTLResource>) +
      pipeline->pipelines.capacity() * sizeof(id<MTLComputePipelineState>) +
      pipeline->direct.capacity() * sizeof(MetalCommand) +
      pipeline->ranges.capacity() * sizeof(MetalRangePlan) +
      pipeline->original_ranges.capacity() * sizeof(MetalRange) +
      pipeline->indirect_bindings.capacity() * sizeof(MetalCommandBinding) +
      pipeline->indirect_threadgroups.capacity() *
          sizeof(MetalThreadgroupBinding) +
      pipeline->telemetry.capacity() * sizeof(MetalPipelineTelemetryRecord) +
      pipeline->step_evidence.capacity() *
          sizeof(PreparedPipelineStepEvidence) +
      recurrence_host_bytes;
  const std::uint64_t device_bytes =
      static_cast<std::uint64_t>([pipeline->commands allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->parameters allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->raw_status allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->control allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->states allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->gate_buffer allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->range_buffer allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->range_owners allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->step_control allocatedSize]) +
      recurrence_device_bytes;
  if (profile_steps) {
    pipeline->instrumentation_byte_count =
        static_cast<std::uint64_t>([pipeline->step_control allocatedSize]);
  }
  pipeline->retained_bytes = host_bytes + device_bytes;
  memory.host = PreparedMemory{.current = host_bytes,
                               .peak = host_bytes,
                               .cumulative = host_bytes,
                               .budget = host_bytes};
  memory.device = PreparedMemory{.current = device_bytes,
                                 .peak = device_bytes,
                                 .cumulative = device_bytes,
                                 .reused = recurrence_map != nullptr &&
                                                   recurrence_map->param.reused
                                               ? recurrence_device_bytes
                                               : 0u,
                                 .budget = device_bytes};
  prepared = std::static_pointer_cast<void>(pipeline);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
