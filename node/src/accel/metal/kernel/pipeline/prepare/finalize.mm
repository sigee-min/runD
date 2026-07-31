#include "../build.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_set>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

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
      advance_count + canonicalize_count;
  [encoder setComputePipelineState:complete];
  [encoder setBuffer:pipeline->control offset:0u atIndex:0u];
  [encoder setBytes:&status_params length:sizeof(status_params) atIndex:1u];
  if (profile_steps) {
    [encoder setBuffer:pipeline->step_control offset:0u atIndex:2u];
  }
  id<MTLBuffer> const states =
      pipeline->states == nil ? pipeline->control : pipeline->states;
  [encoder setBuffer:states offset:0u atIndex:3u];
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
    // Cold finalization authors every retained indirect command on the CPU.
    // Metal validation therefore requires CPU-accessible ICB storage; Private
    // storage is valid only when command encoding itself is performed by the
    // GPU.
    pipeline->commands = [device
        newIndirectCommandBufferWithDescriptor:descriptor
                               maxCommandCount:captured.commands.size()
                                       options:MTLResourceStorageModeShared];
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
      if (source.kind != MetalGrid::Groups &&
          source.kind != MetalGrid::Threads) {
        return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
      }
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
      id<MTLIndirectComputeCommand> const command =
          [pipeline->commands indirectComputeCommandAtIndex:index];
      if (index != 0u && captured.commands[index - 1u].barrier) {
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
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t recurrence_host_bytes = 0u;
  std::uint64_t recurrence_device_bytes = 0u;
  std::uint64_t recurrence_reused_bytes = 0u;
  const auto account_recurrence = [&](const std::shared_ptr<void> &resource) {
    const auto *const map =
        static_cast<const MetalMapEncodeResources *>(resource.get());
    if (map == nullptr) {
      return;
    }
    std::uint64_t host = sizeof(MetalMapEncodeResources);
    for (const std::uint64_t bytes : {
             static_cast<std::uint64_t>(map->input_plans.capacity()) *
                 sizeof(InputWindowPlan),
             static_cast<std::uint64_t>(map->checks.capacity()) *
                 sizeof(MetalMapCheck),
             static_cast<std::uint64_t>(map->windows.capacity()) *
                 sizeof(rund::kernel::ComputeDispatchWindow),
             static_cast<std::uint64_t>(
                 map->resident.overflow_inputs.capacity()) *
                 sizeof(MetalResidentBufferResult),
             static_cast<std::uint64_t>(
                 map->resident.overflow_outputs.capacity()) *
                 sizeof(MetalResidentBufferResult)}) {
      host = ::rund::detail::counter::SaturatingAdd(host, bytes);
    }
    recurrence_host_bytes = ::rund::detail::counter::SaturatingAdd(
        recurrence_host_bytes, host);
    recurrence_device_bytes = ::rund::detail::counter::SaturatingAdd(
        recurrence_device_bytes, map->param.bytes);
    if (map->param.reused) {
      recurrence_reused_bytes = ::rund::detail::counter::SaturatingAdd(
          recurrence_reused_bytes, map->param.bytes);
    }
  };
  account_recurrence(pipeline->recurrence);
  for (const std::shared_ptr<void> &resource : pipeline->transducers) {
    account_recurrence(resource);
  }
  const std::uint64_t host_bytes =
      sizeof(MetalSequence) +
      pipeline->declared.capacity() * sizeof(id<MTLResource>) +
      pipeline->pipelines.capacity() * sizeof(id<MTLComputePipelineState>) +
      pipeline->telemetry.capacity() * sizeof(MetalPipelineTelemetryRecord) +
      pipeline->step_evidence.capacity() *
          sizeof(PreparedPipelineStepEvidence) +
      pipeline->transducers.capacity() * sizeof(std::shared_ptr<void>) +
      recurrence_host_bytes;
  const std::uint64_t device_bytes =
      static_cast<std::uint64_t>([pipeline->commands allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->parameters allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->raw_status allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->control allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->states allocatedSize]) +
      static_cast<std::uint64_t>([pipeline->guard_zero allocatedSize]) +
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
                                 .reused = recurrence_reused_bytes,
                                 .budget = device_bytes};
  prepared = std::static_pointer_cast<void>(pipeline);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
