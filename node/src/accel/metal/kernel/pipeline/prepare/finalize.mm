#include "../build.hpp"
#include "../identity_index.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Finalize(std::shared_ptr<void> &prepared,
                                              PreparedPipelineMemory &memory) {
  if (finished) {
    return rund::AccelCheck{true, "ok"};
  }
  captured.replacements = {};
  captured.replacement_target = nil;
  if (captured.commands.size() == reset_command_count) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  if (aggregate_selected) {
    if (captured.commands.size() != 2u) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    pipeline->control_command_count = 1u;
  } else {
    // The final captured producer also needs an ICB barrier: command-buffer
    // completion alone does not publish a concurrent indirect dispatch's
    // writes to later readback command buffers.
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    pipeline->control_command_count =
        2u + import_count + static_cast<std::uint32_t>(needs_reset) +
        static_cast<std::uint32_t>(pipeline->telemetry.size()) + fold_count +
        advance_count + canonicalize_count + window_publish_count;
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
    const rund::AccelCheck completed = CheckMetalPipelineCapture(captured);
    if (!completed.ok) {
      return completed;
    }
    if (native_publication_count != 0u) {
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      for (const MetalPublish &publication : native_publication_rows()) {
        if (publication.params.kind !=
                static_cast<std::uint32_t>(
                    PreparedKernelPublicationKind::Terminal) ||
            publication.params.count == 0u) {
          continue;
        }
        id<MTLBuffer> const target =
            (__bridge id<MTLBuffer>)publication.target.get();
        std::array<id<MTLBuffer>, 3u> sources{
            (__bridge id<MTLBuffer>)publication.sources[0].get(),
            (__bridge id<MTLBuffer>)publication.sources[1].get(),
            (__bridge id<MTLBuffer>)publication.sources[2].get()};
        if (target == nil || pipeline->states == nil ||
            std::any_of(
                sources.begin(), sources.end(),
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
        [encoder setBuffer:pipeline->control offset:0u atIndex:7u];
        const NSUInteger count =
            static_cast<NSUInteger>(publication.params.count);
        const NSUInteger width =
            std::min(count, [publish maxTotalThreadsPerThreadgroup]);
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        const rund::AccelCheck capture = CheckMetalPipelineCapture(captured);
        if (!capture.ok) {
          return capture;
        }
        captured.commands.back().control = true;
        ++pipeline->dispatch_count;
        ++pipeline->control_command_count;
      }
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }
  const rund::AccelCheck capture = CheckMetalPipelineCapture(captured);
  if (!capture.ok) {
    return capture;
  }
  if (captured.commands.empty()) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  const PreparedKernelPipelineReservation &limit = template_registry.limit;
  const MetalIcbChunkPlan actual_icb_plan = PlanMetalIcbChunks(
      captured.commands.size(), pipeline->adapter->pipeline_icb_calibration);
  if (!limit.ok || captured.commands.size() > limit.backend_command_count ||
      captured.command_bindings.size() > limit.backend_command_binding_count ||
      captured.parameters.size() > limit.backend_parameter_bytes ||
      captured.parameters.capacity() > limit.backend_parameter_bytes ||
      !actual_icb_plan.ok || actual_icb_plan.chunk_count == 0u ||
      actual_icb_plan.chunk_count > limit.backend_command_chunk_count ||
      actual_icb_plan.allocated_bytes >
          limit.backend_command_native_bytes) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Freeze first-command order with one membership-only pointer index. Its
  // 2x power-of-two storage is reused for pipeline states and resources, then
  // released before native allocation; no hash owner reaches the warm path.
  std::uint64_t identity_index_bytes = 0u;
  bool uses_parameters = false;
  std::size_t parameter_residency_index =
      std::numeric_limits<std::size_t>::max();
  {
    const std::uint64_t identity_capacity = std::max<std::uint64_t>(
        captured.commands.size(), captured.command_bindings.size());
    MetalPointerIdentityIndex identities{identity_capacity};
    if (!identities.ready()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    identity_index_bytes = identities.layout().byte_count;
    try {
      pipeline->pipelines.reserve(captured.commands.size());
      pipeline->residency.reserve(captured.command_bindings.size());
      pipeline->command_chunks.reserve(
          static_cast<std::size_t>(actual_icb_plan.chunk_count));
      if (pipeline->pipelines.capacity() != captured.commands.size() ||
          pipeline->residency.capacity() != captured.command_bindings.size() ||
          pipeline->command_chunks.capacity() != actual_icb_plan.chunk_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      for (const MetalCommand &source : captured.commands) {
        const std::size_t binding_end =
            source.binding_begin + source.binding_count;
        if (source.pipeline == nil || (source.kind != MetalGrid::Groups &&
                                       source.kind != MetalGrid::Threads)) {
          return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
        }
        if (binding_end < source.binding_begin ||
            binding_end > captured.command_bindings.size()) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        bool inserted = false;
        if (!identities.insert((__bridge const void *)source.pipeline,
                               inserted)) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        if (inserted) {
          pipeline->pipelines.push_back(source.pipeline);
        }
      }
      identities.clear();
      const char parameter_identity = 0;
      for (const MetalCommand &source : captured.commands) {
        const std::size_t binding_end =
            source.binding_begin + source.binding_count;
        for (std::size_t binding = source.binding_begin; binding < binding_end;
             ++binding) {
          id<MTLBuffer> const buffer =
              captured.command_bindings[binding].buffer;
          bool inserted = false;
          const void *const identity =
              buffer == nil ? static_cast<const void *>(&parameter_identity)
                            : (__bridge const void *)buffer;
          if (!identities.insert(identity, inserted)) {
            return rund::AccelCheck{false, "compute_pipeline_capacity"};
          }
          if (inserted) {
            if (buffer == nil) {
              uses_parameters = true;
              parameter_residency_index = pipeline->residency.size();
            }
            pipeline->residency.push_back(buffer);
          }
        }
      }
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    } catch (const std::length_error &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (uses_parameters && captured.parameters.empty()) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  if (uses_parameters) {
    pipeline->parameters =
        [device newBufferWithLength:captured.parameters.size()
                            options:MTLResourceStorageModeShared];
    if (pipeline->parameters == nil ||
        [pipeline->parameters contents] == nullptr) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
    std::memcpy([pipeline->parameters contents], captured.parameters.data(),
                captured.parameters.size());
    if (parameter_residency_index >= pipeline->residency.size() ||
        pipeline->residency[parameter_residency_index] != nil) {
      return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
    }
    pipeline->residency[parameter_residency_index] = pipeline->parameters;
  }
  pipeline->command_count = captured.commands.size();
  std::uint64_t icb_device_bytes = 0u;
  try {
    std::size_t global_begin = 0u;
    for (std::uint64_t chunk_index = 0u;
         chunk_index < actual_icb_plan.chunk_count; ++chunk_index) {
      const bool full = chunk_index < actual_icb_plan.full_chunk_count;
      const std::uint64_t command_count =
          full ? MetalPipelineIcbFullCommandCapacity
               : actual_icb_plan.tail_command_count;
      const std::uint64_t command_capacity =
          full ? MetalPipelineIcbFullCommandCapacity
               : actual_icb_plan.tail_command_capacity;
      const std::uint32_t class_index =
          MetalIcbClassIndex(command_capacity);
      if (command_count == 0u || command_count > command_capacity ||
          class_index >= MetalPipelineIcbClassCount ||
          command_count > std::numeric_limits<std::uint32_t>::max() ||
          command_count > captured.commands.size() - global_begin) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      // CPU-authored indirect commands require Shared storage. The allocation
      // helper owns the complete calibrated descriptor/options tuple.
      id<MTLIndirectCommandBuffer> const commands =
          AllocateMetalPipelineIcb(device,
                                   static_cast<NSUInteger>(command_capacity));
      const std::uint64_t allocated =
          commands == nil
              ? 0u
              : static_cast<std::uint64_t>(commands.allocatedSize);
      if (commands == nil || commands.size != command_capacity ||
          allocated != pipeline->adapter->pipeline_icb_calibration
                           .allocated_bytes[class_index] ||
          allocated > std::numeric_limits<std::uint64_t>::max() -
                          icb_device_bytes) {
        return rund::AccelCheck{false,
                                "accel_metal_icb_calibration_mismatch"};
      }
      icb_device_bytes += allocated;
      const bool boundary_barrier =
          global_begin != 0u && captured.commands[global_begin - 1u].barrier;
      pipeline->command_chunks.push_back(MetalIcbChunk{
          .commands = commands,
          .command_count = static_cast<std::uint32_t>(command_count),
          .flags = boundary_barrier ? MetalIcbChunkBarrierBefore : 0u,
      });
      for (std::size_t local_index = 0u;
           local_index < static_cast<std::size_t>(command_count);
           ++local_index) {
        const std::size_t global_index = global_begin + local_index;
        const MetalCommand &source = captured.commands[global_index];
        const std::size_t binding_end =
            source.binding_begin + source.binding_count;
        id<MTLIndirectComputeCommand> const command =
            [commands indirectComputeCommandAtIndex:local_index];
        if (command == nil) {
          return rund::AccelCheck{false,
                                  "accel_kernel_primitive_unsupported"};
        }
        // A boundary barrier is encoded on the direct warm encoder because an
        // ICB-local barrier has no predecessor in the new native object.
        if (local_index != 0u &&
            captured.commands[global_index - 1u].barrier) {
          [command setBarrier];
        }
        [command setComputePipelineState:source.pipeline];
        for (std::size_t binding = source.binding_begin;
             binding < binding_end; ++binding) {
          const MetalCommandBinding &argument =
              captured.command_bindings[binding];
          const bool uses_parameter = argument.buffer == nil;
          id<MTLBuffer> const buffer =
              uses_parameter ? pipeline->parameters : argument.buffer;
          const NSUInteger offset =
              uses_parameter ? argument.parameter : argument.offset;
          [command setKernelBuffer:buffer offset:offset atIndex:argument.index];
        }
        if (source.kind == MetalGrid::Groups) {
          [command concurrentDispatchThreadgroups:source.grid
                            threadsPerThreadgroup:source.threads];
        } else {
          [command concurrentDispatchThreads:source.grid
                       threadsPerThreadgroup:source.threads];
        }
      }
      global_begin += static_cast<std::size_t>(command_count);
    }
    if (global_begin != captured.commands.size() ||
        pipeline->command_chunks.size() != actual_icb_plan.chunk_count ||
        icb_device_bytes != actual_icb_plan.allocated_bytes) {
      return rund::AccelCheck{false,
                              "accel_metal_icb_calibration_mismatch"};
    }
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  pipeline->warm = MetalWarmSubmission{
      .resources =
          pipeline->residency.empty() ? nullptr : pipeline->residency.data(),
      .resource_count = pipeline->residency.size(),
      .chunks = pipeline->command_chunks.data(),
      .chunk_count = pipeline->command_chunks.size(),
  };
  if (!pipeline->warm.matches(pipeline->residency, pipeline->command_chunks,
                              pipeline->command_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t recurrence_host_bytes = 0u;
  std::uint64_t recurrence_device_bytes = 0u;
  std::uint64_t recurrence_reused_bytes = 0u;
  const auto account_recurrence = [&](const std::shared_ptr<void> &resource) {
    const auto *const map =
        static_cast<const MetalMapEncodeResources *>(resource.get());
    if (map == nullptr || map->prepared == nullptr) {
      return;
    }
    std::uint64_t host = sizeof(MetalMapEncodeResources);
    for (const std::uint64_t bytes :
         {static_cast<std::uint64_t>(map->windows.capacity()) *
              sizeof(rund::kernel::ComputeDispatchWindow),
          static_cast<std::uint64_t>(map->resident.overflow_inputs.capacity()) *
              sizeof(MetalResidentBufferResult),
          static_cast<std::uint64_t>(
              map->resident.overflow_outputs.capacity()) *
              sizeof(MetalResidentBufferResult)}) {
      host = ::rund::detail::counter::SaturatingAdd(host, bytes);
    }
    // Immutable Map template vectors and native pipelines belong to the
    // pipeline-global registry and are observed there exactly once across
    // primary/alternate streams. A history proof, in contrast, is route-owned
    // and remains live through this encoded recurrence group.
    if (map->binding_owner != nullptr) {
      host = ::rund::detail::counter::SaturatingAdd(
          host, sizeof(MapRecurrenceHistory));
    }
    recurrence_host_bytes =
        ::rund::detail::counter::SaturatingAdd(recurrence_host_bytes, host);
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
      pipeline->command_chunks.capacity() * sizeof(MetalIcbChunk) +
      pipeline->residency.capacity() * sizeof(id<MTLResource>) +
      pipeline->pipelines.capacity() * sizeof(id<MTLComputePipelineState>) +
      pipeline->telemetry.capacity() * sizeof(MetalPipelineTelemetryRecord) +
      pipeline->step_evidence.capacity() *
          sizeof(PreparedPipelineStepEvidence) +
      pipeline->transducers.capacity() * sizeof(std::shared_ptr<void>) +
      recurrence_host_bytes;
  const std::uint64_t device_bytes =
      icb_device_bytes +
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
  std::uint64_t cold_workspace_bytes = identity_index_bytes;
  const auto account_workspace = [&](const std::uint64_t count,
                                     const std::uint64_t element) {
    cold_workspace_bytes = ::rund::detail::counter::SaturatingAdd(
        cold_workspace_bytes,
        ::rund::detail::counter::SaturatingMultiply(count, element));
  };
  account_workspace(native_windows.capacity(), sizeof(MetalWindow));
  account_workspace(status_bindings.capacity(),
                    sizeof(MetalPipelineStatusBindingRecord));
  account_workspace(status_sources.capacity(),
                    sizeof(MetalPipelineStatusSourceMeta));
  account_workspace(status_entries.capacity(),
                    sizeof(MetalPipelineStatusEntryMeta));
  account_workspace(status_resets.capacity(), sizeof(MetalPipelineResetMeta));
  account_workspace(telemetry_steps.capacity(),
                    sizeof(PreparedProgramStatusSlice));
  account_workspace(captured.parameters.capacity(), sizeof(std::byte));
  account_workspace(captured.command_bindings.capacity(),
                    sizeof(MetalCommandBinding));
  account_workspace(captured.commands.capacity(), sizeof(MetalCommand));
  const std::uint64_t host_peak =
      ::rund::detail::counter::SaturatingAdd(host_bytes, cold_workspace_bytes);
  memory.host = PreparedMemory{.current = host_bytes,
                               .peak = host_peak,
                               .cumulative = host_peak,
                               .budget = host_peak};
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
