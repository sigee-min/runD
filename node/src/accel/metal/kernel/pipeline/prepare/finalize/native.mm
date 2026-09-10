#include "internal.hpp"

#include "../../icb.hpp"

#include "../../../../../kernel/backend/exception.hpp"

#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck FinalizeMetalNative(
    MetalPipelineBuild &build,
    MetalPipelineFinalizeProjection &projection) {
  if (projection.uses_parameters) {
    build.pipeline->parameters =
        [build.device newBufferWithLength:build.captured.parameters.size()
                                  options:MTLResourceStorageModeShared];
    if (build.pipeline->parameters == nil ||
        [build.pipeline->parameters contents] == nullptr) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
    std::memcpy([build.pipeline->parameters contents],
                build.captured.parameters.data(),
                build.captured.parameters.size());
    if (projection.parameter_residency_index >=
            build.pipeline->residency.size() ||
        build.pipeline->residency[projection.parameter_residency_index] != nil) {
      return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
    }
    build.pipeline->residency[projection.parameter_residency_index] =
        build.pipeline->parameters;
  }
  build.pipeline->command_count = build.captured.commands.size();
  projection.icb_device_bytes = 0u;
  try {
    std::size_t global_begin = 0u;
    for (std::uint64_t chunk_index = 0u;
         chunk_index < projection.actual_icb_plan.chunk_count; ++chunk_index) {
      const bool full =
          chunk_index < projection.actual_icb_plan.full_chunk_count;
      const std::uint64_t command_count =
          full ? MetalPipelineIcbFullCommandCapacity
               : projection.actual_icb_plan.tail_command_count;
      const std::uint64_t command_capacity =
          full ? MetalPipelineIcbFullCommandCapacity
               : projection.actual_icb_plan.tail_command_capacity;
      const std::uint32_t class_index = MetalIcbClassIndex(command_capacity);
      if (command_count == 0u || command_count > command_capacity ||
          class_index >= MetalPipelineIcbClassCount ||
          command_count > std::numeric_limits<std::uint32_t>::max() ||
          command_count > build.captured.commands.size() - global_begin) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      // CPU-authored indirect commands require Shared storage. The allocation
      // helper owns the complete calibrated descriptor/options tuple.
      id<MTLIndirectCommandBuffer> const commands = AllocateMetalPipelineIcb(
          build.device, static_cast<NSUInteger>(command_capacity));
      const std::uint64_t allocated =
          commands == nil ? 0u
                          : static_cast<std::uint64_t>(commands.allocatedSize);
      if (commands == nil || commands.size != command_capacity ||
          allocated != build.pipeline->adapter->pipeline_icb_calibration
                           .allocated_bytes[class_index] ||
          allocated >
              std::numeric_limits<std::uint64_t>::max() -
                  projection.icb_device_bytes) {
        return rund::AccelCheck{false,
                                "accel_metal_icb_calibration_mismatch"};
      }
      projection.icb_device_bytes += allocated;
      const bool boundary_barrier =
          global_begin != 0u &&
          build.captured.commands[global_begin - 1u].barrier;
      build.pipeline->command_chunks.push_back(MetalIcbChunk{
          .commands = commands,
          .command_count = static_cast<std::uint32_t>(command_count),
          .flags = boundary_barrier ? MetalIcbChunkBarrierBefore : 0u,
      });
      for (std::size_t local_index = 0u;
           local_index < static_cast<std::size_t>(command_count);
           ++local_index) {
        const std::size_t global_index = global_begin + local_index;
        const MetalCommand &source = build.captured.commands[global_index];
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
            build.captured.commands[global_index - 1u].barrier) {
          [command setBarrier];
        }
        [command setComputePipelineState:source.pipeline];
        for (std::size_t binding = source.binding_begin; binding < binding_end;
             ++binding) {
          const MetalCommandBinding &argument =
              build.captured.command_bindings[binding];
          const bool uses_parameter = argument.buffer == nil;
          id<MTLBuffer> const buffer =
              uses_parameter ? build.pipeline->parameters : argument.buffer;
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
    if (global_begin != build.captured.commands.size() ||
        build.pipeline->command_chunks.size() !=
            projection.actual_icb_plan.chunk_count ||
        projection.icb_device_bytes != projection.actual_icb_plan.allocated_bytes) {
      return rund::AccelCheck{false,
                              "accel_metal_icb_calibration_mismatch"};
    }
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  build.pipeline->warm = MetalWarmSubmission{
      .resources = build.pipeline->residency.empty()
                       ? nullptr
                       : build.pipeline->residency.data(),
      .resource_count = build.pipeline->residency.size(),
      .chunks = build.pipeline->command_chunks.data(),
      .chunk_count = build.pipeline->command_chunks.size(),
  };
  // command_count is the complete ICB stream, including control commands;
  // dispatch_count is the public payload-dispatch evidence. Validate native
  // ownership/cardinality here without conflating those two domains.
  if (!build.pipeline->warm.matches(build.pipeline->residency,
                                    build.pipeline->command_chunks,
                                    build.pipeline->command_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
