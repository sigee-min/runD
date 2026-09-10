#include "../internal.hpp"

#include "../../../identity/index.hpp"

#include "../../../../../../kernel/backend/exception.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
FinalizeMetalProjectionIdentity(MetalPipelineBuild &build,
                                MetalPipelineFinalizeProjection &projection) {
  const PreparedKernelPipelineReservation &limit =
      build.template_registry.limit;
  projection.actual_icb_plan =
      PlanMetalIcbChunks(build.captured.commands.size(),
                         build.pipeline->adapter->pipeline_icb_calibration);
  if (!limit.ok ||
      build.captured.commands.size() > limit.backend_command_count ||
      build.captured.command_bindings.size() >
          limit.backend_command_binding_count ||
      build.captured.parameters.size() > limit.backend_parameter_bytes ||
      build.captured.parameters.capacity() > limit.backend_parameter_bytes ||
      !projection.actual_icb_plan.ok ||
      projection.actual_icb_plan.chunk_count == 0u ||
      projection.actual_icb_plan.chunk_count >
          limit.backend_command_chunk_count ||
      projection.actual_icb_plan.allocated_bytes >
          limit.backend_command_native_bytes) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  // Freeze first-command order with one membership-only pointer index. Its
  // 2x power-of-two storage is reused for pipeline states and resources, then
  // released before native allocation; no hash owner reaches the warm path.
  projection.identity_index_bytes = 0u;
  projection.uses_parameters = false;
  projection.parameter_residency_index =
      std::numeric_limits<std::size_t>::max();
  {
    const std::uint64_t identity_capacity = std::max<std::uint64_t>(
        build.captured.commands.size(), build.captured.command_bindings.size());
    MetalPointerIdentityIndex identities{identity_capacity};
    if (!identities.ready()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    projection.identity_index_bytes = identities.layout().byte_count;
    try {
      build.pipeline->pipelines.reserve(build.captured.commands.size());
      build.pipeline->residency.reserve(build.captured.command_bindings.size());
      build.pipeline->command_chunks.reserve(
          static_cast<std::size_t>(projection.actual_icb_plan.chunk_count));
      build.pipeline->residency_steps.reserve(build.status.declared_step_count);
      build.pipeline->trace_commands.reserve(
          static_cast<std::size_t>(build.pipeline->dispatch_count));
      if (build.pipeline->pipelines.capacity() !=
              build.captured.commands.size() ||
          build.pipeline->residency.capacity() !=
              build.captured.command_bindings.size() ||
          build.pipeline->command_chunks.capacity() !=
              projection.actual_icb_plan.chunk_count ||
          build.pipeline->residency_steps.capacity() !=
              build.status.declared_step_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      build.pipeline->residency_steps.resize(build.status.declared_step_count);
      for (std::size_t index = 0u; index < build.captured.commands.size();
           ++index) {
        if (build.captured.commands[index].trace) {
          build.pipeline->trace_commands.push_back(index);
        }
      }
      if (build.pipeline->trace_commands.size() !=
              build.pipeline->dispatch_count ||
          build.pipeline->trace_commands.capacity() !=
              build.pipeline->dispatch_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      for (const MetalCommand &source : build.captured.commands) {
        const std::size_t binding_end =
            source.binding_begin + source.binding_count;
        if (source.pipeline == nil || (source.kind != MetalGrid::Groups &&
                                       source.kind != MetalGrid::Threads)) {
          return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
        }
        if (binding_end < source.binding_begin ||
            binding_end > build.captured.command_bindings.size()) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        bool inserted = false;
        if (!identities.insert((__bridge const void *)source.pipeline,
                               inserted)) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        if (inserted) {
          build.pipeline->pipelines.push_back(source.pipeline);
        }
      }
      identities.clear();
      const char parameter_identity = 0;
      for (const MetalCommand &source : build.captured.commands) {
        const std::size_t binding_end =
            source.binding_begin + source.binding_count;
        for (std::size_t binding = source.binding_begin; binding < binding_end;
             ++binding) {
          id<MTLBuffer> const buffer =
              build.captured.command_bindings[binding].buffer;
          bool inserted = false;
          const void *const identity =
              buffer == nil ? static_cast<const void *>(&parameter_identity)
                            : (__bridge const void *)buffer;
          if (!identities.insert(identity, inserted)) {
            return rund::AccelCheck{false, "compute_pipeline_capacity"};
          }
          if (inserted) {
            if (buffer == nil) {
              projection.uses_parameters = true;
              projection.parameter_residency_index =
                  build.pipeline->residency.size();
            }
            build.pipeline->residency.push_back(buffer);
          }
        }
      }
    } catch (...) {
      backend_exception::RethrowUnlessCapacityException();
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
