#include "internal.hpp"

#include "../../../../runtime/map/api.hpp"

#include <limits>
#include <new>

namespace rund::node::accel::detail::metal_pipeline_program_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck EncodeBody(MetalPipelineBuild &build,
                            const ProgramEntry &entry,
                            const std::size_t entry_index,
                            const std::size_t program_command_begin) {
  if (entry.transducer != nullptr) {
    const PreparedProgramStatusSlice status_slice =
        build.status.slices[entry.template_index];
    const bool telemetry_empty =
        entry.telemetry_range.count == 1u &&
        build.telemetry_steps[entry.telemetry_range.first].count == 0u;
    if (!entry.transducer->recurrence.ready() ||
        entry.transducer_resource == nullptr ||
        *entry.transducer_resource == nullptr ||
        entry.resident_window == nullptr ||
        entry.resident_window->phase != BackendWindowPhase::NestedAction ||
        entry.resident_window->inner_advance != 0u ||
        entry.binding_slice.count != 0u || status_slice.count != 0u ||
        entry.resources->size() != 1u || !telemetry_empty) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    build.captured.owner = entry.resident_window->state;
    const rund::AccelCheck encoded =
        EncodeMetalMap(*build.pipeline->adapter, *entry.transducer_resource,
                       (__bridge void *)build.encoder);
    build.captured.owner = std::numeric_limits<std::uint32_t>::max();
    if (!encoded.ok) {
      return encoded;
    }
    const rund::AccelCheck capture = CheckMetalPipelineCapture(build.captured);
    if (!capture.ok) {
      return capture;
    }
    for (std::size_t command = program_command_begin;
         command < build.captured.commands.size(); ++command) {
      build.captured.commands[command].trace = true;
    }
  } else {
    for (std::size_t index = 0u; index < entry.resources->size(); ++index) {
      build.failure_context.node_route(build.entries[entry_index], index);
      MetalKernelEntry *const step = entry.resources->entry(index);
      if (step == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      build.captured.owner = entry.resident_window == nullptr
                                 ? std::numeric_limits<std::uint32_t>::max()
                                 : entry.resident_window->state;
      const rund::AccelCheck reset = EncodeMetalResets(
          *entry.resources, index, (id<MTLComputeCommandEncoder>)build.encoder);
      if (!reset.ok) {
        return reset;
      }
      if (step->barrier_before && step->resets.empty()) {
        [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      const std::size_t step_command_begin = build.captured.commands.size();
      try {
        const rund::AccelCheck gathered = EncodeMetalViewInputs(
            step->view, (id<MTLComputeCommandEncoder>)build.encoder);
        if (!gathered.ok) {
          return gathered;
        }
        const rund::AccelCheck encoded =
            EncodeMetalStep(*build.pipeline->adapter, step->ops, step->resource,
                            (id<MTLComputeCommandEncoder>)build.encoder);
        if (!encoded.ok) {
          return encoded;
        }
        const rund::AccelCheck published = EncodeMetalViewOutputs(
            step->view, (id<MTLComputeCommandEncoder>)build.encoder);
        if (!published.ok) {
          return published;
        }
      } catch (const std::bad_alloc &) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      const rund::AccelCheck capture =
          CheckMetalPipelineCapture(build.captured);
      if (!capture.ok) {
        return capture;
      }
      const std::size_t step_command_end = build.captured.commands.size();
      // The canonical public dispatch coordinate is every physical Program
      // body and View helper command. Resets were captured before this step
      // and retain their independent reset coordinate.
      for (std::size_t command_index = step_command_begin;
           command_index < step_command_end; ++command_index) {
        build.captured.commands[command_index].trace = true;
      }
      for (std::size_t command_index = step_command_begin;
           command_index + 1u < step_command_end; ++command_index) {
        build.captured.commands[command_index].barrier = true;
      }
      const rund::AccelCheck telemetry_encoded = build.EncodeTelemetry(
          build.telemetry_steps[entry.telemetry_range.first + index],
          entry.binding_slice,
          build.status.declared_steps[entry.template_index]);
      if (!telemetry_encoded.ok) {
        return telemetry_encoded;
      }
      build.captured.owner = std::numeric_limits<std::uint32_t>::max();
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_program_internal
