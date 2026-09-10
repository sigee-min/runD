#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail::metal_pipeline_program_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck PrepareEntry(MetalPipelineBuild &build,
                              const std::size_t entry_index, bool &scratch_seen,
                              ProgramEntry &entry) {
  build.failure_context.occurrence_route(build.entries[entry_index]);
  const std::uint32_t template_index =
      build.entries[entry_index].template_index;
  if (template_index >= build.templates.size() ||
      build.entries[entry_index].occurrence_index != entry_index ||
      (build.entries[entry_index].transducer != NoTileTransducer &&
       (build.entries[entry_index].transducer >= build.transducers.size() ||
        build.entries[entry_index].transducer >=
            build.pipeline->transducers.size()))) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  build.captured.declared_step = build.status.declared_steps[template_index];
  entry.template_index = template_index;
  entry.resources = static_cast<MetalKernelResources *>(
      build.entries[entry_index].prepared->get());
  entry.transducer =
      build.entries[entry_index].transducer == NoTileTransducer
          ? nullptr
          : &build.transducers[build.entries[entry_index].transducer];
  entry.transducer_resource =
      entry.transducer == nullptr
          ? nullptr
          : &build.pipeline->transducers[build.entries[entry_index].transducer];
  if (entry_index != 0u &&
      (build.barriers[entry_index] != 0u ||
       (scratch_seen && entry.resources->shared_scratch))) {
    [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  scratch_seen = scratch_seen || entry.resources->shared_scratch;
  build.captured.replacements = {};
  build.captured.replacement_target = nil;
  entry.binding_slice = build.binding_slices[template_index];
  entry.telemetry_range = build.telemetry_ranges[template_index];
  const std::size_t telemetry_end =
      static_cast<std::size_t>(entry.telemetry_range.first) +
      entry.telemetry_range.count;
  if (entry.telemetry_range.count != entry.resources->size() ||
      telemetry_end > build.telemetry_steps.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const std::size_t replacement_end =
      static_cast<std::size_t>(entry.binding_slice.first) +
      entry.binding_slice.count;
  if (replacement_end > build.status_bindings.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  build.captured.replacements =
      std::span<const MetalPipelineStatusBindingRecord>{build.status_bindings}
          .subspan(entry.binding_slice.first, entry.binding_slice.count);
  build.captured.replacement_target = build.pipeline->raw_status;
  entry.resident_window = build.entries[entry_index].recurrence.window;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_program_internal
