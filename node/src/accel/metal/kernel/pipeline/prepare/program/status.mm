#include "internal.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::node::accel::detail::metal_pipeline_program_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck EncodeStatus(MetalPipelineBuild &build,
                              const ProgramEntry &entry) {
  build.captured.owner =
      entry.resident_window != nullptr && entry.resident_window->nested()
          ? entry.resident_window->state
          : std::numeric_limits<std::uint32_t>::max();
  build.captured.replacements = {};
  build.captured.replacement_target = nil;
  bool imported = false;
  for (std::uint32_t ordinal = entry.binding_slice.first;
       ordinal < entry.binding_slice.first + entry.binding_slice.count;
       ++ordinal) {
    const MetalPipelineStatusBindingRecord &record =
        build.status_bindings[ordinal];
    if (record.binding.replace) {
      continue;
    }
    if (!imported) {
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      imported = true;
    }
    const std::array<std::uint32_t, 2u> range{record.raw_offset,
                                              record.raw_count};
    [build.encoder setComputePipelineState:build.import];
    [build.encoder setBuffer:(__bridge id<MTLBuffer>)record.binding.buffer
                      offset:static_cast<NSUInteger>(record.binding.offset)
                     atIndex:0u];
    [build.encoder setBuffer:build.pipeline->raw_status offset:0u atIndex:1u];
    [build.encoder setBytes:range.data() length:sizeof(range) atIndex:2u];
    const NSUInteger import_threads = record.raw_count;
    const NSUInteger import_width =
        std::min(import_threads, [build.import maxTotalThreadsPerThreadgroup]);
    [build.encoder dispatchThreads:MTLSizeMake(import_threads, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(import_width, 1u, 1u)];
    ++build.import_count;
  }
  const PreparedProgramStatusSlice status_slice =
      build.status.slices[entry.template_index];
  const std::size_t status_end =
      static_cast<std::size_t>(status_slice.first) + status_slice.count;
  const std::size_t source_end =
      static_cast<std::size_t>(entry.binding_slice.first) +
      entry.binding_slice.count;
  if (status_end > build.status_entries.size() ||
      source_end > build.status_sources.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (status_slice.count != 0u) {
    if (entry.binding_slice.count == 0u) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const std::uint32_t no_coordinate = PreparedPipelineNoStep;
    std::uint32_t failed_outer = no_coordinate;
    std::uint32_t failed_inner = no_coordinate;
    std::uint32_t failed_phase = PipelineNestedPhaseNoneCode;
    if (entry.resident_window != nullptr && entry.resident_window->nested()) {
      failed_outer = entry.resident_window->outer_iteration;
      rund::compute::PipelineNestedPhase public_phase{};
      if (!entry.resident_window->nested_phase(public_phase) ||
          !EncodePipelineNestedPhase(public_phase, failed_phase)) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      if (entry.resident_window->phase == BackendWindowPhase::NestedAction) {
        failed_inner = entry.resident_window->inner_iteration;
      }
    }
    const std::span<MetalPipelineStatusSourceMeta> occurrence_sources =
        std::span<MetalPipelineStatusSourceMeta>{build.status_sources}.subspan(
            entry.binding_slice.first, entry.binding_slice.count);
    for (MetalPipelineStatusSourceMeta &source : occurrence_sources) {
      if (source.failed_outer_window != no_coordinate ||
          source.failed_inner_iteration != no_coordinate ||
          source.failed_nested_phase != PipelineNestedPhaseNoneCode) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      source.failed_outer_window = failed_outer;
      source.failed_inner_iteration = failed_inner;
      source.failed_nested_phase = failed_phase;
    }
    MetalPipelineStatusParams fold = build.status_params;
    fold.status_count = status_slice.count;
    fold.source_count = entry.binding_slice.count;
    if (entry.resident_window != nullptr && entry.resident_window->nested()) {
      fold.window_state = entry.resident_window->state;
      fold.window_stop = entry.resident_window->outer_iteration + 1u;
      fold.window_inner_advance =
          entry.resident_window->phase == BackendWindowPhase::NestedFold
              ? entry.resident_window->inner_advance
              : 0u;
    }
    [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [build.encoder setComputePipelineState:build.reduce];
    [build.encoder setBuffer:build.pipeline->raw_status offset:0u atIndex:0u];
    [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:1u];
    [build.encoder
        setBytes:build.status_entries.data() + status_slice.first
          length:status_slice.count * sizeof(MetalPipelineStatusEntryMeta)
         atIndex:2u];
    [build.encoder setBytes:occurrence_sources.data()
                     length:occurrence_sources.size_bytes()
                    atIndex:3u];
    // setBytes snapshots the row immediately into captured.parameters.
    // Restore canonical template metadata before the next occurrence reuses
    // this slice; a second source-row owner is unnecessary.
    for (MetalPipelineStatusSourceMeta &source : occurrence_sources) {
      source.failed_outer_window = no_coordinate;
      source.failed_inner_iteration = no_coordinate;
      source.failed_nested_phase = PipelineNestedPhaseNoneCode;
    }
    [build.encoder setBytes:&fold length:sizeof(fold) atIndex:4u];
    id<MTLBuffer> const states = build.pipeline->states == nil
                                     ? build.pipeline->control
                                     : build.pipeline->states;
    [build.encoder setBuffer:states offset:0u atIndex:5u];
    if (build.profile_steps) {
      [build.encoder setBuffer:build.pipeline->step_control
                        offset:0u
                       atIndex:6u];
    }
    [build.encoder
         dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(kMetalPipelineReductionWidth, 1u,
                                          1u)];
    const rund::AccelCheck capture = CheckMetalPipelineCapture(build.captured);
    if (!capture.ok) {
      return capture;
    }
    build.captured.commands.back().control = true;
    build.captured.commands.back().trace = false;
    [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    ++build.fold_count;
  }
  build.captured.owner = std::numeric_limits<std::uint32_t>::max();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_program_internal
