#include "../build.hpp"

#include "../../../runtime/map/api.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::EncodePrograms() {
  reset_command_count = captured.commands.size();
  import_count = 0u;
  if (recurrence.ready()) {
    if (!status_bindings.empty() || !pipeline->telemetry.empty() ||
        pipeline->recurrence == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const rund::AccelCheck encoded = EncodeMetalMap(
        *pipeline->adapter, pipeline->recurrence, (__bridge void *)encoder);
    if (!encoded.ok) {
      return encoded;
    }
    if (profile_steps) {
      const MetalWork work = MeasureMetalWork(std::span<const MetalCommand>{
          captured.commands.data() + reset_command_count,
          captured.commands.size() - reset_command_count});
      if (work.exact) {
        PreparedPipelineStepEvidence &row =
            pipeline->step_evidence[status.declared_steps[0u]];
        row.workgroup_count = work.workgroup_count;
        row.work_item_count = work.work_item_count;
      }
    }
  } else {
    bool scratch_seen = false;
    for (std::size_t entry_index = 0u; entry_index < entries.size();
         ++entry_index) {
      const std::uint32_t template_index =
          entries[entry_index].template_index;
      if (template_index >= templates.size() ||
          entries[entry_index].occurrence_index != entry_index ||
          (entries[entry_index].transducer != NoTileTransducer &&
           (entries[entry_index].transducer >= transducers.size() ||
            entries[entry_index].transducer >=
                pipeline->transducers.size()))) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      auto *const resources = static_cast<MetalKernelResources *>(
          entries[entry_index].prepared->get());
      const TileTransducer *const transducer =
          entries[entry_index].transducer == NoTileTransducer
              ? nullptr
              : &transducers[entries[entry_index].transducer];
      const std::shared_ptr<void> *const transducer_resource =
          transducer == nullptr
              ? nullptr
              : &pipeline->transducers[entries[entry_index].transducer];
      if (entry_index != 0u && (barriers[entry_index] != 0u ||
                                (scratch_seen && resources->shared_scratch))) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      scratch_seen = scratch_seen || resources->shared_scratch;
      captured.replacements.clear();
      const PreparedProgramStatusSlice binding_slice =
          binding_slices[template_index];
      const PreparedProgramStatusSlice telemetry_range =
          telemetry_ranges[template_index];
      const std::size_t telemetry_end =
          static_cast<std::size_t>(telemetry_range.first) +
          telemetry_range.count;
      if (telemetry_range.count != resources->size() ||
          telemetry_end > telemetry_steps.size()) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      try {
        captured.replacements.reserve(kMetalPipelineStatusBindingCapacity);
        for (std::uint32_t ordinal = binding_slice.first;
             ordinal < binding_slice.first + binding_slice.count; ++ordinal) {
          const MetalPipelineStatusBindingRecord &record =
              status_bindings[ordinal];
          if (!record.binding.replace) {
            continue;
          }
          const id<MTLBuffer> source =
              (__bridge id<MTLBuffer>)record.binding.buffer;
          captured.replacements.push_back(MetalReplacement{
              .source = source,
              .target = pipeline->raw_status,
              .source_offset = static_cast<NSUInteger>(record.binding.offset),
              .source_bytes = static_cast<NSUInteger>(record.binding.bytes),
              .target_offset = static_cast<NSUInteger>(record.raw_offset) *
                               sizeof(std::uint32_t),
          });
        }
      } catch (const std::bad_alloc &) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      const BackendWindow *const resident_window =
          entries[entry_index].recurrence.window;
      const auto window_begin = std::lower_bound(
          native_windows.begin(), native_windows.end(), entry_index,
          [](const MetalWindow &window, const std::size_t entry) {
            return window.entry < entry;
          });
      const auto window_end = std::upper_bound(
          window_begin, native_windows.end(), entry_index,
          [](const std::size_t entry, const MetalWindow &window) {
            return entry < window.entry;
          });
      // 0: nested Seed preflight, 1: ordinary post-program transition,
      // 2: nested Action transition plus the final Fold transition. A status
      // reducer closes a failed nested route directly; each later Seed
      // preflight completes the preceding Fold before admitting new work.
      const auto encode_window_control = [&](const std::uint32_t stage) {
        bool encoded = false;
        for (auto route = window_begin; route != window_end; ++route) {
          const auto phase =
              static_cast<BackendWindowPhase>(route->params.phase);
          const bool selected =
              phase == BackendWindowPhase::NestedSeed
                  ? stage == 0u
                  : (phase == BackendWindowPhase::NestedAction
                         ? stage == 2u && route->params.inner_advance != 0u
                         : (phase == BackendWindowPhase::NestedFold
                                ? stage == 2u && resident_window != nullptr &&
                                      resident_window->outer_iteration + 1u ==
                                          resident_window->outer_bound
                                : stage == 1u));
          if (!selected) {
            continue;
          }
          if (!encoded) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encoded = true;
          }
          id<MTLBuffer> const resident =
              (__bridge id<MTLBuffer>)route->resident.get();
          std::array<id<MTLBuffer>, 3u> terminals{
              (__bridge id<MTLBuffer>)route->terminals[0].get(),
              (__bridge id<MTLBuffer>)route->terminals[1].get(),
              (__bridge id<MTLBuffer>)route->terminals[2].get()};
          if (resident == nil || pipeline->states == nil ||
              std::any_of(
                  terminals.begin(), terminals.end(),
                  [](const id<MTLBuffer> value) { return value == nil; })) {
            return rund::AccelCheck{false, "accel_metal_buffer_failed"};
          }
          [encoder setComputePipelineState:advance];
          [encoder setBuffer:terminals[0] offset:0u atIndex:0u];
          [encoder setBuffer:terminals[1] offset:0u atIndex:1u];
          [encoder setBuffer:terminals[2] offset:0u atIndex:2u];
          [encoder setBuffer:resident offset:0u atIndex:3u];
          [encoder setBuffer:pipeline->states offset:0u atIndex:4u];
          [encoder setBuffer:pipeline->control offset:0u atIndex:5u];
          MetalWindowParams params = route->params;
          [encoder setBytes:&params
                     length:sizeof(params)
                    atIndex:6u];
          [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
          captured.commands.back().control = true;
          if (advance_count == std::numeric_limits<std::uint32_t>::max()) {
            return rund::AccelCheck{false, "compute_pipeline_capacity"};
          }
          ++advance_count;
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        if (encoded) {
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        return rund::AccelCheck{true, "ok"};
      };
      const rund::AccelCheck preflight = encode_window_control(0u);
      if (!preflight.ok) {
        return preflight;
      }
      const std::size_t program_command_begin = captured.commands.size();
      if (transducer != nullptr) {
        const PreparedProgramStatusSlice status_slice =
            status.slices[template_index];
        const bool telemetry_empty =
            telemetry_range.count == 1u &&
            telemetry_steps[telemetry_range.first].count == 0u;
        if (!transducer->recurrence.ready() || transducer_resource == nullptr ||
            *transducer_resource == nullptr || resident_window == nullptr ||
            resident_window->phase != BackendWindowPhase::NestedAction ||
            resident_window->inner_advance != 0u ||
            binding_slice.count != 0u || status_slice.count != 0u ||
            resources->size() != 1u || !telemetry_empty) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
        captured.owner = resident_window->state;
        const rund::AccelCheck encoded = EncodeMetalMap(
            *pipeline->adapter, *transducer_resource,
            (__bridge void *)encoder);
        captured.owner = std::numeric_limits<std::uint32_t>::max();
        if (!encoded.ok) {
          return encoded;
        }
      } else {
        for (std::size_t index = 0u; index < resources->size(); ++index) {
        MetalKernelEntry *const step = resources->entry(index);
        if (step == nullptr) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
        captured.owner = resident_window == nullptr
                             ? std::numeric_limits<std::uint32_t>::max()
                             : resident_window->state;
        const rund::AccelCheck reset = EncodeMetalResets(
            *resources, index, (id<MTLComputeCommandEncoder>)encoder);
        if (!reset.ok) {
          return reset;
        }
        if (step->barrier_before && step->resets.empty()) {
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        const std::size_t step_command_begin = captured.commands.size();
        try {
          const rund::AccelCheck gathered = EncodeMetalViewInputs(
              step->view, (id<MTLComputeCommandEncoder>)encoder);
          if (!gathered.ok) {
            return gathered;
          }
          const rund::AccelCheck encoded =
              EncodeMetalStep(*pipeline->adapter, step->ops, step->resource,
                              (id<MTLComputeCommandEncoder>)encoder);
          if (!encoded.ok) {
            return encoded;
          }
          const rund::AccelCheck published = EncodeMetalViewOutputs(
              step->view, (id<MTLComputeCommandEncoder>)encoder);
          if (!published.ok) {
            return published;
          }
        } catch (const std::bad_alloc &) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        const std::size_t step_command_end = captured.commands.size();
        for (std::size_t command_index = step_command_begin;
             command_index + 1u < step_command_end; ++command_index) {
          captured.commands[command_index].barrier = true;
        }
        const rund::AccelCheck telemetry_encoded =
            EncodeTelemetry(telemetry_steps[telemetry_range.first + index],
                            binding_slice,
                            status.declared_steps[template_index]);
        if (!telemetry_encoded.ok) {
          return telemetry_encoded;
        }
        captured.owner = std::numeric_limits<std::uint32_t>::max();
        }
      }
      const rund::AccelCheck advanced = encode_window_control(1u);
      if (!advanced.ok) {
        return advanced;
      }
      const std::size_t program_command_end = captured.commands.size();
      if (program_command_begin == program_command_end) {
        return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
      }
      if (profile_steps) {
        const MetalWork work = MeasureMetalWork(std::span<const MetalCommand>{
            captured.commands.data() + program_command_begin,
            program_command_end - program_command_begin});
        if (work.exact) {
          PreparedPipelineStepEvidence &row =
              pipeline->step_evidence[status.declared_steps[template_index]];
          if (work.workgroup_count >
                  std::numeric_limits<std::uint64_t>::max() -
                      row.workgroup_count ||
              work.work_item_count >
                  std::numeric_limits<std::uint64_t>::max() -
                      row.work_item_count) {
            return rund::AccelCheck{false, "compute_pipeline_capacity"};
          }
          row.workgroup_count += work.workgroup_count;
          row.work_item_count += work.work_item_count;
        }
      }
      // Nested status storage belongs to the same physical occurrence as the
      // payload that produced it. Compact route templates reuse these buffers
      // across outer windows, so an occurrence suppressed by nested preflight
      // must suppress its import and reduction as well; otherwise a stopped
      // route can fold stale status telemetry from an earlier window.
      //
      // Ordinary windows perform their transition before status reduction;
      // retaining their established unowned status path avoids suppressing
      // the terminating occurrence's status when that transition closes it.
      captured.owner =
          resident_window != nullptr && resident_window->nested()
              ? resident_window->state
              : std::numeric_limits<std::uint32_t>::max();
      captured.replacements.clear();
      bool imported = false;
      for (std::uint32_t ordinal = binding_slice.first;
           ordinal < binding_slice.first + binding_slice.count; ++ordinal) {
        const MetalPipelineStatusBindingRecord &record =
            status_bindings[ordinal];
        if (record.binding.replace) {
          continue;
        }
        if (!imported) {
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          imported = true;
        }
        const std::array<std::uint32_t, 2u> range{record.raw_offset,
                                                  record.raw_count};
        [encoder setComputePipelineState:import];
        [encoder setBuffer:(__bridge id<MTLBuffer>)record.binding.buffer
                    offset:static_cast<NSUInteger>(record.binding.offset)
                   atIndex:0u];
        [encoder setBuffer:pipeline->raw_status offset:0u atIndex:1u];
        [encoder setBytes:range.data() length:sizeof(range) atIndex:2u];
        const NSUInteger import_threads = record.raw_count;
        const NSUInteger import_width =
            std::min(import_threads, [import maxTotalThreadsPerThreadgroup]);
        [encoder dispatchThreads:MTLSizeMake(import_threads, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(import_width, 1u, 1u)];
        ++import_count;
      }
      const PreparedProgramStatusSlice status_slice =
          status.slices[template_index];
      const std::size_t status_end =
          static_cast<std::size_t>(status_slice.first) + status_slice.count;
      const std::size_t source_end =
          static_cast<std::size_t>(binding_slice.first) + binding_slice.count;
      if (status_end > status_entries.size() ||
          source_end > status_sources.size()) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      if (status_slice.count != 0u) {
        if (binding_slice.count == 0u) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
        try {
          occurrence_status_sources.assign(
              status_sources.begin() + binding_slice.first,
              status_sources.begin() + source_end);
        } catch (const std::bad_alloc &) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        const std::uint32_t no_coordinate = PreparedPipelineNoStep;
        std::uint32_t failed_outer = no_coordinate;
        std::uint32_t failed_inner = no_coordinate;
        std::uint32_t failed_phase = 0u;
        if (resident_window != nullptr && resident_window->nested()) {
          failed_outer = resident_window->outer_iteration;
          failed_phase =
              static_cast<std::uint32_t>(resident_window->nested_phase());
          if (resident_window->phase == BackendWindowPhase::NestedAction) {
            failed_inner = resident_window->inner_iteration;
          }
        }
        for (std::size_t source = 0u; source < binding_slice.count; ++source) {
          occurrence_status_sources[source].failed_outer_window = failed_outer;
          occurrence_status_sources[source].failed_inner_iteration =
              failed_inner;
          occurrence_status_sources[source].failed_nested_phase = failed_phase;
        }
        MetalPipelineStatusParams fold = status_params;
        fold.status_count = status_slice.count;
        fold.source_count = binding_slice.count;
        if (resident_window != nullptr && resident_window->nested()) {
          fold.window_state = resident_window->state;
          fold.window_stop = resident_window->outer_iteration + 1u;
          fold.window_inner_advance =
              resident_window->phase == BackendWindowPhase::NestedFold
                  ? resident_window->inner_advance
                  : 0u;
        }
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:reduce];
        [encoder setBuffer:pipeline->raw_status offset:0u atIndex:0u];
        [encoder setBuffer:pipeline->control offset:0u atIndex:1u];
        [encoder
            setBytes:status_entries.data() + status_slice.first
              length:status_slice.count * sizeof(MetalPipelineStatusEntryMeta)
             atIndex:2u];
        [encoder
            setBytes:occurrence_status_sources.data()
              length:binding_slice.count * sizeof(MetalPipelineStatusSourceMeta)
             atIndex:3u];
        [encoder setBytes:&fold length:sizeof(fold) atIndex:4u];
        id<MTLBuffer> const states =
            pipeline->states == nil ? pipeline->control : pipeline->states;
        [encoder setBuffer:states offset:0u atIndex:5u];
        if (profile_steps) {
          [encoder setBuffer:pipeline->step_control offset:0u atIndex:6u];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(kMetalPipelineReductionWidth,
                                                  1u, 1u)];
        captured.commands.back().control = true;
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        ++fold_count;
      }
      captured.owner = std::numeric_limits<std::uint32_t>::max();
      const rund::AccelCheck folded = encode_window_control(2u);
      if (!folded.ok) {
        return folded;
      }
      if (resident_window != nullptr &&
          resident_window->advances_outer_state() &&
          resident_window->outer_iteration + 1u ==
              resident_window->outer_bound) {
        for (const MetalPublish &publication : native_publications) {
          if (publication.params.state != resident_window->state ||
              publication.params.final > 2u ||
              publication.params.count == 0u) {
            continue;
          }
          const std::uint32_t final = publication.params.final;
          id<MTLBuffer> const target =
              (__bridge id<MTLBuffer>)publication.sources[final].get();
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
          MetalPublishParams params = publication.params;
          params.target_offset_words = params.source_offset_words[final];
          params.target_stride_words = params.source_stride_words[final];
          params.stop = std::numeric_limits<std::uint32_t>::max();
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [encoder setComputePipelineState:publish];
          [encoder setBuffer:sources[0] offset:0u atIndex:0u];
          [encoder setBuffer:sources[1] offset:0u atIndex:1u];
          [encoder setBuffer:sources[2] offset:0u atIndex:2u];
          [encoder setBuffer:target offset:0u atIndex:3u];
          [encoder setBuffer:pipeline->control offset:0u atIndex:4u];
          [encoder setBuffer:pipeline->states offset:0u atIndex:5u];
          [encoder setBytes:&params length:sizeof(params) atIndex:6u];
          const NSUInteger count =
              static_cast<NSUInteger>(publication.params.count);
          const NSUInteger width =
              std::min(count, [publish maxTotalThreadsPerThreadgroup]);
          captured.owner = std::numeric_limits<std::uint32_t>::max();
          [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
          captured.commands.back().control = true;
          ++pipeline->dispatch_count;
          ++canonicalize_count;
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
      }
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
