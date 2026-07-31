#include "compute.hpp"

#include <rund/counter.hpp>

namespace rund::telemetry::detail {

Event ComputeEvent(const compute::Stats &stats, const compute::Code code,
                   const Level level) noexcept {
  const bool timing_available =
      stats.kernel_samples != 0u || stats.shader_compile_ns != 0u ||
      stats.spirv_compile_ns != 0u || stats.pipeline_create_ns != 0u ||
      stats.descriptor_setup_ns != 0u || stats.submit_wait_ns != 0u ||
      stats.readback_ns != 0u;
  const bool detailed = level == Level::Detail && timing_available;
  Event event{
      .source = Source::Compute,
      .level = detailed ? Level::Detail : Level::Basic,
      .compute =
          {
              .backend = stats.backend,
              .code = code,
              .graph = stats.graph_hash,
              .workers = stats.worker_count,
              .active_workers = stats.participating_workers,
              .tiles = stats.tile_count,
              .dispatches = stats.dispatches,
              .command_submits = stats.command_submits,
              .buffer_allocations = stats.buffer_allocations,
              .buffer_reuses = stats.buffer_reuses,
              .copied_bytes = ::rund::detail::counter::SaturatingAdd(
                  stats.uploaded_bytes, stats.downloaded_bytes),
              .graph_read_bytes = stats.graph_read_bytes,
              .kernel_ns = detailed ? stats.kernel_ns : 0u,
              .kernel_samples = detailed ? stats.kernel_samples : 0u,
              .submit_wait_ns = detailed ? stats.submit_wait_ns : 0u,
          },
      .queue =
          {
              .depth = stats.command_inflight_peak,
              .capacity = stats.command_capacity,
          },
  };
  if (detailed) {
    event.detail.prepare_ns = ::rund::detail::counter::SaturatingAdd(
        ::rund::detail::counter::SaturatingAdd(stats.shader_compile_ns,
                                             stats.spirv_compile_ns),
        ::rund::detail::counter::SaturatingAdd(stats.pipeline_create_ns,
                                             stats.descriptor_setup_ns));
    event.detail.work_ns =
        stats.submit_wait_ns != 0u ? stats.submit_wait_ns : stats.kernel_ns;
    event.detail.finish_ns = stats.readback_ns;
  }
  return event;
}

Findings ComputeFindings(const compute::Stats &stats) noexcept {
  return ComputeEvent(stats, compute::Code::Ok, Level::Detail).findings();
}

} // namespace rund::telemetry::detail
