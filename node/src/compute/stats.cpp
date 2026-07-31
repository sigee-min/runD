#include "stats.hpp"

#include "program/state.hpp"
#include "run/state.hpp"

#include <accel/kernel/evidence.hpp>

#include <bit>

namespace rund::compute::detail {

Stats stats_from_evidence(const Backend backend,
                          const rund::AccelEvidence &evidence,
                          const std::uint64_t graph_read_bytes) noexcept {
  return Stats{
      .backend = backend,
      .pipeline_compiles = evidence.pipeline_compile_count,
      .buffer_allocations = evidence.buffer_allocation_count,
      .dispatches = evidence.dispatch_count,
      .command_submits = evidence.command_submit_count,
      .command_capacity = evidence.command_capacity,
      .command_inflight_peak = evidence.command_inflight_peak,
      .command_capacity_rejections = evidence.command_capacity_rejection_count,
      .uploaded_bytes = evidence.host_to_device_bytes,
      .downloaded_bytes = evidence.device_to_host_bytes,
      .pipeline_cache_hits = evidence.pipeline_cache_hit_count,
      .pipeline_cache_evictions = evidence.pipeline_cache_eviction_count,
      .buffer_reuses = evidence.buffer_reuse_hit_count,
      .descriptor_pool_creations = evidence.descriptor_pool_create_count,
      .descriptor_set_allocations = evidence.descriptor_set_allocate_count,
      .descriptor_reuses = evidence.descriptor_reuse_hit_count,
      .original_dispatches = evidence.original_dispatch_count,
      .final_dispatches = evidence.final_dispatch_count,
      .fusions =
          evidence.original_operation_count > evidence.fused_operation_count
              ? evidence.original_operation_count -
                    evidence.fused_operation_count
              : 0u,
      .fusion_rejections = evidence.fusion_rejection_count,
      .internal_roundtrip_bytes =
          evidence.internal_producer_consumer_roundtrip_bytes,
      .external_roundtrip_bytes =
          evidence.external_producer_consumer_roundtrip_bytes,
      .reset_bytes = evidence.reset_bytes,
      .reset_commands = evidence.reset_command_count,
      .graph_read_bytes = graph_read_bytes,
      .kernel_ns = evidence.accel_kernel_ns,
      .kernel_samples = evidence.accel_timestamp_count,
      .shader_compile_ns = evidence.shader_compile_ns,
      .spirv_compile_ns = evidence.spirv_compile_ns,
      .pipeline_create_ns = evidence.pipeline_create_ns,
      .descriptor_setup_ns = evidence.descriptor_setup_ns,
      .submit_wait_ns = evidence.command_submit_wait_ns,
      .readback_ns = evidence.readback_ns,
      .graph_hash = evidence.graph_id_hi ^ std::rotl(evidence.graph_id_lo, 1),
      .control =
          ControlStats{
              .generated_item_count = evidence.generated_item_count,
              .generated_capacity = evidence.generated_capacity,
              .indirect_dispatch_count = evidence.indirect_dispatch_count,
              .indirect_work_item_count = evidence.indirect_work_item_count,
              .iteration_count = evidence.iteration_count,
              .skipped_iteration_count = evidence.skipped_iteration_count,
              .conflict_count = evidence.conflict_count,
              .overflow_ordinal = evidence.overflow_ordinal,
          },
  };
}

Stats run_stats(const RunState &run) noexcept { return run.stats; }

} // namespace rund::compute::detail
