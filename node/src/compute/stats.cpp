#include "stats.hpp"

#include "program/state.hpp"
#include "run/state.hpp"

#include <accel/kernel/evidence.hpp>

#include <bit>

namespace rund::compute::detail {

Stats stats_from_evidence(const Backend backend,
                          const rund::AccelEvidence &evidence,
                          const std::uint64_t graph_read_bytes) noexcept {
  const rund::AccelWorkFacts &work = evidence.run.work;
  const rund::AccelTimeFacts &time = evidence.run.time;
  const rund::AccelTransferFacts &transfer = evidence.run.transfer;
  const rund::AccelAllocationFacts &allocations = evidence.run.allocations;
  return Stats{
      .backend = backend,
      .pipeline_compiles = allocations.pipeline_compile_count,
      .buffer_allocations = allocations.buffer_allocation_count,
      .dispatches = work.dispatch_count,
      .command_submits = work.command_submit_count,
      .command_capacity = work.command_capacity,
      .command_inflight_peak = work.command_inflight_peak,
      .command_capacity_rejections = work.command_capacity_rejection_count,
      .uploaded_bytes = transfer.host_to_device_bytes,
      .downloaded_bytes = transfer.device_to_host_bytes,
      .pipeline_cache_hits = allocations.pipeline_cache_hit_count,
      .pipeline_cache_evictions = allocations.pipeline_cache_eviction_count,
      .buffer_reuses = allocations.buffer_reuse_hit_count,
      .descriptor_pool_creations = allocations.descriptor_pool_create_count,
      .descriptor_set_allocations = allocations.descriptor_set_allocate_count,
      .descriptor_reuses = allocations.descriptor_reuse_hit_count,
      .original_dispatches = work.original_dispatch_count,
      .final_dispatches = work.final_dispatch_count,
      .fusions =
          work.original_operation_count > work.fused_operation_count
              ? work.original_operation_count - work.fused_operation_count
              : 0u,
      .fusion_rejections = work.fusion_rejection_count,
      .internal_roundtrip_bytes =
          transfer.internal_producer_consumer_roundtrip_bytes,
      .external_roundtrip_bytes =
          transfer.external_producer_consumer_roundtrip_bytes,
      .reset_bytes = work.reset_bytes,
      .reset_commands = work.reset_command_count,
      .graph_read_bytes = graph_read_bytes,
      .kernel_ns = time.accel_kernel_ns,
      .kernel_samples = time.accel_timestamp_count,
      .shader_compile_ns = time.shader_compile_ns,
      .spirv_compile_ns = time.spirv_compile_ns,
      .pipeline_create_ns = time.pipeline_create_ns,
      .descriptor_setup_ns = time.descriptor_setup_ns,
      .submit_wait_ns = time.command_submit_wait_ns,
      .readback_ns = time.readback_ns,
      .graph_hash = evidence.identity.graph_id_hi ^
                    std::rotl(evidence.identity.graph_id_lo, 1),
      .control =
          ControlStats{
              .generated_item_count = work.generated_item_count,
              .generated_capacity = work.generated_capacity,
              .indirect_dispatch_count = work.indirect_dispatch_count,
              .indirect_work_item_count = work.indirect_work_item_count,
              .iteration_count = work.iteration_count,
              .skipped_iteration_count = work.skipped_iteration_count,
              .conflict_count = work.conflict_count,
              .overflow_ordinal = work.overflow_ordinal,
          },
  };
}

Stats run_stats(const RunState &run) noexcept { return run.stats; }

} // namespace rund::compute::detail
