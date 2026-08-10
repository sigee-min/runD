#include "stats.hpp"

#include "program/state.hpp"
#include "run/state.hpp"

#include <accel/kernel/evidence.hpp>

#include <algorithm>
#include <bit>
#include <rund/counter.hpp>

namespace rund::compute::detail {

Stats stats_from_evidence(const Backend backend,
                          const rund::AccelEvidence &evidence,
                          const std::uint64_t graph_read_bytes) noexcept {
  Stats stats{
      .backend = backend,
      .graph_read_bytes = graph_read_bytes,
      .graph_hash = evidence.identity.graph_id_hi ^
                    std::rotl(evidence.identity.graph_id_lo, 1),
  };
  accumulate_run_facts(stats, evidence.run);
  return stats;
}

Stats run_stats(const RunState &run) noexcept { return run.stats; }

void accumulate_run_facts(Stats &stats,
                          const rund::AccelRunFacts &facts) noexcept {
  const auto add = [](std::uint64_t &target,
                      const std::uint64_t value) noexcept {
    target = ::rund::detail::counter::SaturatingAdd(target, value);
  };
  const rund::AccelWorkFacts &work = facts.work;
  const rund::AccelTimeFacts &time = facts.time;
  const rund::AccelTransferFacts &transfer = facts.transfer;
  const rund::AccelAllocationFacts &allocations = facts.allocations;
  switch (facts.preparation_evidence) {
  case rund::AccelPreparationEvidenceSource::OwnerLocal:
    stats.pipeline.preparation_evidence = PreparationEvidenceSource::OwnerLocal;
    break;
  case rund::AccelPreparationEvidenceSource::BackendGlobalOnly:
    stats.pipeline.preparation_evidence =
        PreparationEvidenceSource::BackendGlobalOnly;
    break;
  case rund::AccelPreparationEvidenceSource::Unavailable:
    break;
  }
  add(stats.dispatches, work.dispatch_count);
  add(stats.command_submits, work.command_submit_count);
  stats.command_capacity =
      std::max(stats.command_capacity, work.command_capacity);
  stats.command_inflight_peak =
      std::max(stats.command_inflight_peak, work.command_inflight_peak);
  add(stats.command_capacity_rejections, work.command_capacity_rejection_count);
  add(stats.original_dispatches, work.original_dispatch_count);
  add(stats.final_dispatches, work.final_dispatch_count);
  add(stats.fusions,
      work.original_operation_count > work.fused_operation_count
          ? work.original_operation_count - work.fused_operation_count
          : 0u);
  add(stats.fusion_rejections, work.fusion_rejection_count);
  add(stats.reset_bytes, work.reset_bytes);
  add(stats.reset_commands, work.reset_command_count);
  add(stats.control.generated_item_count, work.generated_item_count);
  add(stats.control.generated_capacity, work.generated_capacity);
  add(stats.control.indirect_dispatch_count, work.indirect_dispatch_count);
  add(stats.control.indirect_work_item_count, work.indirect_work_item_count);
  add(stats.control.iteration_count, work.iteration_count);
  add(stats.control.skipped_iteration_count, work.skipped_iteration_count);
  add(stats.control.conflict_count, work.conflict_count);
  stats.control.overflow_ordinal =
      std::min(stats.control.overflow_ordinal, work.overflow_ordinal);
  add(stats.kernel_ns, time.accel_kernel_ns);
  add(stats.kernel_samples, time.accel_timestamp_count);
  add(stats.shader_compile_ns, time.shader_compile_ns);
  add(stats.spirv_compile_ns, time.spirv_compile_ns);
  add(stats.pipeline_create_ns, time.pipeline_create_ns);
  add(stats.descriptor_setup_ns, time.descriptor_setup_ns);
  add(stats.submit_wait_ns, time.command_submit_wait_ns);
  add(stats.readback_ns, time.readback_ns);
  add(stats.uploaded_bytes, transfer.host_to_device_bytes);
  add(stats.downloaded_bytes, transfer.device_to_host_bytes);
  add(stats.internal_roundtrip_bytes,
      transfer.internal_producer_consumer_roundtrip_bytes);
  add(stats.external_roundtrip_bytes,
      transfer.external_producer_consumer_roundtrip_bytes);
  add(stats.pipeline_compiles, allocations.pipeline_compile_count);
  add(stats.pipeline_cache_hits, allocations.pipeline_cache_hit_count);
  add(stats.pipeline_cache_evictions,
      allocations.pipeline_cache_eviction_count);
  add(stats.buffer_allocations, allocations.buffer_allocation_count);
  add(stats.buffer_reuses, allocations.buffer_reuse_hit_count);
  add(stats.descriptor_pool_creations,
      allocations.descriptor_pool_create_count);
  add(stats.descriptor_set_allocations,
      allocations.descriptor_set_allocate_count);
  add(stats.descriptor_reuses, allocations.descriptor_reuse_hit_count);
}

} // namespace rund::compute::detail
