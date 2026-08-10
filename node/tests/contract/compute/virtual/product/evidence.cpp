#include "evidence.hpp"

namespace rund_node_test_virtual::product {

bool ProductExecutionMatches(
    const ProductExecutionEvidence &evidence) noexcept {
  const bool cpu = evidence.final_run.backend == rund::compute::Backend::Cpu;
  const bool accelerator =
      evidence.final_run.backend == rund::compute::Backend::Metal ||
      evidence.final_run.backend == rund::compute::Backend::Vulkan;
  const auto &residency = evidence.final_run.pipeline.residency;
  const auto &submissions = evidence.final_run.transfer_submissions;
  return (cpu || accelerator) && residency.logical_bytes == LogicalBytes * 2u &&
         residency.active_count == LogicalElements &&
         residency.logical_bytes > SlotBytes &&
         residency.page_bytes == ResidencyPageBytes &&
         residency.page_count == PageCount &&
         residency.slot_capacity == SlotCapacity &&
         residency.active_slots_peak == SlotCapacity &&
         residency.wave_count == WaveCount &&
         residency.load_count == PageCount &&
         residency.writeback_count == PageCount &&
         residency.backing_read_bytes == FinalBackingBytes &&
         residency.backing_write_bytes == FinalBackingBytes &&
         residency.samples_allocation_free(WarmRuns) &&
         residency.failed_page ==
             rund::compute::ResidencyStats::no_failed_page &&
         evidence.final_run.command_submits == (cpu ? 0u : WaveCount) &&
         evidence.final_run.uploaded_bytes == FinalPhysicalTransferBytes &&
         evidence.final_run.downloaded_bytes == FinalPhysicalTransferBytes &&
         submissions.host_to_device == 0u && submissions.device_to_host == 0u &&
         submissions.device_to_device == 0u &&
         evidence.final_run.pipeline_compiles == 0u &&
         evidence.final_run.buffer_allocations == 0u &&
         evidence.final_run.descriptor_pool_creations == 0u &&
         evidence.final_run.descriptor_set_allocations == 0u &&
         evidence.final_run.pipeline_cache_evictions == 0u &&
         evidence.final_run.command_capacity_rejections == 0u &&
         evidence.final_run.pipeline.claim_ns == 0u &&
         evidence.final_run.pipeline.claim_conflict_count == 0u &&
         evidence.input_cohort.read_count == TotalBackingCalls &&
         evidence.input_cohort.read_bytes == TotalBackingTransferBytes &&
         evidence.input_cohort.write_count == 0u &&
         evidence.input_cohort.write_bytes == 0u &&
         evidence.input_cohort.observation_count == 0u &&
         evidence.input_cohort.observation_bytes == 0u &&
         evidence.input_cohort.read_failure_count == 0u &&
         evidence.input_cohort.write_failure_count == 0u &&
         evidence.input_cohort.partial_write_bytes == 0u &&
         evidence.output_cohort.read_count == 0u &&
         evidence.output_cohort.read_bytes == 0u &&
         evidence.output_cohort.write_count == TotalBackingCalls &&
         evidence.output_cohort.write_bytes == TotalBackingTransferBytes &&
         evidence.output_cohort.observation_count == 1u &&
         evidence.output_cohort.observation_bytes == LogicalBytes &&
         evidence.output_cohort.read_failure_count == 0u &&
         evidence.output_cohort.write_failure_count == 0u &&
         evidence.output_cohort.partial_write_bytes == 0u &&
         evidence.warm_host_allocations == 0u &&
         evidence.final_run.output_hash == GoldenHash &&
         evidence.observed_hash == GoldenHash && evidence.same_capacity &&
         evidence.tail_poisoned && evidence.profile_matches;
}

} // namespace rund_node_test_virtual::product
