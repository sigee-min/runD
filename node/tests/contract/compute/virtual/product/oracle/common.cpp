#include "internal.hpp"

#include <algorithm>

namespace rund_node_test_virtual::product {

bool ProductExecutionCommonMatches(
    const ProductExecutionEvidence &evidence) noexcept {
  using rund::compute::Backend;
  using rund::compute::ResidencyStats;
  const auto &run = evidence.final_run;
  const auto &residency = run.pipeline.residency;
  const bool supported = run.backend == Backend::Cpu ||
                         run.backend == Backend::Metal ||
                         run.backend == Backend::Vulkan;
  return supported && residency.logical_bytes == LogicalBytes * 2u &&
         residency.active_count == LogicalElements &&
         residency.logical_bytes > FrameBytes &&
         residency.page_bytes == ResidencyPageBytes &&
         residency.page_count == PageCount &&
         residency.frame_capacity == FrameCapacity &&
         residency.resident_frames_peak ==
             std::min(PageCount, FrameCapacity * 2u) &&
         residency.epoch_count == EpochCount &&
         residency.samples_allocation_free(WarmRuns) &&
         residency.failed_page == ResidencyStats::no_failed_page &&
         run.pipeline_compiles == 0u && run.buffer_allocations == 0u &&
         run.descriptor_pool_creations == 0u &&
         run.descriptor_set_allocations == 0u &&
         run.pipeline_cache_evictions == 0u &&
         run.command_capacity_rejections == 0u && run.pipeline.claim_ns == 0u &&
         run.pipeline.claim_conflict_count == 0u &&
         evidence.input_cohort.write_count == 0u &&
         evidence.input_cohort.write_bytes == 0u &&
         evidence.input_cohort.observation_count == 0u &&
         evidence.input_cohort.observation_bytes == 0u &&
         evidence.input_cohort.read_failure_count == 0u &&
         evidence.input_cohort.write_failure_count == 0u &&
         evidence.input_cohort.partial_write_bytes == 0u &&
         evidence.output_cohort.read_count == 0u &&
         evidence.output_cohort.read_bytes == 0u &&
         evidence.output_cohort.observation_count == 1u &&
         evidence.output_cohort.observation_bytes == LogicalBytes &&
         evidence.output_cohort.read_failure_count == 0u &&
         evidence.output_cohort.write_failure_count == 0u &&
         evidence.output_cohort.partial_write_bytes == 0u &&
         run.output_hash == GoldenHash &&
         evidence.observed_hash == GoldenHash && evidence.golden_matches &&
         evidence.same_capacity && evidence.tail_poisoned &&
         evidence.profile_matches;
}

} // namespace rund_node_test_virtual::product
