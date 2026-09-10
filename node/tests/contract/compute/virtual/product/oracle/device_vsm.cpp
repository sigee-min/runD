#include "internal.hpp"

namespace rund_node_test_virtual::product {

bool ProductExecutionDeviceVsmMatches(
    const ProductExecutionEvidence &evidence) noexcept {
  const auto &run = evidence.final_run;
  const auto &residency = run.pipeline.residency;
  const auto &submissions = run.transfer_submissions;
  return evidence.route_kind == RouteKind::DeviceVsm &&
         residency.page_in_count == PageCount &&
         residency.cache_hit_count == 0u && residency.eviction_count == 0u &&
         residency.prefetch_count == 0u && residency.late_page_count == 0u &&
         residency.page_out_count == PageCount &&
         residency.page_in_bytes == LogicalBytes &&
         residency.page_out_bytes == LogicalBytes &&
         residency.backing_read_bytes == LogicalBytes &&
         residency.backing_write_bytes == LogicalBytes &&
         residency.stall_ns == 0u && run.command_submits == 1u &&
         run.command_inflight_peak == 1u && run.uploaded_bytes == 0u &&
         run.downloaded_bytes == 0u && submissions.host_to_device == 0u &&
         submissions.device_to_host == 0u &&
         submissions.device_to_device == 0u &&
         evidence.input_cohort.read_count == TotalPages &&
         evidence.input_cohort.read_bytes == TotalRuns * LogicalBytes &&
         evidence.output_cohort.write_count == TotalPages &&
         evidence.output_cohort.write_bytes == TotalRuns * LogicalBytes;
}

} // namespace rund_node_test_virtual::product
