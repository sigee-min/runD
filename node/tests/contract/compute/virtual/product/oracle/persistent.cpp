#include "internal.hpp"

namespace rund_node_test_virtual::product {

bool ProductExecutionPersistentMatches(
    const ProductExecutionEvidence &evidence) noexcept {
  const auto &run = evidence.final_run;
  const auto &residency = run.pipeline.residency;
  return evidence.route_kind == RouteKind::Persistent &&
         residency.page_in_count == 0u &&
         residency.cache_hit_count == PageCount &&
         residency.page_out_count == PageCount &&
         residency.page_out_bytes == LogicalBytes &&
         residency.backing_read_bytes == CoherentFinalBackingReadBytes &&
         residency.backing_write_bytes == LogicalBytes &&
         residency.page_in_bytes == FinalPhysicalTransferBytes &&
         residency.stall_ns != 0u && run.command_submits == 1u &&
         run.command_inflight_peak == 1u &&
         run.transfer_submissions.host_to_device == 0u &&
         run.transfer_submissions.device_to_host == 0u &&
         run.transfer_submissions.device_to_device == 0u &&
         evidence.input_cohort.read_count == CoherentTotalBackingReadCalls &&
         evidence.input_cohort.read_bytes == CoherentTotalBackingReadBytes &&
         evidence.output_cohort.write_count == TotalPages &&
         evidence.output_cohort.write_bytes == TotalRuns * LogicalBytes;
}

} // namespace rund_node_test_virtual::product
