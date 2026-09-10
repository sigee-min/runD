#include "internal.hpp"

namespace rund_node_test_virtual::product {

bool ProductExecutionRollingMatches(
    const ProductExecutionEvidence &evidence) noexcept {
  using rund::compute::Backend;
  const auto &run = evidence.final_run;
  const auto &residency = run.pipeline.residency;
  const auto &submissions = run.transfer_submissions;
  const bool cpu = evidence.route_kind == RouteKind::CpuRolling;
  const bool accelerator = evidence.route_kind == RouteKind::AccelRolling;
  const bool cache_supplied = accelerator && residency.backing_read_bytes == 0u;
  const bool zero_upload =
      run.uploaded_bytes == 0u && submissions.host_to_device == 0u;
  const bool zero_download =
      run.downloaded_bytes == 0u && submissions.device_to_host == 0u;
  const bool physical_upload =
      run.backend == Backend::Vulkan &&
      run.uploaded_bytes == WarmPageIns * ElementPageBytes &&
      submissions.host_to_device == WarmPageIns;
  const bool physical_download =
      run.backend == Backend::Vulkan &&
      run.downloaded_bytes == FinalPhysicalTransferBytes &&
      submissions.device_to_host == EpochCount;
  const std::size_t final_read_bytes =
      cache_supplied ? CoherentFinalBackingReadBytes : FinalBackingReadBytes;
  const std::size_t total_read_calls =
      cache_supplied ? CoherentTotalBackingReadCalls : TotalBackingReadCalls;
  const std::size_t total_read_bytes =
      cache_supplied ? CoherentTotalBackingReadBytes : TotalBackingReadBytes;
  return (cpu || accelerator) && residency.page_in_count == WarmPageIns &&
         residency.cache_hit_count == WarmCacheHits &&
         residency.eviction_count == WarmPageIns &&
         residency.prefetch_count + residency.late_page_count +
                 (cache_supplied ? WarmPageIns : 0u) ==
             residency.page_in_count &&
         residency.page_out_count == PageCount &&
         residency.page_in_bytes ==
             (cpu ? final_read_bytes : WarmPageIns * ElementPageBytes) &&
         residency.page_out_bytes == FinalBackingWriteBytes &&
         residency.backing_read_bytes == final_read_bytes &&
         residency.backing_write_bytes == FinalBackingWriteBytes &&
         residency.stall_ns != 0u &&
         run.command_submits == (cpu ? 0u : EpochCount) &&
         (cpu                             ? run.command_inflight_peak == 0u
          : run.backend == Backend::Metal ? run.command_inflight_peak >= 2u
                                          : true) &&
         (cache_supplied ? zero_upload
          : cpu          ? zero_upload
                         : physical_upload) &&
         (run.backend == Backend::Vulkan ? zero_download || physical_download
                                         : zero_download) &&
         submissions.device_to_device == 0u &&
         evidence.input_cohort.read_count == total_read_calls &&
         evidence.input_cohort.read_bytes == total_read_bytes &&
         evidence.output_cohort.write_count == TotalBackingWriteCalls &&
         evidence.output_cohort.write_bytes == TotalBackingWriteBytes &&
         // MoltenVK queue submission records are opaque driver-private work.
         (run.backend == Backend::Vulkan ||
          evidence.warm_host_allocations == 0u);
}

} // namespace rund_node_test_virtual::product
