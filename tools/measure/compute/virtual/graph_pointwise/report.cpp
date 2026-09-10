#include "internal.hpp"
#include "report/local.hpp"

#include "../../suite/core.hpp"

#include <cstdio>

namespace rund::measure::compute::virtual_graph_pointwise {

bool ReportEnvironment(const Backend backend, const ::rund::compute::Device &,
                       const ::rund::compute::DeviceInfo &info) {
  if (info.backend != backend) {
    std::printf("environment,%s,backend_mismatch,%u,", Name(backend),
                static_cast<unsigned>(::rund::compute::Code::Invalid));
    PrintCsv("compute_device_info_backend_mismatch");
    std::putchar(',');
    PrintCsv(info.name);
    std::putchar(',');
    PrintCsv(info.driver);
    std::putchar(',');
    PrintCsv(info.driver_details);
    std::putchar('\n');
    return false;
  }
  std::printf("environment,%s,ok,%u,\"\",", Name(backend),
              static_cast<unsigned>(::rund::compute::Code::Ok));
  PrintCsv(info.name);
  std::putchar(',');
  PrintCsv(info.driver);
  std::putchar(',');
  PrintCsv(info.driver_details);
  std::putchar('\n');
  return true;
}

void PrintVirtualGraphPointwiseColumns() {
  std::fputs(
      "graph_pointwise,kind,source,backend,status,device,driver,driver_"
      "details,device_code,device_error,route_backend,elements,frame_"
      "elements,tail_elements,inputs,stages,pages,frame_capacity,batches,"
      "map_rows,graph_hi,graph_lo,expected_hash,input_digest_before,"
      "input_digest_after,facts_status_code,facts_status_reason,"
      "facts_status_error,failure_present,failure_phase,failure_check,"
      "failure_stage,failure_batch,failure_page,final_received,owner_present,"
      "may_write,owner_"
      "stable,owner_identity,control_identity,facts_graph_hi,facts_graph_"
      "lo,proof_hi,proof_lo,proof_digest,generation,nonce,input_count,"
      "stage_count,page_count,facts_tail_elements,facts_frame_capacity,"
      "facts_batch_count,facts_map_rows,workgroup_width,proof_valid,"
      "page_map_valid,native_submits,dispatches,finals,epoch_submits,"
      "generated_pages,forecasted_pages,promoted_pages,completed_pages,"
      "drained_pages,persisted_pages,host_turns,host_callbacks,gpu_read_"
      "bytes,gpu_write_bytes,uploaded_bytes,downloaded_bytes,backing_read_"
      "bytes,backing_write_bytes,output_hash,version_before,version_after,"
      "stats_command_submits,stats_dispatches,stats_final_dispatches,"
      "stats_page_in_count,stats_page_out_count,stats_page_in_bytes,"
      "stats_page_out_bytes,stats_epoch_count,stats_sampled_runs,"
      "stats_allocation_free_runs,route_handoff,route_accept,route_terminal,"
      "route_publication,route_prepare,route_prepared,route_executed,"
      "route_successful,route_cold_owner,route_warm_reused,route_max_rearm,"
      "route_hash_observations_first,route_hash_observations_last,"
      "route_hash_reuses_first,route_hash_reuses_last,route_hash_changes,"
      "route_hash_step_errors,packet,cold_us,p25_us,p50_us,p75_us,p95_us,"
      "mad_us,samples\n",
      stdout);
}

void Report(const Result &result) {
  WriteCsvRow(result, "semantic", result.cold, false);
  const Facts &timing = result.warm.status_observed ? result.warm : result.cold;
  WriteCsvRow(result, "timing", timing, true);
  if (!result.ok) {
    WriteDiagnostics(result);
  }
}

} // namespace rund::measure::compute::virtual_graph_pointwise

namespace rund::measure::compute {

void PrintVirtualGraphPointwiseColumns() {
  virtual_graph_pointwise::PrintVirtualGraphPointwiseColumns();
}

} // namespace rund::measure::compute
