#include "../internal.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace rund::measure::compute::route_matrix {
namespace {

void text(const std::string_view value) {
  std::putchar(',');
  for (const char ch : value) {
    switch (ch) {
    case ',':
    case '\n':
    case '\r':
      std::putchar(';');
      break;
    default:
      std::putchar(static_cast<unsigned char>(ch));
      break;
    }
  }
}

void number(const std::uint64_t value) {
  std::printf(",%llu", static_cast<unsigned long long>(value));
}

void flag(const bool value) { number(value ? 1u : 0u); }

void real(const double value) { std::printf(",%.6f", value); }

[[nodiscard]] std::string packet() {
  const char *const value = std::getenv("RUND_MEASURE_PACKET");
  return value == nullptr ? "0" : std::string{value};
}

} // namespace

std::string_view csv_header() noexcept {
  return "route_matrix,row_kind,packet,profile,evidence_scope,backend,api,"
         "driver,"
         "driver_details,case,backing,q,w,page_elements,elements,semantic_hi,"
         "semantic_lo,plan_hi,plan_lo,proof_hi,proof_lo,owner_id,route,"
         "expected_route,status,reason_code,reason,comparable,label,output_"
         "hash,"
         "expected_hash,version_before,version_after,cold_total_us,author_us,"
         "compile_us,prepare_us,seed_us,run_us,read_us,cpu_p25_us,cpu_p50_us,"
         "cpu_p75_us,cpu_p95_us,cpu_mad_us,backend_p25_us,backend_p50_us,"
         "backend_p75_us,backend_p95_us,backend_mad_us,backend_elements_per_s,"
         "command_submits,native_batches,queue_calls,dispatches,h2d_submits,"
         "d2h_submits,d2d_submits,uploaded_bytes,downloaded_bytes,page_in_"
         "count,"
         "page_out_count,page_in_bytes,page_out_bytes,backing_read_bytes,"
         "backing_write_bytes,stall_ns,overlap_ns,resident_peak,native_submit_"
         "count,"
         "epoch_native_submit_count,host_turns,host_callbacks,final_count,"
         "publication_count,gpu_kernel_ns,gpu_kernel_status,submit_span_ns,"
         "submit_span_status,one_final,allocation_free_60,output_ok,cpu_ok,"
         "owner_stable,backing_match,cpu_command_submits,cpu_dispatches,"
         "cpu_output_hash,route_owner_events,route_prepare_attempts,"
         "route_executed_runs,route_successful_finals,"
         "route_one_native_submit_runs,route_host_callbacks,route_publications,"
         "proof_valid,proof_staged_loop,proof_graph_resident,timing_authority,"
         "route_prepared_runs,route_final_callbacks,route_proof_valid_count,"
         "route_proof_identity_stable,route_proof_flags_stable,"
         "route_proof_identity_mismatches,route_proof_flag_mismatches,"
         "route_capability_observations,route_capability_consistent,"
         "route_capability_mismatches,route_proof_seen,"
         "route_per_run_lifecycle_consistent,route_lifecycle_complete,"
         "route_lifecycle_observations,route_lifecycle_mismatches,"
         "route_authority_accept_count,route_pipeline_terminal_count,"
         "route_device_generated_recurrence_runs,route_fixed_native_storage_"
         "runs,"
         "route_fixed_common_storage_runs,route_physical_ring_storage_runs,"
         "route_gpu_addressable_backing_runs,route_host_service_turns_zero_"
         "runs,"
         "route_host_epoch_callbacks_zero_runs,route_aggregate_terminal_once_"
         "runs,"
         "route_bounded_page_io_runs,route_cold_owner_runs,"
         "route_warm_reused_runs,route_max_warm_rearm_count,"
         "route_bounded_external_page_service_runs,route_proof_mode,"
         "route_proof_endpoint,route_window_seed_dispatches,"
         "route_window_compute_dispatches,route_window_internal_dispatches\n";
}

void print_header() { std::fputs(csv_header().data(), stdout); }

void print_row(const Row &row) {
  const auto &stats = row.stats;
  const auto &r = row.residency;
  const auto &route = row.route_evidence;
  std::fputs("route_matrix,raw", stdout);
  text(packet());
  text(profile_name(row.profile));
  text("unsealed_current_source");
  text(Name(row.backend));
  text(row.api);
  text(row.driver);
  text(row.driver_details);
  text(row.spec.family);
  text(row.spec.backing);
  number(row.spec.q);
  number(row.spec.window);
  number(PageElements);
  number(row.elements);
  number(row.semantic_hi);
  number(row.semantic_lo);
  number(row.plan_hi);
  number(row.plan_lo);
  number(row.proof_hi);
  number(row.proof_lo);
  number(row.owner_id);
  text(row.route);
  text(row.expected_route);
  text(row.status);
  number(row.reason_code);
  text(row.reason);
  flag(row.comparable);
  text(row.label);
  number(row.output_hash);
  number(row.expected_hash);
  number(row.version_before);
  number(row.version_after);
  real(row.cold.total_us);
  real(row.cold.author_us);
  real(row.cold.compile_us);
  real(row.cold.prepare_us);
  real(row.cold.seed_us);
  real(row.cold.run_us);
  real(row.cold.read_us);
  real(row.cpu_warm.p25_us);
  real(row.cpu_warm.p50_us);
  real(row.cpu_warm.p75_us);
  real(row.cpu_warm.p95_us);
  real(row.cpu_warm.mad_us);
  real(row.backend_warm.p25_us);
  real(row.backend_warm.p50_us);
  real(row.backend_warm.p75_us);
  real(row.backend_warm.p95_us);
  real(row.backend_warm.mad_us);
  real(row.backend_warm.elements_per_s);
  number(stats.command_submits);
  number(r.window_batch_count);
  number(r.window_queue_call_count);
  number(stats.dispatches);
  number(stats.transfer_submissions.host_to_device);
  number(stats.transfer_submissions.device_to_host);
  number(stats.transfer_submissions.device_to_device);
  number(stats.uploaded_bytes);
  number(stats.downloaded_bytes);
  number(r.page_in_count);
  number(r.page_out_count);
  number(r.page_in_bytes);
  number(r.page_out_bytes);
  number(r.backing_read_bytes);
  number(r.backing_write_bytes);
  number(r.stall_ns);
  number(r.overlap_ns);
  number(r.resident_frames_peak);
  number(route.native_submit_count);
  number(route.epoch_native_submit_count);
  number(route.host_service_turn_count);
  number(route.host_epoch_callback_count);
  number(route.final_callback_count);
  number(stats.publication.commit_count);
  number(row.gpu_kernel_ns);
  text(row.gpu_kernel_status);
  number(row.submit_span_ns);
  text(row.submit_span_status);
  flag(row.one_final);
  flag(row.allocation_free_60);
  flag(row.output_ok);
  flag(row.cpu_ok);
  flag(row.owner_stable);
  flag(row.backing_match);
  number(row.cpu_command_submits);
  number(row.cpu_dispatches);
  number(row.cpu_output_hash);
  number(route.owner_events);
  number(route.prepare_attempts);
  number(route.executed_runs);
  number(route.successful_finals);
  number(route.one_native_submit_runs);
  number(route.host_epoch_callback_count);
  number(route.backing_publication_count);
  flag(route.proof_valid);
  flag(route.proof_staged_loop);
  flag(route.proof_graph_resident);
  text(row.timing_authority);
  number(route.prepared_runs);
  number(route.final_callback_count);
  number(route.proof_valid_count);
  flag(route.proof_identity_stable);
  flag(route.proof_flags_stable);
  number(route.proof_identity_mismatches);
  number(route.proof_flag_mismatches);
  number(route.capability_observations);
  flag(route.capability_consistent);
  number(route.capability_mismatches);
  flag(route.proof_seen);
  flag(route.per_run_lifecycle_consistent);
  flag(route.lifecycle_complete);
  number(route.lifecycle_observations);
  number(route.lifecycle_mismatches);
  number(route.authority_accept_count);
  number(route.pipeline_terminal_count);
  number(route.device_generated_recurrence_runs);
  number(route.fixed_native_storage_runs);
  number(route.fixed_common_storage_runs);
  number(route.physical_ring_storage_runs);
  number(route.gpu_addressable_backing_runs);
  number(route.host_service_turns_zero_runs);
  number(route.host_epoch_callbacks_zero_runs);
  number(route.aggregate_terminal_once_runs);
  number(route.bounded_page_io_runs);
  number(route.cold_owner_runs);
  number(route.warm_reused_runs);
  number(route.max_warm_rearm_count);
  number(route.bounded_external_page_service_runs);
  number(route.proof_mode);
  number(route.proof_endpoint);
  number(route.window_seed_dispatches);
  number(route.window_compute_dispatches);
  number(route.window_internal_dispatches);
  std::putchar('\n');
}

} // namespace rund::measure::compute::route_matrix
