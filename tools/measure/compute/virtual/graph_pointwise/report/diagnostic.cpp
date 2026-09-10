#include "../internal.hpp"
#include "local.hpp"

#include <cstdio>

namespace rund::measure::compute::virtual_graph_pointwise {
namespace {

void diagnostic_facts(const char *name, const Facts &facts) {
  const auto &s = facts.stats;
  const auto &r = facts.residency;
  std::fprintf(
      stderr,
      "graph_pointwise_diagnostic facts=%s ok=%u proof=%u page_map=%u "
      "quarantine=%u final=%u may_write=%u status=(%u,%u,%s) "
      "residency=(logical=%llu active=%llu page_bytes=%llu page_count=%llu "
      "frame=%llu epoch=%llu page_in=%llu page_out=%llu page_in_bytes=%llu "
      "page_out_bytes=%llu backing_read=%llu backing_write=%llu sampled=%u "
      "free=%u) pipeline_resource=%llu transfer=(h2d=%llu d2h=%llu d2d=%llu "
      "uploaded=%llu downloaded=%llu internal=%llu external=%llu) stats="
      "(submits=%llu dispatches=%llu finals=%llu)\n",
      name, static_cast<unsigned>(facts.ok),
      static_cast<unsigned>(facts.proof_valid),
      static_cast<unsigned>(facts.page_map_valid),
      static_cast<unsigned>(facts.quarantined),
      static_cast<unsigned>(facts.final_received),
      static_cast<unsigned>(facts.may_write),
      static_cast<unsigned>(facts.status_code),
      static_cast<unsigned>(facts.status_reason), facts.status_error.c_str(),
      static_cast<unsigned long long>(r.logical_bytes),
      static_cast<unsigned long long>(r.active_count),
      static_cast<unsigned long long>(r.page_bytes),
      static_cast<unsigned long long>(r.page_count),
      static_cast<unsigned long long>(r.frame_capacity),
      static_cast<unsigned long long>(r.epoch_count),
      static_cast<unsigned long long>(r.page_in_count),
      static_cast<unsigned long long>(r.page_out_count),
      static_cast<unsigned long long>(r.page_in_bytes),
      static_cast<unsigned long long>(r.page_out_bytes),
      static_cast<unsigned long long>(r.backing_read_bytes),
      static_cast<unsigned long long>(r.backing_write_bytes),
      static_cast<unsigned>(r.sampled_runs),
      static_cast<unsigned>(r.allocation_free_runs),
      static_cast<unsigned long long>(s.pipeline.resource_count),
      static_cast<unsigned long long>(s.transfer_submissions.host_to_device),
      static_cast<unsigned long long>(s.transfer_submissions.device_to_host),
      static_cast<unsigned long long>(s.transfer_submissions.device_to_device),
      static_cast<unsigned long long>(s.uploaded_bytes),
      static_cast<unsigned long long>(s.downloaded_bytes),
      static_cast<unsigned long long>(s.internal_roundtrip_bytes),
      static_cast<unsigned long long>(s.external_roundtrip_bytes),
      static_cast<unsigned long long>(s.command_submits),
      static_cast<unsigned long long>(s.dispatches),
      static_cast<unsigned long long>(s.final_dispatches));
  std::fprintf(stderr,
               "graph_pointwise_diagnostic facts=%s flags=(status_observed=%u "
               "cpu_reference=%u owner_present=%u graph_pointwise=%u "
               "output_match=%u stats_backend=%u)\n",
               name, static_cast<unsigned>(facts.status_observed),
               static_cast<unsigned>(facts.cpu_reference),
               static_cast<unsigned>(facts.owner_present),
               static_cast<unsigned>(facts.graph_pointwise),
               static_cast<unsigned>(facts.output_match),
               static_cast<unsigned>(s.backend));
}

void diagnostic_route(const char *name,
                      const virtual_residency::ProductRouteEvidence &route) {
  std::fprintf(
      stderr,
      "graph_pointwise_diagnostic route=%s backend=%u owner=%u production=%u "
      "events=%llu prepare=%llu prepared=%llu executed=%llu success=%llu "
      "cold=%llu warm=%llu rearm=%llu capability=(gpu=%llu public=%llu "
      "staging=%llu native=%llu fixed=%llu/%llu) coordinates=(%llu,%llu,%llu "
      "%llu) io=(read=%llu write=%llu overlap=%llu) lifecycle=(wait=%llu "
      "signal=%llu ack=%llu terminal=%llu final=%llu publication=%llu) "
      "completed_ns=%llu retained=%llu hash=(obs=%llu/%llu reuse=%llu/%llu "
      "changes=%llu errors=%llu)\n",
      name, static_cast<unsigned>(route.backend),
      static_cast<unsigned>(route.owner_stable),
      static_cast<unsigned>(route.production_slots),
      static_cast<unsigned long long>(route.owner_events),
      static_cast<unsigned long long>(route.prepare_attempts),
      static_cast<unsigned long long>(route.prepared_runs),
      static_cast<unsigned long long>(route.executed_runs),
      static_cast<unsigned long long>(route.successful_finals),
      static_cast<unsigned long long>(route.cold_owner_runs),
      static_cast<unsigned long long>(route.warm_reused_runs),
      static_cast<unsigned long long>(route.max_warm_rearm_count),
      static_cast<unsigned long long>(route.gpu_addressable_backing_runs),
      static_cast<unsigned long long>(
          route.public_gpu_addressable_backing_runs),
      static_cast<unsigned long long>(route.whole_run_staging_runs),
      static_cast<unsigned long long>(route.one_native_submit_runs),
      static_cast<unsigned long long>(route.fixed_native_storage_runs),
      static_cast<unsigned long long>(route.fixed_common_storage_runs),
      static_cast<unsigned long long>(route.coordinate_count),
      static_cast<unsigned long long>(route.accepted_coordinates),
      static_cast<unsigned long long>(route.gpu_completed_coordinates),
      static_cast<unsigned long long>(route.completed_prefix),
      static_cast<unsigned long long>(route.gpu_backing_read_bytes),
      static_cast<unsigned long long>(route.gpu_backing_write_bytes),
      static_cast<unsigned long long>(route.overlap_reused_bytes),
      static_cast<unsigned long long>(route.backing_wait_count),
      static_cast<unsigned long long>(route.backing_signal_count),
      static_cast<unsigned long long>(route.backing_acknowledgement_count),
      static_cast<unsigned long long>(route.pipeline_terminal_count),
      static_cast<unsigned long long>(route.final_callback_count),
      static_cast<unsigned long long>(route.backing_publication_count),
      static_cast<unsigned long long>(route.completed_ns),
      static_cast<unsigned long long>(route.retained_bytes),
      static_cast<unsigned long long>(
          route.first_output_hash_observation_count),
      static_cast<unsigned long long>(route.last_output_hash_observation_count),
      static_cast<unsigned long long>(route.first_output_hash_reuse_count),
      static_cast<unsigned long long>(route.last_output_hash_reuse_count),
      static_cast<unsigned long long>(route.hash_observation_changes),
      static_cast<unsigned long long>(route.hash_reuse_step_errors));
  std::fprintf(
      stderr,
      "graph_pointwise_diagnostic route=%s capability_extra=(preencoded=%llu "
      "device_generated=%llu external_service=%llu host_zero=%llu "
      "callback_zero=%llu aggregate=%llu bounded_page_io=%llu) pages="
      "(forecast=%llu promote=%llu drain=%llu persist=%llu) lifecycle_extra="
      "(native=%llu epoch=%llu payload=%llu host_turn=%llu backend_cb=%llu "
      "epoch_cb=%llu queue=%llu handoff=%llu authority=%llu) transient=%llu\n",
      name, static_cast<unsigned long long>(route.whole_run_preencoded_runs),
      static_cast<unsigned long long>(route.device_generated_recurrence_runs),
      static_cast<unsigned long long>(route.bounded_external_page_service_runs),
      static_cast<unsigned long long>(route.host_service_turns_zero_runs),
      static_cast<unsigned long long>(route.host_epoch_callbacks_zero_runs),
      static_cast<unsigned long long>(route.aggregate_terminal_once_runs),
      static_cast<unsigned long long>(route.bounded_page_io_runs),
      static_cast<unsigned long long>(route.forecasted_pages),
      static_cast<unsigned long long>(route.promoted_pages),
      static_cast<unsigned long long>(route.drained_pages),
      static_cast<unsigned long long>(route.persisted_pages),
      static_cast<unsigned long long>(route.native_submit_count),
      static_cast<unsigned long long>(route.epoch_native_submit_count),
      static_cast<unsigned long long>(route.payload_dispatch_count),
      static_cast<unsigned long long>(route.host_service_turn_count),
      static_cast<unsigned long long>(route.backend_epoch_callback_count),
      static_cast<unsigned long long>(route.host_epoch_callback_count),
      static_cast<unsigned long long>(route.queue_calls),
      static_cast<unsigned long long>(route.public_handoff_count),
      static_cast<unsigned long long>(route.authority_accept_count),
      static_cast<unsigned long long>(route.transient_bytes));
}

} // namespace

void WriteDiagnostics(const Result &result) {
  const OracleDiagnostic report = Diagnose(result);
  if (report.clean()) {
    return;
  }
  std::fprintf(
      stderr,
      "graph_pointwise_diagnostic mask=%llu first_leaf=%u "
      "first_leaf_name=%s backend=%u result_ok=%u samples=%llu "
      "timing_complete=%u\n",
      static_cast<unsigned long long>(report.failed_mask), report.first_leaf,
      report.first_leaf_name == nullptr ? "unknown" : report.first_leaf_name,
      static_cast<unsigned>(result.backend), static_cast<unsigned>(result.ok),
      static_cast<unsigned long long>(result.completed_samples),
      static_cast<unsigned>(result.timing_complete));
  const std::uint64_t output_bytes = Spec::ElementCount * sizeof(std::uint64_t);
  const std::uint64_t resources = Spec::InputCount + Spec::StageCount;
  const std::uint64_t frame_bytes = Spec::FrameElements * sizeof(std::uint64_t);
  std::fprintf(stderr,
               "graph_pointwise_diagnostic expected=(q=%llu k=%llu stages=%llu "
               "logical=%llu page=%llu input=%llu output=%llu runs=%llu)\n",
               static_cast<unsigned long long>(Spec::PageCount),
               static_cast<unsigned long long>(Spec::FrameCapacity),
               static_cast<unsigned long long>(Spec::StageCount),
               static_cast<unsigned long long>(resources * output_bytes),
               static_cast<unsigned long long>(resources * frame_bytes),
               static_cast<unsigned long long>(Spec::InputCount * output_bytes),
               static_cast<unsigned long long>(output_bytes),
               static_cast<unsigned long long>(SampleCount));
  diagnostic_facts("cold", result.cold);
  diagnostic_facts("warm", result.warm);
  diagnostic_route("conditioning", result.conditioning);
  diagnostic_route("cold", result.cold.route);
  diagnostic_route("warm", result.warm.route);
}

} // namespace rund::measure::compute::virtual_graph_pointwise
