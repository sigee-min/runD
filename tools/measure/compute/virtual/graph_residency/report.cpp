#include "internal.hpp"

#include "../../suite/core.hpp"

#include <cstdio>
#include <cstdlib>

namespace rund::measure::compute::virtual_graph_residency {
namespace {

void text(const std::string &value) { PrintCsv(value); }

void number(const std::uint64_t value) {
  std::printf(",%llu", static_cast<unsigned long long>(value));
}

void decimal(const double value) { std::printf(",%.9f", value); }

void blank(const std::size_t count) {
  for (std::size_t index = 0u; index < count; ++index) {
    std::putchar(',');
  }
}

void phase(const PhaseSample &sample) {
  decimal(sample.p25);
  decimal(sample.p50);
  decimal(sample.p75);
  decimal(sample.p95);
  decimal(sample.mad);
  number(static_cast<std::uint64_t>(sample.valid));
}

void actual(const Result &result, const Facts &facts) {
  number(facts.final_received);
  number(facts.owner_present);
  number(result.owner_stable);
  number(facts.proof_hi);
  number(facts.proof_lo);
  number(result.input_digest_before);
  number(result.input_digest_after);
  number(facts.residency.page_in_count);
  number(facts.residency.page_out_count);
  number(facts.residency.page_in_bytes);
  number(facts.residency.page_out_bytes);
  number(facts.residency.epoch_count);
  number(facts.uploaded_bytes);
  number(facts.downloaded_bytes);
  number(facts.internal_roundtrip_bytes);
  number(facts.external_roundtrip_bytes);
  number(facts.generated_pages);
  number(facts.forecasted_pages);
  number(facts.promoted_pages);
  number(facts.completed_pages);
  number(facts.drained_pages);
  number(facts.persisted_pages);
  number(facts.route.public_handoff_count);
  number(facts.route.authority_accept_count);
  number(facts.route.pipeline_terminal_count);
  number(facts.route.backing_publication_count);
  number(facts.route.prepare_attempts);
  number(facts.route.prepared_runs);
  number(facts.route.executed_runs);
  number(facts.route.successful_finals);
  number(facts.route.cold_owner_runs);
  number(facts.route.warm_reused_runs);
  number(facts.route.max_warm_rearm_count);
  number(facts.hash_observations);
  number(facts.hash_reuses);
  number(facts.route.first_output_hash_observation_count);
  number(facts.route.last_output_hash_observation_count);
  number(facts.route.first_output_hash_reuse_count);
  number(facts.route.last_output_hash_reuse_count);
  number(facts.route.hash_observation_changes);
  number(facts.route.hash_reuse_step_errors);
  number(result.conditioning.first_output_hash_observation_count);
  number(result.conditioning.last_output_hash_observation_count);
  number(result.conditioning.first_output_hash_reuse_count);
  number(result.conditioning.last_output_hash_reuse_count);
  const char *packet = std::getenv("RUND_MEASURE_PACKET");
  text(packet == nullptr ? "single" : packet);
}

void common(const Result &result, const char *kind) {
  const Facts &facts = result.cold;
  std::printf("graph_residency,%s,unsealed_current_source,%s,%u,", kind,
              Name(result.backend), static_cast<unsigned>(result.ok));
  text(result.device);
  std::putchar(',');
  text(result.driver);
  std::putchar(',');
  text(result.driver_details);
  number(result.device_code);
  std::putchar(',');
  text(result.device_error);
  std::putchar(',');
  PrintCsv(Name(facts.route.backend));
  std::putchar(',');
  number(Spec::PageCount);
  number(Spec::FrameElements);
  number(Spec::TailElements);
  number(Spec::StageCount);
  number(facts.resource_count);
  number(facts.owner_count);
  number(facts.endpoint_count);
  number(facts.proof_valid);
  number(facts.alias_reuse);
  number(facts.frame_capacity);
  number(facts.batch_count);
  number(result.topology_digest);
  number(result.workload_digest);
  number(facts.proof_digest);
  number(facts.generation);
  number(facts.nonce);
  number(facts.native_submits);
  number(facts.dispatches);
  number(facts.finals);
  number(facts.epoch_submits);
  number(facts.pages);
  number(facts.wavefront_steps);
  number(facts.gpu_read_bytes);
  number(facts.gpu_write_bytes);
  number(facts.host_turns);
  number(facts.host_callbacks);
  number(facts.backing_read_bytes);
  number(facts.backing_write_bytes);
  number(facts.output_hash);
  number(facts.version_before);
  number(facts.version_after);
  number(result.cold.route.successful_finals);
  number(result.cold.route.one_native_submit_runs);
  number(result.cold.route.host_epoch_callbacks_zero_runs);
  number(result.cold.route.host_service_turns_zero_runs);
  actual(result, facts);
  blank(31u);
  std::putchar('\n');
}

} // namespace

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

void PrintVirtualGraphResidencyColumns() {
  std::fputs(
      "graph_residency,kind,source,backend,status,device,driver,driver_details,"
      "device_code,device_error,route_backend,"
      "pages,frame_elements,tail_elements,stages,resources,internal_owners,"
      "endpoint_classes,proof_valid,alias_reuse,"
      "frame_capacity,batches,topology_digest,workload_digest,proof_digest,"
      "generation,nonce,native_submits,dispatches,finals,epoch_submits,"
      "pages_observed,wavefront_steps,gpu_read_bytes,gpu_write_bytes,"
      "host_turns,host_callbacks,backing_read_bytes,backing_write_bytes,"
      "output_hash,version_before,version_after,route_finals,route_submits,"
      "route_host_callbacks_zero,route_host_turns_zero,final_received,"
      "owner_present,owner_stable,proof_hi,proof_lo,input_digest_before,"
      "input_digest_after,"
      "stats_page_in_count,stats_page_out_count,stats_page_in_bytes,"
      "stats_page_out_bytes,stats_epoch_count,stats_uploaded_bytes,"
      "stats_downloaded_bytes,stats_internal_roundtrip,stats_external_"
      "roundtrip,"
      "generated_pages,forecasted_pages,promoted_pages,completed_pages,drained_"
      "pages,"
      "persisted_pages,route_handoff,route_accept,route_terminal,"
      "route_publication,route_prepare,route_prepared,route_executed,"
      "route_successful,route_cold_owner,route_warm_reused,route_max_rearm,"
      "hash_observations,hash_reuses,route_hash_observations_first,"
      "route_hash_observations_last,route_hash_reuses_first,"
      "route_hash_reuses_last,route_hash_observation_changes,"
      "route_hash_reuse_step_errors,conditioning_hash_observations_first,"
      "conditioning_hash_observations_last,conditioning_hash_reuses_first,"
      "conditioning_hash_reuses_last,packet,"
      "cold_us,p25_us,p50_us,p75_us,p95_us,mad_us,samples,"
      "pre_p25_us,pre_p50_us,pre_p75_us,pre_p95_us,pre_mad_us,pre_valid,"
      "queue_p25_us,queue_p50_us,queue_p75_us,queue_p95_us,queue_mad_us,"
      "queue_valid,finish_p25_us,finish_p50_us,finish_p75_us,finish_p95_us,"
      "finish_mad_us,finish_valid,post_p25_us,post_p50_us,post_p75_us,"
      "post_p95_us,post_mad_us,post_valid\n",
      stdout);
}

void Report(const Result &result) {
  common(result, "semantic");
  const Facts &facts = result.warm;
  std::printf("graph_residency,timing,unsealed_current_source,%s,%u,",
              Name(result.backend), static_cast<unsigned>(result.ok));
  text(result.device);
  std::putchar(',');
  text(result.driver);
  std::putchar(',');
  text(result.driver_details);
  number(result.device_code);
  std::putchar(',');
  text(result.device_error);
  std::putchar(',');
  PrintCsv(Name(facts.route.backend));
  std::putchar(',');
  number(Spec::PageCount);
  number(Spec::FrameElements);
  number(Spec::TailElements);
  number(Spec::StageCount);
  number(facts.resource_count);
  number(facts.owner_count);
  number(facts.endpoint_count);
  number(facts.proof_valid);
  number(facts.alias_reuse);
  number(facts.frame_capacity);
  number(facts.batch_count);
  number(result.topology_digest);
  number(result.workload_digest);
  number(facts.proof_digest);
  number(facts.generation);
  number(facts.nonce);
  number(facts.native_submits);
  number(facts.dispatches);
  number(facts.finals);
  number(facts.epoch_submits);
  number(facts.pages);
  number(facts.wavefront_steps);
  number(facts.gpu_read_bytes);
  number(facts.gpu_write_bytes);
  number(facts.host_turns);
  number(facts.host_callbacks);
  number(facts.backing_read_bytes);
  number(facts.backing_write_bytes);
  number(facts.output_hash);
  number(facts.version_before);
  number(result.warm.version_after);
  number(facts.route.successful_finals);
  number(facts.route.one_native_submit_runs);
  number(facts.route.host_epoch_callbacks_zero_runs);
  number(facts.route.host_service_turns_zero_runs);
  actual(result, facts);
  decimal(result.cold_us);
  decimal(result.timing.p25);
  decimal(result.timing.p50);
  decimal(result.timing.p75);
  decimal(result.timing.p95);
  decimal(result.timing.mad);
  number(SampleCount);
  phase(result.timing.phases.pre);
  phase(result.timing.phases.queue);
  phase(result.timing.phases.finish);
  phase(result.timing.phases.post);
  std::putchar('\n');
}

} // namespace rund::measure::compute::virtual_graph_residency

namespace rund::measure::compute {

void PrintVirtualGraphResidencyColumns() {
  virtual_graph_residency::PrintVirtualGraphResidencyColumns();
}

} // namespace rund::measure::compute
