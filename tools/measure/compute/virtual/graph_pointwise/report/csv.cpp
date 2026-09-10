#include "../internal.hpp"
#include "local.hpp"

#include "../../../suite/core.hpp"

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace rund::measure::compute::virtual_graph_pointwise {
namespace {

void text(const std::string_view value) { PrintCsv(value); }

void number(const std::uint64_t value) {
  std::printf(",%llu", static_cast<unsigned long long>(value));
}

void plain_number(const std::uint64_t value) {
  std::printf("%llu", static_cast<unsigned long long>(value));
}

void decimal(const double value) { std::printf(",%.9f", value); }

void facts_csv(const Result &result, const Facts &facts) {
  number(static_cast<std::uint64_t>(facts.status_code));
  number(static_cast<std::uint64_t>(facts.status_reason));
  std::putchar(',');
  text(facts.status_error);
  number(facts.failure_present);
  number(facts.failure_phase);
  number(facts.failure_check);
  number(facts.failure_stage);
  number(facts.failure_batch);
  number(facts.failure_page);
  number(facts.final_received);
  number(facts.owner_present);
  number(facts.may_write);
  number(result.owner_stable);
  number(facts.owner_identity);
  number(facts.control_identity);
  number(facts.graph_hi);
  number(facts.graph_lo);
  number(facts.proof_hi);
  number(facts.proof_lo);
  number(facts.proof_digest);
  number(facts.generation);
  number(facts.nonce);
  number(facts.input_count);
  number(facts.stage_count);
  number(facts.page_count);
  number(facts.tail_elements);
  number(facts.frame_capacity);
  number(facts.batch_count);
  number(facts.map_rows);
  number(facts.workgroup_width);
  number(facts.proof_valid);
  number(facts.page_map_valid);
  number(facts.native_submits);
  number(facts.dispatches);
  number(facts.finals);
  number(facts.epoch_submits);
  number(facts.generated_pages);
  number(facts.forecasted_pages);
  number(facts.promoted_pages);
  number(facts.completed_pages);
  number(facts.drained_pages);
  number(facts.persisted_pages);
  number(facts.host_turns);
  number(facts.host_callbacks);
  number(facts.gpu_read_bytes);
  number(facts.gpu_write_bytes);
  number(facts.uploaded_bytes);
  number(facts.downloaded_bytes);
  number(facts.backing_read_bytes);
  number(facts.backing_write_bytes);
  number(facts.output_hash);
  number(facts.version_before);
  number(facts.version_after);
  number(facts.stats.command_submits);
  number(facts.stats.dispatches);
  number(facts.stats.final_dispatches);
  number(facts.residency.page_in_count);
  number(facts.residency.page_out_count);
  number(facts.residency.page_in_bytes);
  number(facts.residency.page_out_bytes);
  number(facts.residency.epoch_count);
  number(facts.residency.sampled_runs);
  number(facts.residency.allocation_free_runs);
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
  number(facts.route.first_output_hash_observation_count);
  number(facts.route.last_output_hash_observation_count);
  number(facts.route.first_output_hash_reuse_count);
  number(facts.route.last_output_hash_reuse_count);
  number(facts.route.hash_observation_changes);
  std::putchar(',');
  plain_number(facts.route.hash_reuse_step_errors);
  std::putchar(',');
  const char *packet = std::getenv("RUND_MEASURE_PACKET");
  text(packet == nullptr ? "single" : packet);
}

} // namespace

void WriteCsvRow(const Result &result, const char *kind, const Facts &facts,
                 const bool timing) {
  std::printf("graph_pointwise,%s,unsealed_current_source,%s,%u,", kind,
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
  number(Spec::ElementCount);
  number(Spec::FrameElements);
  number(Spec::TailElements);
  number(Spec::InputCount);
  number(Spec::StageCount);
  number(Spec::PageCount);
  number(Spec::FrameCapacity);
  number(Spec::BatchCount);
  number(Spec::MapRows);
  number(result.graph_hi);
  number(result.graph_lo);
  number(result.expected_hash);
  number(result.input_digest_before);
  number(result.input_digest_after);
  facts_csv(result, facts);
  if (timing) {
    decimal(result.cold_us);
    if (result.timing_complete) {
      decimal(result.timing.p25);
      decimal(result.timing.p50);
      decimal(result.timing.p75);
      decimal(result.timing.p95);
      decimal(result.timing.mad);
    } else {
      std::fputs(",,,,,", stdout);
    }
    number(result.completed_samples);
  } else {
    std::fputs(",,,,,,,", stdout);
  }
  std::putchar('\n');
}

} // namespace rund::measure::compute::virtual_graph_pointwise
