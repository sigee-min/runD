#include "internal.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::virtual_graph_residency {
namespace {

[[nodiscard]] bool add_exact(const std::uint64_t left,
                             const std::uint64_t right,
                             const std::uint64_t expected) noexcept {
  return left <= std::numeric_limits<std::uint64_t>::max() - right &&
         left + right == expected;
}

[[nodiscard]] bool route_ok(const virtual_residency::ProductRouteEvidence &e,
                            const Backend backend, const std::uint64_t runs,
                            const std::uint64_t stage_count) noexcept {
  const std::uint64_t coordinates = Spec::PageCount * runs;
  const std::uint64_t bytes = Spec::ElementCount * sizeof(std::uint64_t) * runs;
  const bool owner_lifetime =
      (runs == 1u && e.cold_owner_runs == 1u && e.warm_reused_runs == 0u &&
       e.max_warm_rearm_count == 0u) ||
      (runs == SampleCount && e.cold_owner_runs == 0u &&
       e.warm_reused_runs == runs && e.max_warm_rearm_count >= runs);
  return e.backend == backend && e.owner_stable &&
         e.owner_events == runs * 2u && e.production_slots &&
         e.prepare_attempts == runs && e.prepared_runs == runs &&
         e.executed_runs == runs && e.successful_finals == runs &&
         owner_lifetime && e.whole_run_preencoded_runs == 0u &&
         e.device_generated_recurrence_runs == runs &&
         e.fixed_native_storage_runs == runs &&
         e.fixed_common_storage_runs == runs &&
         e.gpu_addressable_backing_runs == runs &&
         e.public_gpu_addressable_backing_runs == runs &&
         e.whole_run_staging_runs == 0u &&
         e.bounded_external_page_service_runs == 0u &&
         e.one_native_submit_runs == runs &&
         e.host_service_turns_zero_runs == runs &&
         e.host_epoch_callbacks_zero_runs == runs &&
         e.aggregate_terminal_once_runs == runs &&
         e.bounded_page_io_runs == runs && e.coordinate_count == coordinates &&
         e.accepted_coordinates == coordinates &&
         e.gpu_completed_coordinates == coordinates &&
         e.completed_prefix == coordinates &&
         e.forecasted_pages == coordinates && e.promoted_pages == coordinates &&
         e.drained_pages == coordinates && e.persisted_pages == coordinates &&
         e.overlap_reused_bytes == 0u && e.gpu_backing_read_bytes == 0u &&
         e.gpu_backing_write_bytes == bytes && e.native_submit_count == runs &&
         e.epoch_native_submit_count == 0u &&
         e.payload_dispatch_count == runs && e.host_service_turn_count == 0u &&
         e.backend_epoch_callback_count == 0u &&
         e.host_epoch_callback_count == 0u && e.backing_wait_count == 0u &&
         e.backing_signal_count == 0u &&
         e.backing_acknowledgement_count == 0u &&
         e.final_callback_count == runs && e.queue_calls == runs &&
         e.public_handoff_count == runs && e.authority_accept_count == runs &&
         e.pipeline_terminal_count == runs * stage_count &&
         e.backing_publication_count == runs && e.completed_ns != 0u &&
         e.retained_bytes != 0u;
}

[[nodiscard]] bool
hash_cache_ok(const virtual_residency::ProductRouteEvidence &e,
              const std::uint64_t runs) noexcept {
  // This is the existing ProductRouteValidation hash-cache law, applied to
  // the same observer counters; it does not create another cache authority.
  if (runs == 1u) {
    return e.first_output_hash_observation_count == 1u &&
           e.last_output_hash_observation_count == 1u &&
           e.first_output_hash_reuse_count == 0u &&
           e.last_output_hash_reuse_count == 0u &&
           e.hash_observation_changes == 0u && e.hash_reuse_step_errors == 0u;
  }
  return runs == SampleCount && e.first_output_hash_observation_count != 0u &&
         e.first_output_hash_observation_count ==
             e.last_output_hash_observation_count &&
         e.first_output_hash_reuse_count != 0u &&
         e.last_output_hash_reuse_count >= e.first_output_hash_reuse_count &&
         e.last_output_hash_reuse_count - e.first_output_hash_reuse_count +
                 1u ==
             runs &&
         e.hash_observation_changes == 0u && e.hash_reuse_step_errors == 0u;
}

[[nodiscard]] bool hash_phases(const Result &result) noexcept {
  const auto &cold = result.cold.route;
  const auto &conditioning = result.conditioning;
  const auto &warm = result.warm.route;
  const bool conditioning_route =
      conditioning.backend == result.backend && conditioning.owner_stable &&
      conditioning.owner_events == 2u && conditioning.prepare_attempts == 1u &&
      conditioning.prepared_runs == 1u && conditioning.executed_runs == 1u &&
      conditioning.successful_finals == 1u;
  return conditioning_route && cold.first_output_hash_observation_count == 1u &&
         cold.last_output_hash_observation_count == 1u &&
         cold.first_output_hash_reuse_count == 0u &&
         cold.last_output_hash_reuse_count == 0u &&
         conditioning.first_output_hash_observation_count == 1u &&
         conditioning.last_output_hash_observation_count == 1u &&
         conditioning.first_output_hash_reuse_count == 1u &&
         conditioning.last_output_hash_reuse_count == 1u &&
         warm.first_output_hash_observation_count == 1u &&
         warm.last_output_hash_observation_count == 1u &&
         warm.first_output_hash_reuse_count == 2u &&
         warm.last_output_hash_reuse_count == SampleCount + 1u &&
         cold.hash_observation_changes == 0u &&
         cold.hash_reuse_step_errors == 0u &&
         conditioning.hash_observation_changes == 0u &&
         conditioning.hash_reuse_step_errors == 0u &&
         warm.hash_observation_changes == 0u &&
         warm.hash_reuse_step_errors == 0u;
}

[[nodiscard]] bool residency_ok(const Facts &facts) noexcept {
  const auto &r = facts.residency;
  const std::uint64_t output_bytes = Spec::ElementCount * sizeof(std::uint64_t);
  return facts.stats.command_submits == 1u && facts.stats.dispatches == 1u &&
         facts.stats.final_dispatches == 1u &&
         facts.stats.uploaded_bytes == 0u &&
         facts.stats.downloaded_bytes == 0u &&
         facts.stats.internal_roundtrip_bytes == 0u &&
         facts.stats.external_roundtrip_bytes == 0u &&
         r.page_count == Spec::PageCount && r.frame_capacity == 2u &&
         r.epoch_count == Spec::BatchCount &&
         r.page_in_count == Spec::InputCount * Spec::PageCount &&
         r.page_in_bytes == 0u && r.page_out_bytes == output_bytes &&
         r.backing_read_bytes == 0u && r.backing_write_bytes == 0u;
}

[[nodiscard]] bool native_ok(const Facts &facts,
                             const Backend backend) noexcept {
  return facts.ok && facts.final_received && facts.owner_present &&
         facts.stats.backend == backend && facts.graph_resident &&
         facts.proof_valid && facts.alias_reuse &&
         facts.resource_count == Spec::ResourceCount &&
         facts.owner_count == Spec::InternalOwnerCount &&
         facts.endpoint_count == Spec::InputCount + 1u &&
         facts.stage_count == Spec::StageCount && facts.frame_capacity == 2u &&
         facts.batch_count == Spec::BatchCount && facts.native_submits == 1u &&
         facts.dispatches == 1u && facts.finals == 1u &&
         facts.epoch_submits == 0u && facts.pages == Spec::PageCount &&
         facts.wavefront_steps == Spec::StageCount * Spec::BatchCount &&
         facts.gpu_read_bytes == 0u &&
         facts.gpu_write_bytes == Spec::ElementCount * sizeof(std::uint64_t) &&
         facts.generated_pages == Spec::PageCount &&
         facts.forecasted_pages == Spec::PageCount &&
         facts.promoted_pages == Spec::PageCount &&
         facts.completed_pages == Spec::PageCount &&
         facts.drained_pages == Spec::PageCount &&
         facts.persisted_pages == Spec::PageCount && facts.host_turns == 0u &&
         facts.host_callbacks == 0u && facts.backing_read_bytes == 0u &&
         facts.backing_write_bytes == 0u && !facts.quarantined &&
         facts.may_write &&
         facts.hash_observations ==
             facts.route.last_output_hash_observation_count &&
         facts.hash_reuses == facts.route.last_output_hash_reuse_count &&
         residency_ok(facts);
}

} // namespace

[[nodiscard]] bool Validate(const Result &result) noexcept {
  if (!std::isfinite(result.cold_us) || result.cold_us <= 0.0 ||
      !timing_summary_matches(result.timing) ||
      result.input_digest_before == 0u ||
      result.input_digest_before != Spec::input_digest() ||
      result.input_digest_after != result.input_digest_before ||
      result.cold.output_hash != Spec::expected_hash() ||
      result.warm.output_hash != Spec::expected_hash() ||
      !result.cold.output_match || !result.warm.output_match ||
      !add_exact(result.cold.version_before, 1u, result.cold.version_after) ||
      !add_exact(result.warm.version_before, SampleCount,
                 result.warm.version_after) ||
      !result.device_valid) {
    return false;
  }
  if (result.backend == Backend::Cpu) {
    return !result.cold.graph_resident && !result.warm.graph_resident &&
           !result.cold.owner_present && !result.warm.owner_present &&
           result.cold.stats.backend == Backend::Cpu &&
           result.warm.stats.backend == Backend::Cpu &&
           result.cold.stats.command_submits == 0u &&
           result.warm.stats.command_submits == 0u &&
           result.cold.stats.uploaded_bytes == 0u &&
           result.warm.stats.uploaded_bytes == 0u &&
           result.cold.stats.downloaded_bytes == 0u &&
           result.warm.stats.downloaded_bytes == 0u &&
           !result.cold.route.production_slots &&
           !result.warm.route.production_slots;
  }
  if (result.backend != Backend::Metal && result.backend != Backend::Vulkan) {
    return false;
  }
  const Facts &cold = result.cold;
  const Facts &warm = result.warm;
  return native_ok(cold, result.backend) && native_ok(warm, result.backend) &&
         route_ok(cold.route, result.backend, 1u, cold.stage_count) &&
         route_ok(warm.route, result.backend, SampleCount, warm.stage_count) &&
         hash_cache_ok(cold.route, 1u) &&
         hash_cache_ok(warm.route, SampleCount) && hash_phases(result) &&
         phase_timing_available(result.timing.phases) &&
         result.owner_stable && cold.proof_digest == warm.proof_digest &&
         cold.proof_hi == warm.proof_hi && cold.proof_lo == warm.proof_lo &&
         cold.generation != 0u && warm.generation > cold.generation &&
         cold.nonce != 0u && warm.nonce != 0u && warm.nonce != cold.nonce;
}

} // namespace rund::measure::compute::virtual_graph_residency
