#include "internal.hpp"

namespace rund::measure::compute::virtual_residency {

bool exact_zero(const ProductRouteEvidence &evidence) noexcept {
  ProductRouteEvidence zero{};
  zero.backend = evidence.backend;
  zero.production_slots = evidence.production_slots;
  zero.owner_stable = evidence.owner_stable;
  return evidence.prepare_attempts == zero.prepare_attempts &&
         evidence.prepared_runs == zero.prepared_runs &&
         evidence.executed_runs == zero.executed_runs &&
         evidence.successful_finals == zero.successful_finals &&
         evidence.cold_owner_runs == 0u && evidence.warm_reused_runs == 0u &&
         evidence.max_warm_rearm_count == 0u &&
         evidence.whole_run_preencoded_runs == 0u &&
         evidence.device_generated_recurrence_runs == 0u &&
         evidence.fixed_native_storage_runs == 0u &&
         evidence.fixed_common_storage_runs == 0u &&
         evidence.gpu_addressable_backing_runs == 0u &&
         evidence.public_gpu_addressable_backing_runs == 0u &&
         evidence.whole_run_staging_runs == 0u &&
         evidence.bounded_external_page_service_runs == 0u &&
         evidence.one_native_submit_runs == 0u &&
         evidence.host_service_turns_zero_runs == 0u &&
         evidence.host_epoch_callbacks_zero_runs == 0u &&
         evidence.aggregate_terminal_once_runs == 0u &&
         evidence.bounded_page_io_runs == 0u &&
         evidence.coordinate_count == 0u &&
         evidence.accepted_coordinates == 0u &&
         evidence.gpu_completed_coordinates == 0u &&
         evidence.completed_prefix == 0u && evidence.forecasted_pages == 0u &&
         evidence.promoted_pages == 0u && evidence.drained_pages == 0u &&
         evidence.persisted_pages == 0u &&
         evidence.overlap_reused_bytes == 0u &&
         evidence.gpu_backing_read_bytes == 0u &&
         evidence.gpu_backing_write_bytes == 0u &&
         evidence.native_submit_count == 0u &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.payload_dispatch_count == 0u &&
         evidence.host_service_turn_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.backing_wait_count == 0u &&
         evidence.backing_signal_count == 0u &&
         evidence.backing_acknowledgement_count == 0u &&
         evidence.final_callback_count == 0u && evidence.queue_calls == 0u &&
         evidence.public_handoff_count == 0u &&
         evidence.authority_accept_count == 0u &&
         evidence.pipeline_terminal_count == 0u &&
         evidence.backing_publication_count == 0u &&
         evidence.first_output_hash_observation_count == 0u &&
         evidence.last_output_hash_observation_count == 0u &&
         evidence.first_output_hash_reuse_count == 0u &&
         evidence.last_output_hash_reuse_count == 0u &&
         evidence.owner_events == 0u &&
         evidence.hash_observation_changes == 0u &&
         evidence.hash_reuse_step_errors == 0u && evidence.completed_ns == 0u &&
         evidence.retained_bytes == 0u && evidence.transient_bytes == 0u;
}

namespace {

[[nodiscard]] bool
exact_output_hash_cache(const ProductRouteEvidence &evidence,
                        const std::uint64_t expected_runs,
                        const bool resident_backing) noexcept {
  if (!resident_backing) {
    return evidence.first_output_hash_observation_count == 0u &&
           evidence.last_output_hash_observation_count == 0u &&
           evidence.first_output_hash_reuse_count == 0u &&
           evidence.last_output_hash_reuse_count == 0u;
  }
  if (expected_runs == 1u) {
    return evidence.first_output_hash_observation_count == 1u &&
           evidence.last_output_hash_observation_count == 1u &&
           evidence.first_output_hash_reuse_count == 0u &&
           evidence.last_output_hash_reuse_count == 0u;
  }
  return evidence.first_output_hash_observation_count != 0u &&
         evidence.first_output_hash_observation_count ==
             evidence.last_output_hash_observation_count &&
         evidence.first_output_hash_reuse_count != 0u &&
         evidence.last_output_hash_reuse_count >=
             evidence.first_output_hash_reuse_count &&
         evidence.last_output_hash_reuse_count -
                 evidence.first_output_hash_reuse_count + 1u ==
             expected_runs &&
         evidence.hash_observation_changes == 0u &&
         evidence.hash_reuse_step_errors == 0u;
}

} // namespace

bool ExactProductRoute(const ProductRouteEvidence &evidence,
                       const ::rund::compute::Backend backend,
                       const std::uint64_t pages,
                       const std::uint64_t active_bytes,
                       const std::uint64_t expected_runs,
                       const bool resident_backing) noexcept {
  if (backend == ::rund::compute::Backend::Cpu || pages <= 1u) {
    return exact_zero(evidence);
  }
  if (!evidence.production_slots) {
    return false;
  }
  if (evidence.backend != backend) {
    return false;
  }
  const std::uint64_t coordinates = pages * expected_runs;
  const std::uint64_t bytes = active_bytes * expected_runs;
  const bool exact_preparation_lifetime =
      (expected_runs == 1u && evidence.cold_owner_runs == 1u &&
       evidence.warm_reused_runs == 0u &&
       evidence.max_warm_rearm_count == 0u) ||
      (evidence.cold_owner_runs == 0u &&
       evidence.warm_reused_runs == expected_runs &&
       evidence.max_warm_rearm_count >= expected_runs);
  const bool exact_public_backing =
      resident_backing
          ? evidence.public_gpu_addressable_backing_runs == expected_runs &&
                evidence.whole_run_staging_runs == 0u
          : evidence.public_gpu_addressable_backing_runs == 0u &&
                evidence.whole_run_staging_runs == expected_runs;
  return evidence.prepare_attempts == expected_runs &&
         evidence.prepared_runs == expected_runs &&
         evidence.executed_runs == expected_runs &&
         evidence.successful_finals == expected_runs &&
         exact_preparation_lifetime &&
         evidence.whole_run_preencoded_runs == 0u &&
         evidence.device_generated_recurrence_runs == expected_runs &&
         evidence.fixed_native_storage_runs == expected_runs &&
         evidence.fixed_common_storage_runs == expected_runs &&
         evidence.gpu_addressable_backing_runs == expected_runs &&
         exact_public_backing &&
         evidence.bounded_external_page_service_runs == 0u &&
         evidence.one_native_submit_runs == expected_runs &&
         evidence.host_service_turns_zero_runs == expected_runs &&
         evidence.host_epoch_callbacks_zero_runs == expected_runs &&
         evidence.aggregate_terminal_once_runs == expected_runs &&
         evidence.bounded_page_io_runs == expected_runs &&
         evidence.coordinate_count == coordinates &&
         evidence.accepted_coordinates == coordinates &&
         evidence.gpu_completed_coordinates == coordinates &&
         evidence.completed_prefix == coordinates &&
         evidence.forecasted_pages == coordinates &&
         evidence.promoted_pages == coordinates &&
         evidence.drained_pages == coordinates &&
         evidence.persisted_pages == coordinates &&
         evidence.overlap_reused_bytes == 0u &&
         evidence.gpu_backing_read_bytes == bytes &&
         evidence.gpu_backing_write_bytes == bytes &&
         evidence.native_submit_count == expected_runs &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.payload_dispatch_count == expected_runs &&
         evidence.host_service_turn_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.backing_wait_count == 0u &&
         evidence.backing_signal_count == 0u &&
         evidence.backing_acknowledgement_count == 0u &&
         evidence.final_callback_count == expected_runs &&
         evidence.queue_calls == expected_runs &&
         evidence.public_handoff_count == expected_runs &&
         evidence.authority_accept_count == expected_runs &&
         evidence.pipeline_terminal_count == expected_runs * 2u &&
         evidence.backing_publication_count == expected_runs &&
         exact_output_hash_cache(evidence, expected_runs, resident_backing) &&
         evidence.completed_ns != 0u && evidence.retained_bytes != 0u;
}

bool ExactGpuDrivenProductRoute(const ProductRouteEvidence &evidence,
                                const ::rund::compute::Backend backend,
                                const std::uint64_t pages,
                                const std::uint64_t active_bytes,
                                const std::uint64_t expected_runs,
                                const bool resident_backing) noexcept {
  return backend != ::rund::compute::Backend::Cpu && pages >= 2u &&
         ExactProductRoute(evidence, backend, pages, active_bytes,
                           expected_runs, resident_backing);
}

const char *ProductRouteStatus(const ProductRouteEvidence &evidence,
                               const ::rund::compute::Backend backend,
                               const std::uint64_t pages,
                               const bool resident_backing) noexcept {
  if (backend == ::rund::compute::Backend::Cpu) {
    return "cpu_reference";
  }
  if (pages == 0u) {
    return "empty_reference";
  }
  if (pages == 1u) {
    return "q1_nonpersistent";
  }
  if (evidence.successful_finals == 0u) {
    return "device_vsm_not_observed";
  }
  if (evidence.device_generated_recurrence_runs != evidence.successful_finals ||
      evidence.fixed_native_storage_runs != evidence.successful_finals ||
      evidence.fixed_common_storage_runs != evidence.successful_finals ||
      (resident_backing && evidence.public_gpu_addressable_backing_runs !=
                               evidence.successful_finals) ||
      (!resident_backing &&
       evidence.whole_run_staging_runs != evidence.successful_finals) ||
      evidence.bounded_external_page_service_runs != 0u ||
      evidence.host_service_turns_zero_runs != evidence.successful_finals) {
    return "device_vsm_contract_invalid";
  }
  return resident_backing ? "gpu_driven_device_vsm_resident_product"
                          : "gpu_driven_device_vsm_whole_run_staging_product";
}

} // namespace rund::measure::compute::virtual_residency
