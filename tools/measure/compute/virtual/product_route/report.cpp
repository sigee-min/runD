#include "report.hpp"

#include <array>
#include <cstdio>

namespace rund::measure::compute::virtual_residency {

void PrintProductRouteEvidence(const ProductRouteEvidence &route,
                               const std::uint64_t expected_runs) {
  std::printf(",%llu,%u", static_cast<unsigned long long>(expected_runs),
              static_cast<unsigned>(route.production_slots));
  const std::array values{
      route.prepare_attempts,
      route.prepared_runs,
      route.executed_runs,
      route.successful_finals,
      route.cold_owner_runs,
      route.warm_reused_runs,
      route.max_warm_rearm_count,
      route.whole_run_preencoded_runs,
      route.device_generated_recurrence_runs,
      route.fixed_native_storage_runs,
      route.fixed_common_storage_runs,
      route.gpu_addressable_backing_runs,
      route.public_gpu_addressable_backing_runs,
      route.whole_run_staging_runs,
      route.bounded_external_page_service_runs,
      route.one_native_submit_runs,
      route.host_service_turns_zero_runs,
      route.host_epoch_callbacks_zero_runs,
      route.aggregate_terminal_once_runs,
      route.bounded_page_io_runs,
      route.coordinate_count,
      route.accepted_coordinates,
      route.gpu_completed_coordinates,
      route.completed_prefix,
      route.forecasted_pages,
      route.promoted_pages,
      route.drained_pages,
      route.persisted_pages,
      route.overlap_reused_bytes,
      route.gpu_backing_read_bytes,
      route.gpu_backing_write_bytes,
      route.native_submit_count,
      route.epoch_native_submit_count,
      route.payload_dispatch_count,
      route.host_service_turn_count,
      route.backend_epoch_callback_count,
      route.host_epoch_callback_count,
      route.backing_wait_count,
      route.backing_signal_count,
      route.backing_acknowledgement_count,
      route.final_callback_count,
      route.queue_calls,
      route.public_handoff_count,
      route.authority_accept_count,
      route.pipeline_terminal_count,
      route.backing_publication_count,
      route.first_output_hash_observation_count,
      route.last_output_hash_observation_count,
      route.first_output_hash_reuse_count,
      route.last_output_hash_reuse_count,
      route.completed_ns,
      route.retained_bytes,
      route.transient_bytes,
  };
  for (const std::uint64_t value : values) {
    std::printf(",%llu", static_cast<unsigned long long>(value));
  }
}

} // namespace rund::measure::compute::virtual_residency
