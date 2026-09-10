#include "local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::route_matrix::oracle {

namespace oracle_detail {

[[nodiscard]] bool version_ok(const Row &row) noexcept {
  return row.version_after >= row.version_before &&
         row.version_after - row.version_before == CohortRuns;
}

[[nodiscard]] bool window_counts(const Row &row, std::uint64_t &pages,
                                 std::uint64_t &internal) noexcept {
  if (row.spec.family != "spatial_window" || row.spec.q == 0u ||
      !mul(row.spec.q, CohortRuns, pages) ||
      !mul(pages, WindowInternalFactor, internal)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool device_vsm_cohort(const Row &row) noexcept {
  const auto &route = row.route_evidence;
  return row.route == "device_vsm" && row.owner_id != 0u &&
         route.owner_events == CohortRuns && route.owner_stable &&
         route.proof_seen && route.proof_valid &&
         route.proof_valid_count == CohortRuns && route.proof_identity_stable &&
         route.proof_flags_stable &&
         route.capability_observations == CohortRuns &&
         route.capability_consistent &&
         (route.proof_hi != 0u || route.proof_lo != 0u) &&
         route.lifecycle_observations == CohortRuns &&
         route.lifecycle_mismatches == 0u &&
         route.per_run_lifecycle_consistent && route.lifecycle_complete &&
         route.cold_owner_runs == 1u &&
         route.warm_reused_runs == CohortRuns - 1u &&
         route.max_warm_rearm_count >= WarmSamples &&
         route.bounded_external_page_service_runs == 0u &&
         route.prepare_attempts == CohortRuns &&
         route.prepared_runs == CohortRuns &&
         route.executed_runs == CohortRuns &&
         route.successful_finals == CohortRuns &&
         route.final_callback_count == CohortRuns &&
         route.backing_publication_count == CohortRuns &&
         route.authority_accept_count == CohortRuns &&
         route.native_submit_count == CohortRuns &&
         route.queue_calls == CohortRuns &&
         route.payload_dispatch_count == CohortRuns &&
         route.public_handoff_count == CohortRuns &&
         route.one_native_submit_runs == CohortRuns &&
         route.device_generated_recurrence_runs == CohortRuns &&
         route.fixed_native_storage_runs == CohortRuns &&
         route.fixed_common_storage_runs == CohortRuns &&
         route.physical_ring_storage_runs == CohortRuns &&
         route.gpu_addressable_backing_runs == CohortRuns &&
         route.host_service_turns_zero_runs == CohortRuns &&
         route.host_epoch_callbacks_zero_runs == CohortRuns &&
         route.aggregate_terminal_once_runs == CohortRuns &&
         route.bounded_page_io_runs == CohortRuns &&
         route.host_service_turn_count == 0u &&
         route.host_epoch_callback_count == 0u && version_ok(row) &&
         row.stats.final_dispatches == 1u;
}

[[nodiscard]] bool window_ring_cohort(const Row &row) noexcept {
  const auto &route = row.route_evidence;
  const auto &residency = row.residency;
  std::uint64_t pages = 0u;
  std::uint64_t internal = 0u;
  if (!window_counts(row, pages, internal)) {
    return false;
  }
  std::uint64_t bytes = 0u;
  if (!mul(row.elements, sizeof(std::int32_t), bytes) || bytes == 0u ||
      bytes > std::numeric_limits<std::uint64_t>::max() / 2u) {
    return false;
  }
  const std::uint64_t logical_bytes = bytes * 2u;
  const bool backing_io = row.spec.resident
                              ? residency.backing_read_bytes == 0u &&
                                    residency.backing_write_bytes == 0u
                              : residency.backing_read_bytes == bytes &&
                                    residency.backing_write_bytes == bytes;
  const std::uint64_t endpoint =
      row.spec.resident ? ResidentEndpoint : StagedEndpoint;
  return row.route == "device_vsm" && row.owner_id != 0u &&
         route.owner_events == CohortRuns && route.owner_stable &&
         route.proof_seen && route.proof_valid && !route.proof_staged_loop &&
         !route.proof_graph_resident && route.proof_mode == WindowRingMode &&
         route.proof_endpoint == endpoint &&
         route.proof_valid_count == CohortRuns && route.proof_identity_stable &&
         route.proof_flags_stable &&
         route.capability_observations == CohortRuns &&
         route.capability_consistent && route.lifecycle_complete &&
         route.lifecycle_observations == CohortRuns &&
         route.lifecycle_mismatches == 0u &&
         route.prepare_attempts == CohortRuns &&
         route.prepared_runs == CohortRuns &&
         route.executed_runs == CohortRuns &&
         route.successful_finals == CohortRuns &&
         route.final_callback_count == CohortRuns &&
         route.backing_publication_count == CohortRuns &&
         route.authority_accept_count == CohortRuns &&
         route.native_submit_count == CohortRuns &&
         route.queue_calls == CohortRuns &&
         route.payload_dispatch_count == CohortRuns &&
         route.public_handoff_count == CohortRuns &&
         route.one_native_submit_runs == CohortRuns &&
         route.device_generated_recurrence_runs == CohortRuns &&
         route.fixed_native_storage_runs == CohortRuns &&
         route.fixed_common_storage_runs == CohortRuns &&
         route.physical_ring_storage_runs == CohortRuns &&
         route.host_service_turns_zero_runs == CohortRuns &&
         route.host_epoch_callbacks_zero_runs == CohortRuns &&
         route.aggregate_terminal_once_runs == CohortRuns &&
         route.bounded_page_io_runs == CohortRuns &&
         residency.logical_bytes == logical_bytes &&
         residency.page_count == row.spec.q &&
         residency.page_in_count == row.spec.q &&
         residency.page_out_count == row.spec.q &&
         residency.page_in_bytes == bytes &&
         residency.page_out_bytes == bytes && backing_io &&
         route.coordinate_count == pages &&
         route.accepted_coordinates == pages &&
         route.gpu_completed_coordinates == pages &&
         route.window_seed_dispatches == pages * WindowSeedFactor &&
         route.window_compute_dispatches == pages * WindowComputeFactor &&
         route.window_internal_dispatches == internal &&
         route.epoch_native_submit_count == 0u &&
         route.host_service_turn_count == 0u &&
         route.host_epoch_callback_count == 0u &&
         row.stats.command_submits == WindowPhysicalCount &&
         row.stats.dispatches == WindowPhysicalCount &&
         row.stats.final_dispatches == WindowPhysicalCount &&
         row.stats.transfer_submissions.host_to_device == 0u &&
         row.stats.transfer_submissions.device_to_host == 0u &&
         row.stats.transfer_submissions.device_to_device == 0u &&
         row.stats.uploaded_bytes == 0u && row.stats.downloaded_bytes == 0u &&
         residency.window_handoff_count == WindowLogicalHandoff &&
         residency.window_batch_count == WindowLogicalBatch &&
         residency.window_queue_call_count == WindowPhysicalCount &&
         version_ok(row);
}

} // namespace oracle_detail

using oracle_detail::device_vsm_cohort;
using oracle_detail::version_ok;
using oracle_detail::window_ring_cohort;

bool cohort_ok(const Row &row) noexcept {
  if (row.spec.family == "pointwise_callback" ||
      row.spec.family == "pointwise_resident") {
    return device_vsm_cohort(row);
  }
  if (row.spec.family == "spatial_window") {
    return window_ring_cohort(row);
  }
  return row.route != "ordinary" && version_ok(row) &&
         row.stats.final_dispatches == 1u;
}

} // namespace rund::measure::compute::route_matrix::oracle
