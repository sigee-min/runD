#include "evidence.hpp"

#include <limits>

namespace rund_node_test_device_vsm_product {

namespace {

using Owner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using Endpoint = rund::compute::detail::VirtualDeviceVsmEndpoint;
using Route = rund::compute::detail::VirtualDeviceVsmRouteKind;

[[nodiscard]] bool exact_extent(const std::size_t actual,
                                const std::uint64_t expected) noexcept {
  return expected <= std::numeric_limits<std::size_t>::max() &&
         actual == static_cast<std::size_t>(expected);
}

[[nodiscard]] bool no_resident(const Owner &owner) noexcept {
  for (const auto &input : owner.resident_inputs) {
    if (input != nullptr) {
      return false;
    }
  }
  return owner.resident_output == nullptr;
}

[[nodiscard]] bool no_staging(const Owner &owner) noexcept {
  for (const auto &input : owner.input_staging) {
    if (!input.empty()) {
      return false;
    }
  }
  return owner.output_staging.empty();
}

[[nodiscard]] bool one_input_staging(const Owner &owner,
                                     const std::uint64_t bytes) noexcept {
  if (!exact_extent(owner.input_staging[0u].size(), bytes)) {
    return false;
  }
  for (std::size_t index = 1u; index < owner.input_staging.size(); ++index) {
    if (!owner.input_staging[index].empty()) {
      return false;
    }
  }
  return exact_extent(owner.output_staging.size(), bytes);
}

} // namespace

bool ExactWholeRunTransfers(
    const rund::compute::detail::device_vsm_product_detail::
        DeviceVsmProductOwner &owner,
    const rund::compute::Stats &stats,
    const std::uint64_t active_bytes) noexcept {
  if (owner.proof == nullptr || !owner.route_proof.valid() ||
      owner.input_count != 1u ||
      active_bytes > std::numeric_limits<std::size_t>::max() ||
      !no_resident(owner)) {
    return false;
  }
  const auto &transfers = stats.transfer_submissions;
  if (owner.route_proof.kind() == Route::StagedLoop &&
      owner.route_proof.endpoint() == Endpoint::Staged) {
    return no_staging(owner) && stats.uploaded_bytes == 0u &&
           stats.downloaded_bytes == 0u && transfers.host_to_device == 0u &&
           transfers.device_to_host == 0u &&
           stats.host_write_bytes == active_bytes;
  }
  if (owner.route_proof.kind() != Route::Direct ||
      owner.route_proof.endpoint() != Endpoint::Invalid ||
      !one_input_staging(owner, active_bytes) || stats.uploaded_bytes != 0u ||
      stats.host_write_bytes != active_bytes) {
    return false;
  }
  // This fixture authenticates coherent private whole-run buffers and also
  // requires exactly one adapter submit. A staging upload/readback must not
  // be hidden in that recurrence count, even for the legacy Direct label.
  return stats.downloaded_bytes == 0u && transfers.host_to_device == 0u &&
         transfers.device_to_host == 0u;
}

bool ExactDeviceVsmEvidence(const RouteObservation &observation,
                            const std::uint64_t pages,
                            const std::uint64_t output_pages,
                            const std::uint64_t read_bytes,
                            const std::uint64_t write_bytes,
                            const std::uint64_t overlap) noexcept {
  const auto &product = observation.evidence;
  const auto &native = product.native;
  return observation.production_route && observation.prepared &&
         observation.executed && product.final_received &&
         !product.quarantined && product.public_handoff_count == 1u &&
         product.authority_accept_count == 1u &&
         product.pipeline_terminal_count == 2u &&
         product.backing_publication_count == 1u &&
         native.page_count == pages && native.generated_epochs == pages &&
         native.completed_epochs == pages && native.forecasted_pages == pages &&
         native.promoted_pages == pages &&
         native.drained_pages == output_pages &&
         native.persisted_pages == output_pages &&
         native.overlap_reused_bytes == overlap &&
         native.gpu_backing_read_bytes == read_bytes &&
         native.gpu_backing_write_bytes == write_bytes &&
         native.native_submit_count == 1u &&
         native.epoch_native_submit_count == 0u &&
         native.payload_dispatch_count == 1u &&
         native.host_service_turn_count == 0u &&
         native.host_epoch_callback_count == 0u &&
         native.final_callback_count == 1u && native.max_live_frames == 2u &&
         native.may_write;
}

bool ExactWindowRingEvidence(
    const RouteObservation &observation, const std::uint64_t pages,
    const std::uint64_t read_bytes, const std::uint64_t write_bytes,
    const std::uint64_t overlap,
    const rund::node::accel::detail::DeviceVsmWindowRingPlan &plan) noexcept {
  const auto &native = observation.evidence.native;
  return ExactDeviceVsmEvidence(observation, pages, pages, read_bytes,
                                write_bytes, overlap) &&
         native.ring_reuse_transitions == pages &&
         native.ring_schedule_checksum == plan.schedule_checksum &&
         native.ring_round_trips == pages &&
         native.ring_state_bytes == plan.state_bytes &&
         native.ring_scratch_bytes == plan.scratch_bytes &&
         native.window_seed_dispatches == pages &&
         native.window_compute_dispatches == pages &&
         native.window_internal_dispatches == pages * 2u;
}

bool ExactPublication(
    const rund_node_test_persistent_product::PublicationSnapshot before_primary,
    const rund_node_test_persistent_product::PublicationSnapshot after_primary,
    const rund_node_test_persistent_product::PublicationSnapshot
        before_alternate,
    const rund_node_test_persistent_product::PublicationSnapshot
        after_alternate,
    const std::uint64_t version_before, const std::uint64_t version_after,
    const std::uint64_t recovery_after) noexcept {
  return after_primary.generation == before_primary.generation + 1u &&
         after_primary.payload_epoch == before_primary.payload_epoch + 1u &&
         after_alternate.generation == before_alternate.generation + 1u &&
         after_alternate.payload_epoch == before_alternate.payload_epoch + 1u &&
         version_after == version_before + 1u && recovery_after == 0u;
}

} // namespace rund_node_test_device_vsm_product
