#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <span>

namespace rund_node_test_device_vsm_product::window_test::model {

using rund::compute::ResidencyStats;
using rund_node_test_persistent_product::PreparedProduct;
using DeviceVsmProductOwner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using WindowRingPlan = rund::node::accel::detail::DeviceVsmWindowRingPlan;
using VirtualBackingAccess = rund::compute::detail::VirtualBackingAccess;

[[nodiscard]] std::uint64_t
RingLogicalBytes(const std::uint64_t bytes) noexcept {
  return bytes * 2u;
}

[[nodiscard]] std::uint64_t RingPeak(const std::uint64_t pages) noexcept {
  constexpr std::uint64_t frame_capacity =
      rund_node_test_persistent_product::ProductFrameCapacity;
  constexpr std::uint64_t bank_count =
      rund::compute::detail::residency::Pool::BankCount;
  const std::uint64_t capacity = frame_capacity * bank_count;
  return pages < capacity ? pages : capacity;
}

[[nodiscard]] std::uint64_t I32BackingVersion(
    const std::shared_ptr<rund::compute::VirtualBacking> &backing) noexcept {
  return backing == nullptr ? 0u : VirtualBackingAccess::version(*backing);
}

[[nodiscard]] std::uint64_t I32BackingRecovery(
    const std::shared_ptr<rund::compute::VirtualBacking> &backing) noexcept {
  return backing == nullptr ? 0u
                            : VirtualBackingAccess::recovery_bytes(*backing);
}

[[nodiscard]] constexpr std::int32_t
SaturatingAdd(const std::int32_t left, const std::int32_t right) noexcept {
  const std::int64_t value = static_cast<std::int64_t>(left) + right;
  if (value > std::numeric_limits<std::int32_t>::max()) {
    return std::numeric_limits<std::int32_t>::max();
  }
  if (value < std::numeric_limits<std::int32_t>::min()) {
    return std::numeric_limits<std::int32_t>::min();
  }
  return static_cast<std::int32_t>(value);
}

static_assert(SaturatingAdd(std::numeric_limits<std::int32_t>::max(), 1) ==
              std::numeric_limits<std::int32_t>::max());
static_assert(SaturatingAdd(std::numeric_limits<std::int32_t>::min(), -1) ==
              std::numeric_limits<std::int32_t>::min());

[[nodiscard]] bool ExactRingStats(const rund::compute::Stats &stats,
                                  const std::uint64_t pages,
                                  const std::uint64_t bytes,
                                  const std::uint64_t runs) noexcept {
  const ResidencyStats &residency = stats.pipeline.residency;
  const std::uint64_t epochs =
      pages / 2u + static_cast<std::uint64_t>(pages % 2u != 0u);
  const std::uint64_t logical_bytes = RingLogicalBytes(bytes);
  return stats.command_submits == runs && stats.dispatches == runs &&
         stats.final_dispatches == runs &&
         stats.external_roundtrip_bytes == 0u &&
         residency.logical_bytes == logical_bytes &&
         residency.page_count == pages && residency.frame_capacity == 2u &&
         residency.epoch_count == epochs * runs &&
         residency.window_handoff_count == runs &&
         residency.window_batch_count == runs &&
         residency.window_queue_call_count == runs &&
         residency.page_in_count == pages * runs &&
         residency.page_out_count == pages * runs &&
         residency.backing_read_bytes == bytes * runs &&
         residency.backing_write_bytes == bytes * runs &&
         residency.page_in_bytes == bytes * runs &&
         residency.page_out_bytes == bytes * runs &&
         residency.resident_frames_peak == RingPeak(pages);
}

[[nodiscard]] bool
ExactRingProof(const PreparedProduct &prepared,
               const RouteObservation &observation,
               const std::shared_ptr<DeviceVsmProductOwner> &owner,
               WindowRingPlan &plan) noexcept {
  if (prepared.state == nullptr || owner == nullptr ||
      owner->proof == nullptr || !observation.production_route ||
      !observation.prepared || !observation.executed ||
      owner->proof->topology !=
          rund::node::accel::detail::DeviceVsmTopology::Window ||
      !owner->proof->window.ring.gpu_owned ||
      !rund::node::accel::detail::device_vsm_window_ring_plan_expected(
          owner->proof->geometry, plan) ||
      owner->proof->window.ring != plan || owner->proof->width != 2u ||
      !owner->preparation.capability.physical_ring_storage ||
      !owner->preparation.capability.one_native_submit ||
      owner->resident_inputs[0u] != nullptr ||
      owner->resident_output != nullptr || owner->evidence == nullptr ||
      owner->evidence->public_resident_input_count != 0u ||
      owner->evidence->whole_run_staged_input_count != 1u ||
      owner->evidence->public_resident_output ||
      !owner->evidence->whole_run_staged_output) {
    return false;
  }
  return true;
}

[[nodiscard]] bool ExactRingFailure(
    const rund::compute::Status &status, const RouteObservation &observation,
    const std::uint64_t bytes, const std::uint64_t version_before,
    const std::uint64_t version_after, const std::uint64_t recovery,
    const std::uint64_t queue_before, const std::uint64_t queue_after,
    const rund::compute::Stats &stats) noexcept {
  const auto &native = observation.evidence.native;
  return !status && status.reason() == rund::compute::Reason::DeviceLost &&
         !observation.execution_status &&
         observation.execution_reason ==
             static_cast<unsigned>(rund::compute::Reason::DeviceLost) &&
         observation.production_route && observation.prepared &&
         observation.executed && observation.evidence.final_received &&
         observation.evidence.quarantined &&
         observation.evidence.authority_accept_count == 0u &&
         observation.evidence.pipeline_terminal_count == 0u &&
         observation.evidence.backing_publication_count == 0u &&
         observation.execution_poison && native.native_submit_count == 1u &&
         native.epoch_native_submit_count == 0u &&
         native.payload_dispatch_count == 1u &&
         native.host_service_turn_count == 0u &&
         native.host_epoch_callback_count == 0u &&
         native.final_callback_count == 1u && native.may_write &&
         queue_after == queue_before + 1u && version_after == version_before &&
         recovery == bytes && stats.command_submits == 1u &&
         stats.dispatches == 1u && stats.final_dispatches == 1u &&
         stats.pipeline.residency.window_handoff_count == 1u &&
         stats.pipeline.residency.window_batch_count == 1u &&
         stats.pipeline.residency.window_queue_call_count == 1u;
}

[[nodiscard]] bool PrepareRingShape(const rund::compute::Backend backend,
                                    const PipelineShape shape,
                                    PreparedProduct &prepared,
                                    bool &unavailable) {
  if (shape == PipelineShape::Window) {
    return PrepareWindowRingProduct(backend, prepared, unavailable);
  }
  if (!PrepareWindowProduct(backend, WindowRingPages, Operation::Sum,
                            Boundary::Clamp, shape, prepared, unavailable)) {
    return unavailable;
  }
  return prepared.state != nullptr &&
         !prepared.state->geometry.device_vsm_required;
}

[[nodiscard]] bool ExactI32Ring(const I32WindowProduct &prepared,
                                const std::uint64_t pages,
                                const rund::compute::Status &status,
                                const RouteObservation &observation,
                                const std::uint64_t queue_before,
                                const std::uint64_t queue_after,
                                const std::uint64_t version_before,
                                const std::uint64_t version_after) noexcept {
  if (prepared.state == nullptr || prepared.input == nullptr ||
      prepared.output == nullptr || !status ||
      !ExactI32WindowOutput(prepared) || queue_after != queue_before + 1u ||
      version_after != version_before + 1u ||
      I32BackingRecovery(prepared.output) != 0u ||
      !observation.production_route || !observation.prepared ||
      !observation.executed || !observation.execution_status ||
      observation.execution_poison || !observation.evidence.final_received ||
      observation.evidence.quarantined ||
      observation.evidence.public_handoff_count != 1u ||
      observation.evidence.authority_accept_count != 1u ||
      observation.evidence.pipeline_terminal_count != 2u ||
      observation.evidence.backing_publication_count != 1u ||
      observation.evidence.native.page_count != pages ||
      observation.evidence.native.generated_epochs != pages ||
      observation.evidence.native.completed_epochs != pages ||
      observation.evidence.native.window_seed_dispatches != pages ||
      observation.evidence.native.window_compute_dispatches != pages ||
      observation.evidence.native.window_internal_dispatches != pages * 2u ||
      observation.evidence.native.native_submit_count != 1u ||
      observation.evidence.native.epoch_native_submit_count != 0u ||
      observation.evidence.native.payload_dispatch_count != 1u ||
      observation.evidence.native.host_service_turn_count != 0u ||
      observation.evidence.native.host_epoch_callback_count != 0u ||
      observation.evidence.native.final_callback_count != 1u ||
      !observation.evidence.native.may_write) {
    return false;
  }
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(observation.owner);
  if (owner == nullptr || owner->proof == nullptr ||
      owner->proof->topology !=
          rund::node::accel::detail::DeviceVsmTopology::Window ||
      owner->proof->window.semantic.element !=
          rund::kernel::WindowElement::U32 ||
      owner->proof->window.semantic.domain !=
          rund::kernel::ComputeDomain::I32 ||
      owner->proof->window.semantic.op != rund::kernel::WindowOp::Sum ||
      owner->proof->window.semantic.boundary !=
          rund::kernel::WindowBoundary::Clamp ||
      !owner->proof->window.ring.gpu_owned || owner->proof->width != 2u ||
      !owner->preparation.capability.physical_ring_storage ||
      !owner->preparation.capability.one_native_submit ||
      owner->route_proof.endpoint !=
          (prepared.resident
               ? rund::compute::detail::VirtualDeviceVsmEndpoint::Resident
               : rund::compute::detail::VirtualDeviceVsmEndpoint::Staged)) {
    return false;
  }
  WindowRingPlan plan{};
  if (!rund::node::accel::detail::device_vsm_window_ring_plan_expected(
          owner->proof->geometry, plan) ||
      owner->proof->window.ring != plan ||
      prepared.state->pipeline == nullptr ||
      prepared.state->pipeline->residency == nullptr) {
    return false;
  }
  const auto &residency = prepared.state->stats.pipeline.residency;
  const std::uint64_t bytes = prepared.expected.size() * sizeof(std::int32_t);
  const std::uint64_t epochs = pages / 2u + (pages % 2u != 0u ? 1u : 0u);
  const bool page_io =
      residency.page_in_count == pages && residency.page_out_count == pages &&
      residency.page_in_bytes == bytes && residency.page_out_bytes == bytes;
  const bool backing_io = prepared.resident
                              ? residency.backing_read_bytes == 0u &&
                                    residency.backing_write_bytes == 0u
                              : residency.backing_read_bytes == bytes &&
                                    residency.backing_write_bytes == bytes;
  const auto &transfers = prepared.state->stats.transfer_submissions;
  return prepared.state->stats.command_submits == 1u &&
         prepared.state->stats.dispatches == 1u &&
         prepared.state->stats.final_dispatches == 1u &&
         prepared.state->stats.external_roundtrip_bytes == 0u &&
         transfers.host_to_device == 0u && transfers.device_to_host == 0u &&
         transfers.device_to_device == 0u &&
         prepared.state->stats.uploaded_bytes == 0u &&
         prepared.state->stats.downloaded_bytes == 0u &&
         residency.logical_bytes == bytes * 2u &&
         residency.page_count == pages && residency.frame_capacity == 2u &&
         residency.epoch_count == epochs &&
         residency.window_handoff_count == 1u &&
         residency.window_batch_count == 1u &&
         residency.window_queue_call_count == 1u && page_io && backing_io;
}

} // namespace rund_node_test_device_vsm_product::window_test::model

#endif
