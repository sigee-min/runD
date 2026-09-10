#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

node::accel::detail::DeviceVsmFinal
unknown_final(const DeviceVsmProductRun &run) noexcept {
  namespace accel = node::accel::detail;
  const std::uint32_t submission_count = run.submission_control.count();
  if (run.owner == nullptr || run.owner->proof == nullptr ||
      !run.lease.has_value() || !*run.lease ||
      submission_count != 1u) {
    return {};
  }
  const accel::DeviceVsmProof &proof = *run.owner->proof;
  const bool ring = proof.topology == accel::DeviceVsmTopology::Pointwise ||
                    proof.topology == accel::DeviceVsmTopology::GraphPointwise;
  std::uint64_t ring_state_bytes = 0u;
  std::uint64_t ring_scratch_bytes = 0u;
  if (ring && !accel::device_vsm_ring_storage_expected(
                  proof.geometry, proof.width, proof.residents.count,
                  ring_state_bytes, ring_scratch_bytes)) {
    return {};
  }
  return accel::DeviceVsmFinal{
      .check = {false, "compute_device_lost"},
      .terminal = accel::DeviceVsmTerminal::UnknownMayWrite,
      .evidence =
          accel::DeviceVsmEvidence{
              .proof = proof.identity,
              .token = run.lease->token(),
              .generation = run.lease->generation(),
              .nonce = run.lease->owner(),
              .page_count = run.lease->iterations(),
              .ring_state_bytes = ring_state_bytes,
              .ring_scratch_bytes = ring_scratch_bytes,
              .native_submit_count = submission_count,
              .epoch_native_submit_count = 0u,
              .payload_dispatch_count = 1u,
              .host_service_turn_count = 0u,
              .host_epoch_callback_count = 0u,
              .final_callback_count = 1u,
              .max_live_frames = proof.width,
              .may_write = true,
          },
  };
}

} // namespace rund::compute::detail::device_vsm_product_detail
