#include "local.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test {

accel::DeviceVsmRearmResult
FakeRearm(const std::shared_ptr<void> &lowering,
          const std::shared_ptr<const accel::DeviceVsmProof> &proof) noexcept {
  const auto state = std::static_pointer_cast<FakeState>(lowering);
  if (state == nullptr || proof == nullptr ||
      !accel::device_vsm_proof_valid(*proof)) {
    return {.check = {false, "accel_kernel_pipeline_invalid"},
            .mutated = false};
  }
  if (!state->rearm_check.ok) {
    return {.check = state->rearm_check, .mutated = state->rearm_mutated};
  }
  state->submit_count = 0u;
  state->final_count = 0u;
  return {.check = state->rearm_check, .mutated = state->rearm_mutated};
}

rund::AccelCheck FakeSubmit(const accel::DeviceVsmRequest &request) noexcept {
  const auto state = std::static_pointer_cast<FakeState>(request.lowering);
  if (state == nullptr || state->submit_count != 0u ||
      !accel::device_vsm_request_valid(state->capability, request)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  std::uint64_t overlap = 0u;
  if (!accel::device_vsm_overlap_reuse_bytes(request.proof->geometry,
                                             overlap)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  if (!request.submission_control->accept()) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  ++state->submit_count;
  ++state->final_count;
  request.final(
      request.user,
      accel::DeviceVsmFinal{
          .check = {true, "ok"},
          .terminal = accel::DeviceVsmTerminal::Known,
          .evidence =
              accel::DeviceVsmEvidence{
                  .proof = request.proof->identity,
                  .token = request.token,
                  .generation = request.generation,
                  .nonce = request.nonce,
                  .page_count = request.proof->geometry.page_count,
                  .generated_epochs = request.proof->geometry.page_count,
                  .completed_epochs = request.proof->geometry.page_count,
                  .canonical_boundary_transitions =
                      request.proof->window.footprint.boundary_transition_count,
                  .canonical_footprint_checksum =
                      request.proof->window.footprint.projection_checksum,
                  .forecasted_pages = request.proof->geometry.page_count,
                  .promoted_pages = request.proof->geometry.page_count,
                  .drained_pages = request.proof->geometry.page_count,
                  .persisted_pages = request.proof->geometry.page_count,
                  .overlap_reused_bytes = overlap,
                  .gpu_backing_read_bytes =
                      request.proof->geometry.logical_bytes,
                  .gpu_backing_write_bytes = request.proof->output_bytes,
                  .native_submit_count = request.submission_control->count(),
                  .epoch_native_submit_count = 0u,
                  .payload_dispatch_count = 1u,
                  .host_service_turn_count = 0u,
                  .host_epoch_callback_count = 0u,
                  .final_callback_count = state->final_count,
                  .completed_ns = request.proof->geometry.page_count,
                  .max_live_frames = request.proof->width,
                  .may_write = true,
              },
      });
  return {true, "ok"};
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
