#include "local.hpp"

#include <memory>

namespace rund_node_test_pipeline_residency::service_free_direct_test {

rund::AccelCheck
FakeSubmit(const accel::ServiceFreeDirectRequest &request) noexcept {
  const auto state = std::static_pointer_cast<FakeState>(request.lowering);
  if (state == nullptr || state->submit_count != 0u ||
      !accel::service_free_direct_request_valid(state->capability, request)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }

  ++state->submit_count;
  ++state->final_count;
  request.final(request.user,
                accel::ServiceFreeDirectFinal{
                    .check = {true, "ok"},
                    .terminal = accel::ServiceFreeDirectTerminal::Known,
                    .evidence =
                        accel::ServiceFreeDirectEvidence{
                            .proof = request.proof->identity,
                            .token = request.token,
                            .generation = request.generation,
                            .nonce = request.nonce,
                            .iterations = request.proof->iterations,
                            .completed_iterations = request.proof->iterations,
                            .native_submit_count = 1u,
                            .epoch_native_submit_count = 0u,
                            .payload_dispatch_count = 1u,
                            .host_service_turn_count = 0u,
                            .host_epoch_callback_count = 0u,
                            .final_callback_count = state->final_count,
                            .completed_ns = request.proof->iterations,
                            .may_write = true,
                        },
                });
  return {true, "ok"};
}

} // namespace rund_node_test_pipeline_residency::service_free_direct_test
