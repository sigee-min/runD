#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_residency {

[[nodiscard]] int CheckAbort(ResidencyFixture &fixture) {
  // Abort is a drain, not cancellation. All four batches were accepted by
  // the one vkQueueSubmit call. The native service opens their no-write gates
  // in bank-safe order, waits each done point, then conservatively publishes
  // UnknownMayWrite for every dispatched batch and quarantines the owner.
  auto &wait = fixture.wait;
  auto &control = fixture.control;
  wait.done.store(false, std::memory_order_relaxed);
  wait.valid.store(true, std::memory_order_relaxed);
  wait.releases.store(0u, std::memory_order_relaxed);
  wait.final = {};
  wait.count = 4u;
  wait.first_epoch = 0u;
  wait.suppressed = std::numeric_limits<std::size_t>::max();
  wait.aborted = true;
  wait.token = 500u;
  wait.generation = 600u;
  wait.control = 4000u;
  accel::PreparedResidencyWindowRequest abort_request{
      .plan_identity = wait.plan,
      .token = wait.token,
      .generation = wait.generation,
      .batch_count = 4u,
      .release = CompleteRelease,
      .final = CompleteFinal,
      .user = &wait,
  };
  for (std::size_t epoch = 0u; epoch < abort_request.batch_count; ++epoch) {
    abort_request.batches[epoch].pipeline = wait.pipelines[epoch % 2u];
    abort_request.batches[epoch].locals[0u] = 0u;
    abort_request.batches[epoch].local_count = 1u;
    abort_request.batches[epoch].epoch = epoch;
    abort_request.batches[epoch].control_generation =
        wait.control + static_cast<std::uint32_t>(epoch);
    abort_request.batches[epoch].bank =
        static_cast<std::uint8_t>(epoch % 2u);
  }
  if (!accel::SubmitPreparedKernelPipelineWindow(wait.context, abort_request,
                                                 control)
           .ok ||
      !accel::AbortPreparedKernelPipelineWindow(
           wait.context, control,
           accel::BackendResidencyWindowAbort{
               .failure = {false, "compute_backend_failed"},
               .plan_identity = wait.plan,
               .token = wait.token,
               .generation = wait.generation,
           })
           .ok) {
    return 8;
  }
  wait.done.wait(false, std::memory_order_acquire);
  if (!wait.valid.load(std::memory_order_acquire) ||
      wait.releases.load(std::memory_order_acquire) != 4u ||
      wait.final.check.ok ||
      wait.final.terminal != accel::NativeTerminal::UnknownMayWrite ||
      wait.final.public_handoffs != 1u || wait.final.native_batches != 4u ||
      wait.final.queue_calls != 1u || wait.final.native_inflight_peak != 1u ||
      wait.final.receipt_count != 4u || control.active ||
      !control.quarantined) {
    return 9;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_vulkan_residency

#endif
