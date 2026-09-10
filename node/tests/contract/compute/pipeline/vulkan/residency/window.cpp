#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstdio>

namespace rund_node_test_pipeline_vulkan_residency {

[[nodiscard]] int CheckWindow(ResidencyFixture &fixture) {
  using namespace rund::compute;
  auto &wait = fixture.wait;
  auto &control = fixture.control;
  for (std::size_t sample = 1u; sample <= 5u; ++sample) {
    const bool shifted = sample == 5u;
    const std::size_t count = shifted ? 4u : sample;
    wait.done.store(false, std::memory_order_relaxed);
    wait.valid.store(true, std::memory_order_relaxed);
    wait.releases.store(0u, std::memory_order_relaxed);
    wait.final = {};
    wait.count = count;
    wait.first_epoch = shifted ? 4u : 0u;
    wait.suppressed =
        !shifted && count == 4u ? 2u : std::numeric_limits<std::size_t>::max();
    wait.aborted = false;
    wait.token = 100u + count;
    wait.generation = 200u + count;
    wait.control = static_cast<std::uint32_t>(1000u + count * 8u);
    accel::PreparedResidencyWindowRequest request{
        .plan_identity = wait.plan,
        .token = wait.token,
        .generation = wait.generation,
        .first_epoch = wait.first_epoch,
        .batch_count = count,
        .release = CompleteRelease,
        .final = CompleteFinal,
        .user = &wait,
    };
    for (std::size_t slot = 0u; slot < count; ++slot) {
      const std::uint64_t epoch = wait.first_epoch + slot;
      request.batches[slot].pipeline = wait.pipelines[slot % 2u];
      request.batches[slot].locals[0u] = 0u;
      request.batches[slot].local_count = 1u;
      request.batches[slot].epoch = epoch;
      request.batches[slot].control_generation =
          wait.control + static_cast<std::uint32_t>(slot);
      request.batches[slot].bank = static_cast<std::uint8_t>(epoch % 2u);
    }
    if (!accel::SubmitPreparedKernelPipelineWindow(wait.context, request,
                                                   control)
             .ok) {
      return 4;
    }
    // The raw handoff returns after the single native queue call. With every
    // ready gate still closed, neither a Release nor Final can have run.
    if (wait.done.load(std::memory_order_acquire) ||
        wait.releases.load(std::memory_order_acquire) != 0u) {
      return 4;
    }
    if (shifted &&
        accel::SignalPreparedKernelPipelineWindow(
            wait.context, wait.pipelines[0u],
            accel::BackendResidencyWindowSignal{
                .plan_identity = wait.plan,
                .token = wait.token,
                .generation = wait.generation,
                .epoch = 0u,
                .control_generation = wait.control,
                .bank = 0u,
            })
            .ok) {
      return 5;
    }
    const std::array<std::size_t, 2u> bootstrap{1u, 0u};
    for (const std::size_t slot : bootstrap) {
      if (slot >= count) {
        continue;
      }
      const std::uint64_t epoch = wait.first_epoch + slot;
      accel::BackendResidencyWindowSignal signal{
          .plan_identity = wait.plan,
          .token = wait.token,
          .generation = wait.generation,
          .epoch = epoch,
          .control_generation = wait.control +
                                static_cast<std::uint32_t>(slot),
          .bank = static_cast<std::uint8_t>(epoch % 2u),
      };
      if (epoch == wait.suppressed) {
        signal.admission = {false, "compute_backend_failed"};
      }
      if (!accel::SignalPreparedKernelPipelineWindow(wait.context,
                                                     wait.pipelines[slot],
                                                     signal)
               .ok) {
        return 5;
      }
    }
    wait.done.wait(false, std::memory_order_acquire);
    const bool failed = wait.suppressed < count;
    if (!wait.valid.load(std::memory_order_acquire) ||
        wait.releases.load(std::memory_order_acquire) != count ||
        wait.final.check.ok == failed ||
        wait.final.terminal != accel::NativeTerminal::Known ||
        wait.final.public_handoffs != 1u ||
        wait.final.native_batches != count || wait.final.queue_calls != 1u ||
        wait.final.native_inflight_peak != (count == 1u ? 1u : 2u) ||
        wait.final.receipt_count != count || control.active ||
        control.quarantined) {
      std::fprintf(
          stderr,
          "vulkan window q=%zu valid=%u releases=%zu check=%u reason=%s "
          "terminal=%u handoff=%llu batches=%llu queues=%llu peak=%llu "
          "receipts=%zu active=%u quarantine=%u\n",
          count,
          static_cast<unsigned>(wait.valid.load(std::memory_order_acquire)),
          wait.releases.load(std::memory_order_acquire),
          static_cast<unsigned>(wait.final.check.ok), wait.final.check.reason,
          static_cast<unsigned>(wait.final.terminal),
          static_cast<unsigned long long>(wait.final.public_handoffs),
          static_cast<unsigned long long>(wait.final.native_batches),
          static_cast<unsigned long long>(wait.final.queue_calls),
          static_cast<unsigned long long>(wait.final.native_inflight_peak),
          wait.final.receipt_count, static_cast<unsigned>(control.active),
          static_cast<unsigned>(control.quarantined));
      return 6;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline_vulkan_residency

#endif
