#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../allocation.hpp"
#include "product/rolling.hpp"
#include "product/window.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_pipeline_vulkan_residency {

[[nodiscard]] int CheckWarm(ResidencyFixture &fixture) {
  auto &wait = fixture.wait;
  auto &control = fixture.control;
  const auto submit_warm_window = [&](const std::size_t sample) noexcept {
    constexpr std::size_t count = 4u;
    wait.done.store(false, std::memory_order_relaxed);
    wait.valid.store(true, std::memory_order_relaxed);
    wait.releases.store(0u, std::memory_order_relaxed);
    wait.final = {};
    wait.count = count;
    wait.first_epoch = 0u;
    wait.suppressed = std::numeric_limits<std::size_t>::max();
    wait.aborted = false;
    wait.token = 10'000u + sample;
    wait.generation = 20'000u + sample;
    wait.control = static_cast<std::uint32_t>(30'000u + sample * 8u);
    accel::PreparedResidencyWindowRequest request{
        .plan_identity = wait.plan,
        .token = wait.token,
        .generation = wait.generation,
        .batch_count = count,
        .release = CompleteRelease,
        .final = CompleteFinal,
        .user = &wait,
    };
    for (std::size_t epoch = 0u; epoch < count; ++epoch) {
      request.batches[epoch].pipeline = wait.pipelines[epoch % 2u];
      request.batches[epoch].locals[0u] = 0u;
      request.batches[epoch].local_count = 1u;
      request.batches[epoch].epoch = epoch;
      request.batches[epoch].control_generation =
          wait.control + static_cast<std::uint32_t>(epoch);
      request.batches[epoch].bank = static_cast<std::uint8_t>(epoch % 2u);
    }
    if (!accel::SubmitPreparedKernelPipelineWindow(wait.context, request,
                                                   control)
             .ok ||
        wait.done.load(std::memory_order_acquire) ||
        wait.releases.load(std::memory_order_acquire) != 0u) {
      return false;
    }
    for (const std::size_t epoch : std::array<std::size_t, 2u>{1u, 0u}) {
      if (!accel::SignalPreparedKernelPipelineWindow(
               wait.context, wait.pipelines[epoch],
               accel::BackendResidencyWindowSignal{
                   .plan_identity = wait.plan,
                   .token = wait.token,
                   .generation = wait.generation,
                   .epoch = epoch,
                   .control_generation =
                       wait.control + static_cast<std::uint32_t>(epoch),
                   .bank = static_cast<std::uint8_t>(epoch % 2u),
               })
               .ok) {
        return false;
      }
    }
    wait.done.wait(false, std::memory_order_acquire);
    return wait.valid.load(std::memory_order_acquire) &&
           wait.releases.load(std::memory_order_acquire) == count &&
           wait.final.check.ok &&
           wait.final.terminal == accel::NativeTerminal::Known &&
           wait.final.public_handoffs == 1u &&
           wait.final.native_batches == count &&
           wait.final.queue_calls == 1u &&
           wait.final.native_inflight_peak == 2u && !control.active &&
           !control.quarantined;
  };
  constexpr std::size_t warm_runs = 60u;
  if (!submit_warm_window(0u)) {
    return 10;
  }
  const rund::RuntimeStats cold_native =
      rund::node::accel::ReadRuntimeStats(fixture.native->pick);
  node_compute_allocation::Start();
  bool warm_clean = true;
  for (std::size_t sample = 0u; warm_clean && sample < warm_runs; ++sample) {
    warm_clean = submit_warm_window(sample + 1u);
  }
  node_compute_allocation::Stop();
  const rund::RuntimeStats warm_native =
      rund::node::accel::ReadRuntimeStats(fixture.native->pick);
  const auto &cold_allocations = cold_native.run.allocations;
  const auto &warm_allocations = warm_native.run.allocations;
  // The global operator-new probe also sees MoltenVK's opaque implementation
  // allocations. runD-owned warm allocation evidence is the unchanged exact
  // native resource counters below; keep the opaque number diagnostic-only.
  const std::uint64_t opaque_allocations = node_compute_allocation::Count();
  if (!warm_clean ||
      cold_allocations.pipeline_compile_count !=
          warm_allocations.pipeline_compile_count ||
      cold_allocations.descriptor_pool_create_count !=
          warm_allocations.descriptor_pool_create_count ||
      cold_allocations.descriptor_set_allocate_count !=
          warm_allocations.descriptor_set_allocate_count ||
      cold_allocations.buffer_allocation_count !=
          warm_allocations.buffer_allocation_count) {
    std::fprintf(stderr,
                 "vulkan window warm valid=%u allocation=%llu "
                 "native=%llu/%llu/%llu/%llu -> %llu/%llu/%llu/%llu\n",
                 static_cast<unsigned>(warm_clean),
                 static_cast<unsigned long long>(opaque_allocations),
                 static_cast<unsigned long long>(
                     cold_allocations.pipeline_compile_count),
                 static_cast<unsigned long long>(
                     cold_allocations.descriptor_pool_create_count),
                 static_cast<unsigned long long>(
                     cold_allocations.descriptor_set_allocate_count),
                 static_cast<unsigned long long>(
                     cold_allocations.buffer_allocation_count),
                 static_cast<unsigned long long>(
                     warm_allocations.pipeline_compile_count),
                 static_cast<unsigned long long>(
                     warm_allocations.descriptor_pool_create_count),
                 static_cast<unsigned long long>(
                     warm_allocations.descriptor_set_allocate_count),
                 static_cast<unsigned long long>(
                     warm_allocations.buffer_allocation_count));
    return 10;
  }
  for (std::size_t epochs = 2u; epochs <= 4u; ++epochs) {
    if (!rund_node_test_pipeline::ProductStagedLoop(epochs)) {
      return 7;
    }
  }
  for (const std::size_t epochs : {5u, 9u, 17u}) {
    if (!rund_node_test_pipeline::ProductStagedLoop(epochs)) {
      return 7;
    }
  }
  if (!rund_node_test_pipeline::ProductStagedLoop(
          5u, rund_node_test_pipeline::ProductFault::KnownInput) ||
      !rund_node_test_pipeline::ProductStagedLoop(
          5u, rund_node_test_pipeline::ProductFault::DeviceLoss)) {
    return 7;
  }
  if (!rund_node_test_pipeline::ProductLegacyPersistentFallback()) {
    return 7;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_vulkan_residency

#endif
