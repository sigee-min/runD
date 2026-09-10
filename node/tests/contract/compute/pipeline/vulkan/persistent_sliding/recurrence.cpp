#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

template <std::size_t Iterations>
[[nodiscard]] bool RunFusedDirectRecurrenceCase(
    accel::VulkanFusedDirectRecurrenceDiagnostics &diagnostics,
    std::uint32_t &state_count,
    accel::PreparedPipelineMemory &common_memory_out) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 8u> seed{1u, 3u,  5u,  7u,
                                               9u, 11u, 13u, 15u};
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable;
  }
  Device device = std::move(opened).value();
  auto input =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{seed});
  auto output = device.buffer<std::uint32_t>(seed.size());
  auto body =
      on(device)
          .map<std::uint32_t>("vulkan-fused-direct-recurrence", seed.size(),
                              [](auto value) { return value + 1u; })
          .compile();
  auto fused =
      input && output && body
          ? pipeline(device)
                .repeat<Iterations>(*body, read(*input), write_final(*output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!fused) {
    std::fprintf(stderr, "Vulkan fused recurrence Q=%zu prepare=%u\n",
                 Iterations, static_cast<unsigned>(fused.reason()));
    return false;
  }
  const std::shared_ptr<detail::PipelineState> &state =
      detail::PipelineStateAccess::state(*fused);
  const std::shared_ptr<void> backend = NativeBackend(state);
  std::shared_ptr<const accel::ServiceFreeDirectProof> proof;
  if (state != nullptr) {
    proof =
        accel::PreparedKernelPipelineServiceFreeDirectProof(state->prepared);
  }
  const accel::PreparedPipelineMemory common_memory =
      state == nullptr
          ? accel::PreparedPipelineMemory{}
          : accel::ReadPreparedKernelPipelineMemory(state->prepared);
  accel::VulkanFusedDirectRecurrenceDiagnostics before{};
  if (backend == nullptr || proof == nullptr ||
      !accel::service_free_direct_proof_valid(*proof) ||
      proof->iterations != Iterations ||
      proof->artifact->key.api != rund::kernel::ComputeApi::Vulkan ||
      proof->retention != accel::ServiceFreeDirectRetention::Terminal ||
      !accel::InspectVulkanFusedDirectRecurrence(backend, before) ||
      before.iteration_count != Iterations ||
      before.native_command_buffer_count != 1u ||
      before.native_dispatch_count != 1u || before.descriptor_set_count != 1u ||
      before.device_buffer_bytes == 0u || before.route_host_bytes == 0u ||
      common_memory.host.current == 0u ||
      before.push_constant_bytes != 2u * sizeof(std::uint32_t) ||
      before.command_buffer_allocation_bytes_observable) {
    std::fprintf(
        stderr,
        "Vulkan fused recurrence Q=%zu proof=%u inspect iterations=%llu "
        "command=%llu "
        "dispatch=%llu descriptor=%llu buffer=%llu route=%llu "
        "common=%llu/%llu/%llu push=%llu opaque=%u\n",
        Iterations, static_cast<unsigned>(proof != nullptr),
        static_cast<unsigned long long>(before.iteration_count),
        static_cast<unsigned long long>(before.native_command_buffer_count),
        static_cast<unsigned long long>(before.native_dispatch_count),
        static_cast<unsigned long long>(before.descriptor_set_count),
        static_cast<unsigned long long>(before.device_buffer_bytes),
        static_cast<unsigned long long>(before.route_host_bytes),
        static_cast<unsigned long long>(common_memory.host.current),
        static_cast<unsigned long long>(common_memory.device.current),
        static_cast<unsigned long long>(common_memory.staging.current),
        static_cast<unsigned long long>(before.push_constant_bytes),
        static_cast<unsigned>(
            before.command_buffer_allocation_bytes_observable));
    return false;
  }
  state_count = proof->state_count;
  if (state_count == 0u) {
    return false;
  }
  const std::uint64_t generation_before = fused->generation();
  const Status executed = fused->run();
  const detail::ServiceFreeDirectProductEvidence aggregate =
      state->service_free_direct;
  const Stats execution = fused->stats();
  std::array<std::uint32_t, seed.size()> observed{};
  const bool read =
      executed && rund_node_test_pipeline::ReadExact(*fused, *output, observed);
  bool exact = read;
  for (std::size_t index = 0u; exact && index < seed.size(); ++index) {
    exact = observed[index] == seed[index] + Iterations;
  }
  accel::VulkanFusedDirectRecurrenceDiagnostics after{};
  if (!exact || !aggregate.selected || aggregate.quarantined ||
      !aggregate.fixed_native_storage || !aggregate.fixed_common_storage ||
      aggregate.iterations != Iterations ||
      aggregate.completed_iterations != Iterations ||
      aggregate.native_submit_count != 1u ||
      aggregate.epoch_native_submit_count != 0u ||
      aggregate.payload_dispatch_count != 1u ||
      aggregate.host_service_turn_count != 0u ||
      aggregate.host_epoch_callback_count != 0u ||
      aggregate.final_callback_count != 1u ||
      aggregate.authority_publication_count != 1u ||
      aggregate.registered_state_count != state_count ||
      fused->generation() != generation_before + 1u ||
      execution.command_submits != 1u || execution.dispatches != 1u ||
      !accel::InspectVulkanFusedDirectRecurrence(backend, after) ||
      after.iteration_count != before.iteration_count ||
      after.native_command_buffer_count != before.native_command_buffer_count ||
      after.native_dispatch_count != before.native_dispatch_count ||
      after.descriptor_set_count != before.descriptor_set_count ||
      after.device_buffer_bytes != before.device_buffer_bytes ||
      after.route_host_bytes != before.route_host_bytes ||
      after.push_constant_bytes != before.push_constant_bytes ||
      after.command_buffer_allocation_bytes_observable) {
    std::fprintf(
        stderr,
        "Vulkan fused recurrence Q=%zu run=%u read=%u exact=%u "
        "submits=%llu dispatches=%llu storage=%llu/%llu route=%llu/%llu\n",
        Iterations, static_cast<unsigned>(executed.reason()),
        static_cast<unsigned>(read), static_cast<unsigned>(exact),
        static_cast<unsigned long long>(aggregate.native_submit_count),
        static_cast<unsigned long long>(aggregate.payload_dispatch_count),
        static_cast<unsigned long long>(before.device_buffer_bytes),
        static_cast<unsigned long long>(after.device_buffer_bytes),
        static_cast<unsigned long long>(before.route_host_bytes),
        static_cast<unsigned long long>(after.route_host_bytes));
    return false;
  }
  diagnostics = after;
  common_memory_out = common_memory;
  if (!rund_node_test_pipeline_residency::service_free_direct_test::
          product_known_rejection_retry(*fused, state, proof)) {
    std::fprintf(stderr, "Vulkan fused recurrence Q=%zu Known retry failed\n",
                 Iterations);
    return false;
  }
  observed = {};
  if (!rund_node_test_pipeline::ReadExact(*fused, *output, observed)) {
    return false;
  }
  for (std::size_t index = 0u; index < seed.size(); ++index) {
    if (observed[index] != seed[index] + Iterations) {
      return false;
    }
  }
  if (!rund_node_test_pipeline_residency::service_free_direct_test::
          product_unknown_quarantine(*fused, state)) {
    std::fprintf(stderr,
                 "Vulkan fused recurrence Q=%zu Unknown quarantine failed\n",
                 Iterations);
    return false;
  }
  std::fprintf(
      stderr,
      "Vulkan fused recurrence Q=%zu handoff=%llu command=%llu dispatch=%llu "
      "descriptor=%llu buffer=%llu route=%llu common=%llu/%llu/%llu "
      "push=%llu opaque=%u\n",
      Iterations,
      static_cast<unsigned long long>(aggregate.public_handoff_count),
      static_cast<unsigned long long>(after.native_command_buffer_count),
      static_cast<unsigned long long>(after.native_dispatch_count),
      static_cast<unsigned long long>(after.descriptor_set_count),
      static_cast<unsigned long long>(after.device_buffer_bytes),
      static_cast<unsigned long long>(after.route_host_bytes),
      static_cast<unsigned long long>(common_memory.host.current),
      static_cast<unsigned long long>(common_memory.device.current),
      static_cast<unsigned long long>(common_memory.staging.current),
      static_cast<unsigned long long>(after.push_constant_bytes),
      static_cast<unsigned>(after.command_buffer_allocation_bytes_observable));
  return true;
}

[[nodiscard]] bool CheckFusedDirectRecurrence() {
  std::array<accel::VulkanFusedDirectRecurrenceDiagnostics, 3u> fused{};
  std::array<std::uint32_t, 3u> state_counts{};
  std::array<accel::PreparedPipelineMemory, 3u> common_memory{};
  if (!RunFusedDirectRecurrenceCase<5u>(fused[0], state_counts[0],
                                        common_memory[0]) ||
      !RunFusedDirectRecurrenceCase<9u>(fused[1], state_counts[1],
                                        common_memory[1]) ||
      !RunFusedDirectRecurrenceCase<257u>(fused[2], state_counts[2],
                                          common_memory[2]) ||
      state_counts[0] != state_counts[1] ||
      state_counts[0] != state_counts[2] ||
      !SameFusedDirectStorage(fused[0], fused[1]) ||
      !SameFusedDirectStorage(fused[0], fused[2]) ||
      !SameCurrentCommonMemory(common_memory[0], common_memory[1]) ||
      !SameCurrentCommonMemory(common_memory[0], common_memory[2])) {
    std::fprintf(stderr,
                 "Vulkan fused recurrence fixed storage failed "
                 "buffer=%llu/%llu/%llu route=%llu/%llu/%llu\n",
                 static_cast<unsigned long long>(fused[0].device_buffer_bytes),
                 static_cast<unsigned long long>(fused[1].device_buffer_bytes),
                 static_cast<unsigned long long>(fused[2].device_buffer_bytes),
                 static_cast<unsigned long long>(fused[0].route_host_bytes),
                 static_cast<unsigned long long>(fused[1].route_host_bytes),
                 static_cast<unsigned long long>(fused[2].route_host_bytes));
    return false;
  }
  return true;
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
