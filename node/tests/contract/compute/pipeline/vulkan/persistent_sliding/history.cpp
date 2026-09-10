#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

template <std::size_t Iterations>
[[nodiscard]] bool RunFusedDirectHistoryCase(
    accel::VulkanFusedDirectRecurrenceDiagnostics &diagnostics,
    std::uint32_t &state_count,
    accel::PreparedPipelineMemory &common_memory_out) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 8u> seed{1u, 3u,  5u,  7u,
                                               9u, 11u, 13u, 15u};
  constexpr std::size_t history_count = seed.size() * Iterations;
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable;
  }
  Device device = std::move(opened).value();
  auto input =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{seed});
  auto output = device.buffer<std::uint32_t>(history_count);
  auto body =
      on(device)
          .map<std::uint32_t>("vulkan-fused-direct-history", seed.size(),
                              [](auto value) { return value + 1u; })
          .compile();
  auto fused =
      input && output && body
          ? pipeline(device)
                .repeat<Iterations>(*body, read(*input), write_each(*output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!fused) {
    std::fprintf(stderr, "Vulkan fused history Q=%zu prepare=%u\n", Iterations,
                 static_cast<unsigned>(fused.reason()));
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
  const bool exact_proof =
      proof != nullptr &&
      proof->retention == accel::ServiceFreeDirectRetention::History &&
      proof->output_count == 1u && proof->outputs[0].count == history_count &&
      proof->outputs[0].bytes == history_count * sizeof(std::uint32_t) &&
      proof->outputs[0].element_bytes == sizeof(std::uint32_t) &&
      proof->outputs[0].stride_bytes == sizeof(std::uint32_t) &&
      proof->output_pitch_bytes[0] == seed.size() * sizeof(std::uint32_t) &&
      proof->output_handles[0] != nullptr;
  if (backend == nullptr || !exact_proof ||
      !accel::service_free_direct_proof_valid(*proof) ||
      proof->iterations != Iterations ||
      proof->artifact->key.api != rund::kernel::ComputeApi::Vulkan ||
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
        "Vulkan fused history Q=%zu proof=%u inspect iterations=%llu "
        "command=%llu dispatch=%llu descriptor=%llu buffer=%llu "
        "route=%llu pitch=%llu bytes=%llu\n",
        Iterations, static_cast<unsigned>(proof != nullptr),
        static_cast<unsigned long long>(before.iteration_count),
        static_cast<unsigned long long>(before.native_command_buffer_count),
        static_cast<unsigned long long>(before.native_dispatch_count),
        static_cast<unsigned long long>(before.descriptor_set_count),
        static_cast<unsigned long long>(before.device_buffer_bytes),
        static_cast<unsigned long long>(before.route_host_bytes),
        static_cast<unsigned long long>(
            proof == nullptr ? 0u : proof->output_pitch_bytes[0]),
        static_cast<unsigned long long>(
            proof == nullptr ? 0u : proof->outputs[0].bytes));
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
  std::array<std::uint32_t, history_count> observed{};
  const bool read =
      executed && rund_node_test_pipeline::ReadExact(*fused, *output, observed);
  bool exact = read;
  for (std::size_t iteration = 0u; exact && iteration < Iterations;
       ++iteration) {
    for (std::size_t index = 0u; exact && index < seed.size(); ++index) {
      exact = observed[iteration * seed.size() + index] ==
              seed[index] + static_cast<std::uint32_t>(iteration + 1u);
    }
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
        "Vulkan fused history Q=%zu run=%u read=%u exact=%u "
        "submits=%llu dispatches=%llu selected=%u quarantine=%u "
        "final=%llu service=%llu authority=%llu\n",
        Iterations, static_cast<unsigned>(executed.reason()),
        static_cast<unsigned>(read), static_cast<unsigned>(exact),
        static_cast<unsigned long long>(aggregate.native_submit_count),
        static_cast<unsigned long long>(aggregate.payload_dispatch_count),
        static_cast<unsigned>(aggregate.selected),
        static_cast<unsigned>(aggregate.quarantined),
        static_cast<unsigned long long>(aggregate.final_callback_count),
        static_cast<unsigned long long>(aggregate.host_service_turn_count),
        static_cast<unsigned long long>(aggregate.authority_publication_count));
    return false;
  }
  diagnostics = after;
  common_memory_out = common_memory;
  if (!rund_node_test_pipeline_residency::service_free_direct_test::
          product_known_rejection_retry(*fused, state, proof)) {
    std::fprintf(stderr, "Vulkan fused history Q=%zu Known retry failed\n",
                 Iterations);
    return false;
  }
  observed = {};
  if (!rund_node_test_pipeline::ReadExact(*fused, *output, observed)) {
    return false;
  }
  for (std::size_t iteration = 0u; iteration < Iterations; ++iteration) {
    for (std::size_t index = 0u; index < seed.size(); ++index) {
      if (observed[iteration * seed.size() + index] !=
          seed[index] + static_cast<std::uint32_t>(iteration + 1u)) {
        return false;
      }
    }
  }
  if (!rund_node_test_pipeline_residency::service_free_direct_test::
          product_unknown_quarantine(*fused, state)) {
    std::fprintf(stderr,
                 "Vulkan fused history Q=%zu Unknown quarantine failed\n",
                 Iterations);
    return false;
  }
  std::fprintf(
      stderr,
      "Vulkan fused history Q=%zu command=%llu dispatch=%llu "
      "descriptor=%llu buffer=%llu route=%llu pitch=%llu bytes=%llu\n",
      Iterations,
      static_cast<unsigned long long>(after.native_command_buffer_count),
      static_cast<unsigned long long>(after.native_dispatch_count),
      static_cast<unsigned long long>(after.descriptor_set_count),
      static_cast<unsigned long long>(after.device_buffer_bytes),
      static_cast<unsigned long long>(after.route_host_bytes),
      static_cast<unsigned long long>(proof->output_pitch_bytes[0]),
      static_cast<unsigned long long>(proof->outputs[0].bytes));
  return true;
}

[[nodiscard]] bool CheckFusedDirectHistory() {
  std::array<accel::VulkanFusedDirectRecurrenceDiagnostics, 3u> history{};
  std::array<std::uint32_t, 3u> history_state_counts{};
  std::array<accel::PreparedPipelineMemory, 3u> history_memory{};
  if (!RunFusedDirectHistoryCase<5u>(history[0], history_state_counts[0],
                                     history_memory[0]) ||
      !RunFusedDirectHistoryCase<9u>(history[1], history_state_counts[1],
                                     history_memory[1]) ||
      !RunFusedDirectHistoryCase<257u>(history[2], history_state_counts[2],
                                       history_memory[2]) ||
      history_state_counts[0] != history_state_counts[1] ||
      history_state_counts[0] != history_state_counts[2] ||
      !SameFusedDirectStorage(history[0], history[1]) ||
      !SameFusedDirectStorage(history[0], history[2]) ||
      !SameCurrentCommonMemory(history_memory[0], history_memory[1]) ||
      !SameCurrentCommonMemory(history_memory[0], history_memory[2])) {
    std::fprintf(stderr, "Vulkan fused Direct History contract failed\n");
    return false;
  }
  return true;
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
