#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline_metal_persistent {

template <std::size_t Iterations>
[[nodiscard]] bool RunFusedDirectRecurrenceCase(
    accel::MetalFusedDirectRecurrenceDiagnostics &diagnostics,
    std::uint32_t &state_count,
    accel::PreparedPipelineMemory &common_memory_out) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> seed{1, 3, 5, 7, 9, 11, 13, 15};
  auto opened = open(Target::metal());
  if (!opened) {
#if defined(__APPLE__)
    std::fprintf(stderr, "fused recurrence Q=%zu open=%u\n", Iterations,
                 static_cast<unsigned>(opened.reason()));
    return false;
#else
    return opened.reason() == Reason::AdapterUnavailable;
#endif
  }
  Device device = std::move(opened).value();
  auto input = device.upload<std::int32_t>(std::span<const std::int32_t>{seed});
  auto output = device.buffer<std::int32_t>(seed.size());
  auto body =
      on(device)
          .map<std::int32_t>("metal-fused-direct-recurrence", seed.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto fused =
      input && output && body
          ? pipeline(device)
                .repeat<Iterations>(*body, read(*input), write_final(*output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!fused) {
    std::fprintf(stderr, "fused recurrence Q=%zu prepare=%u\n", Iterations,
                 static_cast<unsigned>(fused.reason()));
    return false;
  }
  const std::shared_ptr<detail::PipelineState> &state =
      detail::PipelineStateAccess::state(*fused);
  auto *const common = state == nullptr || !state->prepared.ok
                           ? nullptr
                           : static_cast<accel::prepared::PipelineState *>(
                                 state->prepared.owner.get());
  const std::shared_ptr<void> backend =
      common == nullptr ? std::shared_ptr<void>{} : common->backend;
  const std::shared_ptr<const accel::ServiceFreeDirectProof> proof =
      accel::PreparedKernelPipelineServiceFreeDirectProof(state->prepared);
  const accel::PreparedPipelineMemory common_memory =
      accel::ReadPreparedKernelPipelineMemory(state->prepared);
  accel::MetalFusedDirectRecurrenceDiagnostics before{};
  if (backend == nullptr || proof == nullptr ||
      !accel::service_free_direct_proof_valid(*proof) ||
      proof->iterations != Iterations ||
      proof->artifact->key.api != rund::kernel::ComputeApi::Metal ||
      proof->retention != accel::ServiceFreeDirectRetention::Terminal ||
      !accel::InspectMetalFusedDirectRecurrence(backend, before) ||
      before.iteration_count != Iterations ||
      before.native_command_count == 0u || before.native_dispatch_count != 1u ||
      before.native_storage_bytes == 0u || before.route_host_bytes == 0u ||
      before.retained_bytes == 0u || common_memory.host.current == 0u ||
      before.iteration_argument_bytes != sizeof(std::uint32_t)) {
    std::fprintf(
        stderr,
        "fused recurrence Q=%zu proof=%u inspect iterations=%llu command=%llu "
        "dispatch=%llu native=%llu route=%llu retained=%llu argument=%llu\n",
        Iterations, static_cast<unsigned>(proof != nullptr),
        static_cast<unsigned long long>(before.iteration_count),
        static_cast<unsigned long long>(before.native_command_count),
        static_cast<unsigned long long>(before.native_dispatch_count),
        static_cast<unsigned long long>(before.native_storage_bytes),
        static_cast<unsigned long long>(before.route_host_bytes),
        static_cast<unsigned long long>(before.retained_bytes),
        static_cast<unsigned long long>(before.iteration_argument_bytes));
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
  std::array<std::int32_t, seed.size()> observed{};
  accel::MetalFusedDirectRecurrenceDiagnostics after{};
  const bool read =
      executed && rund_node_test_pipeline::ReadExact(*fused, *output, observed);
  bool exact = read;
  for (std::size_t index = 0u; exact && index < seed.size(); ++index) {
    exact =
        observed[index] == seed[index] + static_cast<std::int32_t>(Iterations);
  }
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
      !accel::InspectMetalFusedDirectRecurrence(backend, after) ||
      after.iteration_count != before.iteration_count ||
      after.native_command_count != before.native_command_count ||
      after.native_dispatch_count != before.native_dispatch_count ||
      after.native_storage_bytes != before.native_storage_bytes ||
      after.route_host_bytes != before.route_host_bytes ||
      after.retained_bytes != before.retained_bytes ||
      after.iteration_argument_bytes != before.iteration_argument_bytes) {
    std::fprintf(
        stderr,
        "fused recurrence Q=%zu run=%u read=%u exact=%u submits=%llu "
        "dispatches=%llu selected=%u quarantine=%u fixed=%u/%u "
        "iterations=%llu/%llu callbacks=%llu service=%llu authority=%llu "
        "states=%u/%u generation=%llu/%llu stats=%llu/%llu/%llu "
        "storage=%llu/%llu retained=%llu/%llu\n",
        Iterations, static_cast<unsigned>(executed.reason()),
        static_cast<unsigned>(read), static_cast<unsigned>(exact),
        static_cast<unsigned long long>(aggregate.native_submit_count),
        static_cast<unsigned long long>(aggregate.payload_dispatch_count),
        static_cast<unsigned>(aggregate.selected),
        static_cast<unsigned>(aggregate.quarantined),
        static_cast<unsigned>(aggregate.fixed_native_storage),
        static_cast<unsigned>(aggregate.fixed_common_storage),
        static_cast<unsigned long long>(aggregate.iterations),
        static_cast<unsigned long long>(aggregate.completed_iterations),
        static_cast<unsigned long long>(aggregate.final_callback_count),
        static_cast<unsigned long long>(aggregate.host_service_turn_count),
        static_cast<unsigned long long>(aggregate.authority_publication_count),
        aggregate.registered_state_count, state_count,
        static_cast<unsigned long long>(generation_before),
        static_cast<unsigned long long>(fused->generation()),
        static_cast<unsigned long long>(execution.command_submits),
        static_cast<unsigned long long>(execution.dispatches),
        static_cast<unsigned long long>(
            execution.pipeline.coalesced_repetition_count),
        static_cast<unsigned long long>(before.native_storage_bytes),
        static_cast<unsigned long long>(after.native_storage_bytes),
        static_cast<unsigned long long>(before.retained_bytes),
        static_cast<unsigned long long>(after.retained_bytes));
    return false;
  }
  diagnostics = after;
  common_memory_out = common_memory;
  if (!rund_node_test_pipeline_residency::service_free_direct_test::
          product_known_rejection_retry(*fused, state, proof)) {
    std::fprintf(stderr, "fused recurrence Q=%zu Known retry failed\n",
                 Iterations);
    return false;
  }
  observed = {};
  if (!rund_node_test_pipeline::ReadExact(*fused, *output, observed)) {
    return false;
  }
  for (std::size_t index = 0u; index < seed.size(); ++index) {
    if (observed[index] !=
        seed[index] + static_cast<std::int32_t>(Iterations)) {
      return false;
    }
  }
  if (!rund_node_test_pipeline_residency::service_free_direct_test::
          product_unknown_quarantine(*fused, state)) {
    std::fprintf(stderr, "fused recurrence Q=%zu Unknown quarantine failed\n",
                 Iterations);
    return false;
  }
  std::fprintf(stderr,
               "fused product recurrence Q=%zu handoff=%llu command=%llu "
               "dispatch=%llu native=%llu route=%llu retained=%llu "
               "common=%llu/%llu/%llu argument=%llu\n",
               Iterations,
               static_cast<unsigned long long>(aggregate.public_handoff_count),
               static_cast<unsigned long long>(after.native_command_count),
               static_cast<unsigned long long>(after.native_dispatch_count),
               static_cast<unsigned long long>(after.native_storage_bytes),
               static_cast<unsigned long long>(after.route_host_bytes),
               static_cast<unsigned long long>(after.retained_bytes),
               static_cast<unsigned long long>(common_memory.host.current),
               static_cast<unsigned long long>(common_memory.device.current),
               static_cast<unsigned long long>(common_memory.staging.current),
               static_cast<unsigned long long>(after.iteration_argument_bytes));
  return true;
}

[[nodiscard]] bool CheckFusedDirectRecurrence() {
  std::array<accel::MetalFusedDirectRecurrenceDiagnostics, 3u> fused{};
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
                 "Metal fused Direct recurrence fixed-storage contract failed "
                 "commands=%llu/%llu/%llu native=%llu/%llu/%llu "
                 "route=%llu/%llu/%llu retained=%llu/%llu/%llu\n",
                 static_cast<unsigned long long>(fused[0].native_command_count),
                 static_cast<unsigned long long>(fused[1].native_command_count),
                 static_cast<unsigned long long>(fused[2].native_command_count),
                 static_cast<unsigned long long>(fused[0].native_storage_bytes),
                 static_cast<unsigned long long>(fused[1].native_storage_bytes),
                 static_cast<unsigned long long>(fused[2].native_storage_bytes),
                 static_cast<unsigned long long>(fused[0].route_host_bytes),
                 static_cast<unsigned long long>(fused[1].route_host_bytes),
                 static_cast<unsigned long long>(fused[2].route_host_bytes),
                 static_cast<unsigned long long>(fused[0].retained_bytes),
                 static_cast<unsigned long long>(fused[1].retained_bytes),
                 static_cast<unsigned long long>(fused[2].retained_bytes));
    return false;
  }
  return true;
}


} // namespace rund_node_test_pipeline_metal_persistent

#endif
