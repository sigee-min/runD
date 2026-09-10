#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline_metal_persistent {

[[nodiscard]] bool RejectOversizedPreparation(
    const accel::PersistentResidencySlidingRequest &request) noexcept {
  accel::PersistentResidencySlidingRequest oversized = request;
  oversized.coordinate_count = std::numeric_limits<std::uint64_t>::max();
  const accel::MetalPersistentResidencySlidingPreparation rejected =
      accel::PrepareMetalPersistentResidencySliding(oversized);
  return !rejected.capability.check.ok &&
         rejected.capability.check.reason != nullptr &&
         std::strcmp(rejected.capability.check.reason,
                     "compute_pipeline_capacity") == 0 &&
         rejected.lowering == nullptr;
}

[[nodiscard]] bool RunPersistentCase(const std::uint64_t coordinate_count,
                                     const std::uint64_t identity,
                                     const bool known_failure,
                                     const bool unknown_failure,
                                     const bool failed_admission_only) {
  using namespace rund::compute;
  constexpr std::size_t elements = 8u;
  const std::uint64_t failure_coordinate = failed_admission_only ? 0u : 2u;
  constexpr const char *failure_reason = "compute_backend_failed";
  const bool known_suffix_failure = known_failure || failed_admission_only;
  if ((known_failure && unknown_failure) ||
      (known_failure && failed_admission_only) ||
      (unknown_failure && failed_admission_only)) {
    return false;
  }
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::int32_t>("metal-persistent-sliding", 1u,
                                        [](auto value) { return value + 1; })
                     .compile();
  auto input_backing =
      std::make_shared<PersistentBacking>(elements * sizeof(std::int32_t));
  auto output_backing =
      std::make_shared<PersistentBacking>(elements * sizeof(std::int32_t));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, input_backing);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, output_backing);
  auto prepared =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!prepared || prepared.value() == nullptr ||
      prepared.value()->pipeline == nullptr ||
      prepared.value()->alternate_pipeline == nullptr) {
    std::fprintf(stderr, "persistent: virtual prepare\n");
    return false;
  }
  const auto common_backend = [](const auto &pipeline) {
    auto *const common = pipeline == nullptr || !pipeline->prepared.ok
                             ? nullptr
                             : static_cast<accel::prepared::PipelineState *>(
                                   pipeline->prepared.owner.get());
    return common == nullptr ? std::shared_ptr<void>{} : common->backend;
  };
  const std::shared_ptr<void> primary =
      common_backend(prepared.value()->pipeline);
  const std::shared_ptr<void> alternate =
      common_backend(prepared.value()->alternate_pipeline);
  if (primary == nullptr || alternate == nullptr || primary == alternate) {
    std::fprintf(stderr, "persistent: raw owners\n");
    return false;
  }

  FinalWait wait{};
  wait.request.plan_identity = 0x4d'50'53'4cu + identity;
  wait.request.token = 100'000u + identity;
  wait.request.generation = 200'000u + identity;
  wait.request.owner_nonce = 900'000u + identity;
  wait.request.coordinate_count = coordinate_count;
  wait.request.tail_local_count = 1u;
  wait.request.admission = std::make_shared<std::uint8_t>(1u);
  wait.request.final = CompletePersistent;
  wait.request.user = &wait;
  wait.request.memory = accel::ResidencySlidingMemory::HostCoherent;
  wait.request.width = 2u;
  for (std::size_t slot = 0u; slot < wait.request.width; ++slot) {
    wait.request.roles[slot] = accel::PersistentResidencySlidingRole{
        .prepared = slot == 0u ? primary : alternate,
        .locals = {0u},
        .local_count = 1u,
        .first_control_generation =
            static_cast<std::uint32_t>(300'000u + identity * 1024u + slot),
        .control_generation_stride = 2u,
        .first_descriptor_generation = 400'000u + identity * 1024u + slot,
        .descriptor_generation_stride = 2u,
        .slot = static_cast<std::uint8_t>(slot),
    };
  }
  if (identity == 1u && !RejectOversizedPreparation(wait.request)) {
    std::fprintf(stderr, "persistent: oversized preparation accepted\n");
    return false;
  }
  const auto &primary_common = prepared.value()->pipeline->prepared;
  const auto &alternate_common = prepared.value()->alternate_pipeline->prepared;
  std::array<accel::PreparedResidencyPersistentSlidingRole, 2u> roles{};
  for (std::size_t slot = 0u; slot < wait.request.width; ++slot) {
    roles[slot] = accel::PreparedResidencyPersistentSlidingRole{
        .pipeline = slot == 0u ? primary_common : alternate_common,
        .locals = {0u},
        .local_count = 1u,
        .first_control_generation =
            static_cast<std::uint32_t>(300'000u + identity * 1024u + slot),
        .control_generation_stride = 2u,
        .first_descriptor_generation = 400'000u + identity * 1024u + slot,
        .descriptor_generation_stride = 2u,
        .slot = static_cast<std::uint8_t>(slot),
    };
  }
  const accel::PreparedResidencyPersistentSlidingPreparation lowering =
      accel::PrepareKernelPipelinePersistentSliding(
          wait.request,
          std::span<const accel::PreparedResidencyPersistentSlidingRole>{
              roles.data(), roles.size()});
  if (!lowering) {
    std::fprintf(stderr, "persistent: common lowering\n");
    return false;
  }
  wait.request = lowering.active_request();
  wait.request.lowering = lowering.backend.lowering;
  const auto &capability = lowering.backend.capability;
  if (!accel::persistent_sliding_product_capable(capability) ||
      accel::persistent_sliding_gpu_driven_capable(capability) ||
      lowering.backend.lowering == nullptr ||
      lowering.backend.encoded_coordinate_count != coordinate_count ||
      lowering.backend.encoded_intermediate_bytes == 0u ||
      capability.retained_bytes < lowering.backend.encoded_intermediate_bytes) {
    std::fprintf(
        stderr,
        "persistent: lowering ok=%u reason=%s q=%llu encoded=%llu bytes=%llu "
        "retained=%llu\n",
        static_cast<unsigned>(capability.check.ok), capability.check.reason,
        static_cast<unsigned long long>(coordinate_count),
        static_cast<unsigned long long>(
            lowering.backend.encoded_coordinate_count),
        static_cast<unsigned long long>(
            lowering.backend.encoded_intermediate_bytes),
        static_cast<unsigned long long>(capability.retained_bytes));
    return false;
  }
  accel::PersistentResidencySlidingRequest replay = wait.request;
  ++replay.token;
  accel::PersistentResidencySlidingControl replay_control{};
  const auto replay_result = lowering.backend.service.submit_result(
      replay, replay_control);
  if (replay_result.status.ok ||
      replay_result.event !=
          accel::PersistentResidencySlidingSubmitEvent::Declined) {
    std::fprintf(stderr, "persistent: accepted prepared-request replay\n");
    return false;
  }
  accel::MetalResidencySlidingDiagnostics primary_before{};
  if (!accel::InspectMetalResidencySliding(primary, primary_before)) {
    std::fprintf(stderr, "persistent: pre-inspect\n");
    return false;
  }
  accel::PersistentResidencySlidingControl control{};
  const accel::PersistentResidencySlidingServiceOps &ops =
      accel::MetalPersistentResidencySlidingServiceOps();
  if (ops.submit_result == nullptr || ops.signal_ready == nullptr ||
      ops.wait_done == nullptr || ops.ack_done == nullptr ||
      ops.fail_service == nullptr) {
    std::fprintf(stderr, "persistent: submit result unavailable\n");
    return false;
  }
  const auto submitted = ops.submit_result(wait.request, control);
  if (!submitted.status.ok ||
      submitted.event !=
          accel::PersistentResidencySlidingSubmitEvent::Accepted) {
    std::fprintf(stderr, "persistent: submit reason=%s event=%u\n",
                 submitted.status.reason,
                 static_cast<unsigned>(submitted.event));
    return false;
  }
  if (!AcceptOneSubmit(wait.request, control)) {
    std::fprintf(stderr, "persistent: acceptance transition\n");
    return false;
  }
  if (unknown_failure) {
    accel::PersistentResidencySlidingServiceIdentity service_identity{};
    const bool identified = accel::persistent_sliding_service_identity(
        wait.request, 0u, service_identity);
    const std::uint64_t publication_before = output_backing->write_count();
    const rund::AccelCheck failed =
        identified ? ops.fail_service(
                         control,
                         accel::PersistentResidencySlidingServiceFailure{
                             .identity = service_identity,
                             .check = {false, "compute_device_lost"},
                             .terminal = accel::NativeTerminal::UnknownMayWrite,
                             .may_write = true})
                   : rund::AccelCheck{false, "identity"};
    accel::MetalPersistentResidencySlidingDiagnostics diagnostics{};
    accel::MetalResidencySlidingDiagnostics primary_after{};
    accel::MetalResidencySlidingDiagnostics alternate_after{};
    accel::PersistentResidencySlidingControl retry_control{};
    const auto retry = ops.submit_result(wait.request, retry_control);
    const rund::AccelCheck late_signal =
        ops.signal_ready(control, accel::PersistentResidencySlidingReadySignal{
                                      .identity = service_identity});
    const bool valid =
        failed.ok &&
        wait.callback_count.load(std::memory_order_acquire) == 1u &&
        accel::persistent_sliding_final_valid(wait.request, wait.final) &&
        !wait.final.check.ok &&
        std::strcmp(wait.final.check.reason, "compute_device_lost") == 0 &&
        wait.final.terminal == accel::NativeTerminal::UnknownMayWrite &&
        wait.final.evidence.gpu_completed_coordinates == 0u &&
        wait.final.evidence.native_submit_count == 1u &&
        wait.final.evidence.epoch_native_submit_count == 0u &&
        wait.final.evidence.backend_epoch_callback_count == 0u &&
        wait.final.evidence.host_epoch_callback_count == 0u &&
        wait.final.evidence.backing_signal_count == 0u &&
        wait.final.evidence.backing_wait_count == 0u &&
        wait.final.evidence.backing_acknowledgement_count == 0u &&
        wait.final.evidence.backing_service_failure_count == 1u &&
        wait.final.evidence.first_service_failure_coordinate == 0u &&
        wait.final.evidence.final_callback_count == 1u &&
        wait.final.evidence.queue_calls == 1u &&
        wait.final.evidence.first_failure_coordinate == 0u && !control.active &&
        control.quarantined &&
        !retry.status.ok &&
        retry.event == accel::PersistentResidencySlidingSubmitEvent::Declined &&
        !late_signal.ok &&
        accel::InspectMetalPersistentResidencySliding(lowering.backend.lowering,
                                                      diagnostics) &&
        diagnostics.final_callback_count == 1u && diagnostics.submitted &&
        diagnostics.final_sent && diagnostics.quarantined &&
        accel::InspectMetalResidencySliding(primary, primary_after) &&
        accel::InspectMetalResidencySliding(alternate, alternate_after) &&
        primary_after.command_submit_count ==
            primary_before.command_submit_count + 1u &&
        primary_after.gpu_result_read_count ==
            primary_before.gpu_result_read_count &&
        primary_after.quarantined && alternate_after.quarantined &&
        output_backing->write_count() == publication_before &&
        wait.callback_count.load(std::memory_order_acquire) == 1u;
    if (!valid) {
      std::fprintf(
          stderr,
          "persistent Unknown: fail=%s callbacks=%llu check=%u term=%u "
          "submit=%llu queue=%llu service=%llu active=%u quarantine=%u "
          "retry=%s signal=%s diag=%llu/%llu/%u/%u gates=%u/%u reads=%llu/%llu "
          "publication=%llu/%llu\n",
          failed.reason,
          static_cast<unsigned long long>(
              wait.callback_count.load(std::memory_order_acquire)),
          static_cast<unsigned>(wait.final.check.ok),
          static_cast<unsigned>(wait.final.terminal),
          static_cast<unsigned long long>(
              wait.final.evidence.native_submit_count),
          static_cast<unsigned long long>(wait.final.evidence.queue_calls),
          static_cast<unsigned long long>(
              wait.final.evidence.backing_service_failure_count),
          static_cast<unsigned>(control.active),
          static_cast<unsigned>(control.quarantined), retry.status.reason,
          late_signal.reason,
          static_cast<unsigned long long>(diagnostics.native_submit_count),
          static_cast<unsigned long long>(diagnostics.queue_commit_count),
          static_cast<unsigned>(diagnostics.final_sent),
          static_cast<unsigned>(diagnostics.quarantined),
          static_cast<unsigned>(primary_after.quarantined),
          static_cast<unsigned>(alternate_after.quarantined),
          static_cast<unsigned long long>(primary_before.gpu_result_read_count),
          static_cast<unsigned long long>(primary_after.gpu_result_read_count),
          static_cast<unsigned long long>(publication_before),
          static_cast<unsigned long long>(output_backing->write_count()));
    }
    return valid;
  }
  for (std::uint64_t coordinate = 0u; coordinate < coordinate_count;
       ++coordinate) {
    accel::PersistentResidencySlidingServiceIdentity service_identity{};
    const bool identified = accel::persistent_sliding_service_identity(
        wait.request, coordinate, service_identity);
    const bool failed_suffix =
        known_suffix_failure && coordinate >= failure_coordinate;
    const rund::AccelCheck failed =
        identified && known_failure && coordinate == failure_coordinate
            ? ops.fail_service(control,
                               accel::PersistentResidencySlidingServiceFailure{
                                   .identity = service_identity,
                                   .check = {false, failure_reason},
                                   .terminal = accel::NativeTerminal::Known,
                                   .may_write = false})
            : rund::AccelCheck{true, "ok"};
    const rund::AccelCheck signaled =
        identified && failed.ok
            ? ops.signal_ready(
                  control,
                  accel::PersistentResidencySlidingReadySignal{
                      .identity = service_identity,
                      .admission = failed_suffix
                                       ? rund::AccelCheck{false, failure_reason}
                                       : rund::AccelCheck{true, "ok"}})
            : rund::AccelCheck{false, "identity"};
    if (!identified || !failed.ok || !signaled.ok) {
      std::fprintf(stderr,
                   "persistent: signal coordinate=%llu fail=%s reason=%s\n",
                   static_cast<unsigned long long>(coordinate), failed.reason,
                   signaled.reason);
      return false;
    }
    accel::PersistentResidencySlidingDoneObservation observation{};
    const rund::AccelCheck waited = ops.wait_done(
        control,
        accel::PersistentResidencySlidingDoneWait{.identity = service_identity},
        observation);
    const rund::AccelCheck acknowledged =
        waited.ok && observation.terminal == accel::NativeTerminal::Known &&
                observation.completed
            ? ops.ack_done(control,
                           accel::PersistentResidencySlidingAcknowledgeDone{
                               .identity = service_identity,
                               .service_check =
                                   failed_suffix
                                       ? rund::AccelCheck{false, failure_reason}
                                       : rund::AccelCheck{true, "ok"}})
            : rund::AccelCheck{false, "not-acknowledged"};
    if (!waited.ok || observation.check.ok == failed_suffix ||
        observation.terminal != accel::NativeTerminal::Known ||
        observation.dispatched == failed_suffix || !observation.completed ||
        observation.may_write == failed_suffix || !acknowledged.ok) {
      std::fprintf(
          stderr,
          "persistent: terminal coordinate=%llu wait=%s result=%s term=%u "
          "dispatch=%u complete=%u write=%u ack=%s\n",
          static_cast<unsigned long long>(coordinate), waited.reason,
          observation.check.reason, static_cast<unsigned>(observation.terminal),
          static_cast<unsigned>(observation.dispatched),
          static_cast<unsigned>(observation.completed),
          static_cast<unsigned>(observation.may_write), acknowledged.reason);
      return false;
    }
  }
  accel::MetalPersistentResidencySlidingDiagnostics diagnostics{};
  accel::MetalResidencySlidingDiagnostics primary_after{};
  const std::uint64_t expected_failed =
      known_suffix_failure ? coordinate_count - failure_coordinate : 0u;
  const std::uint64_t expected_first =
      known_suffix_failure ? failure_coordinate
                           : accel::PersistentSlidingNoCoordinate;
  const std::uint64_t expected_service_first =
      known_failure ? failure_coordinate : accel::PersistentSlidingNoCoordinate;
  const bool got_diagnostics = accel::InspectMetalPersistentResidencySliding(
      lowering.backend.lowering, diagnostics);
  const bool got_primary =
      accel::InspectMetalResidencySliding(primary, primary_after);
  const bool final_ok =
      wait.callback_count.load(std::memory_order_acquire) == 1u &&
      accel::persistent_sliding_final_valid(wait.request, wait.final) &&
      wait.final.check.ok == !known_suffix_failure &&
      (!known_suffix_failure ||
       std::strcmp(wait.final.check.reason, failure_reason) == 0) &&
      wait.final.terminal == accel::NativeTerminal::Known &&
      wait.final.evidence.gpu_completed_coordinates == coordinate_count &&
      wait.final.evidence.completed_prefix ==
          (known_suffix_failure ? failure_coordinate : coordinate_count) &&
      wait.final.evidence.suppressed_first == expected_first &&
      wait.final.evidence.suppressed_count == expected_failed &&
      wait.final.evidence.native_submit_count == 1u &&
      wait.final.evidence.epoch_native_submit_count == 0u &&
      wait.final.evidence.backend_epoch_callback_count == 0u &&
      wait.final.evidence.host_epoch_callback_count == 0u &&
      wait.final.evidence.backing_signal_count == coordinate_count &&
      wait.final.evidence.backing_wait_count == coordinate_count &&
      wait.final.evidence.backing_acknowledgement_count == coordinate_count &&
      wait.final.evidence.backing_service_failure_count ==
          static_cast<std::uint64_t>(known_failure) &&
      wait.final.evidence.failed_admission_count == expected_failed &&
      wait.final.evidence.first_failed_admission_coordinate == expected_first &&
      wait.final.evidence.first_service_failure_coordinate ==
          expected_service_first &&
      wait.final.evidence.final_callback_count == 1u &&
      wait.final.evidence.queue_calls == 1u &&
      wait.final.evidence.first_failure_coordinate == expected_first;
  const bool evidence_ok =
      got_diagnostics &&
      diagnostics.final_callback_count == 1u;
  const bool lifecycle_ok = !diagnostics.submitted && !diagnostics.final_sent &&
                            !diagnostics.quarantined;
  const bool counter_ok =
      got_primary && primary_after.command_submit_count ==
                         primary_before.command_submit_count + 1u;
  const bool valid = final_ok && evidence_ok && lifecycle_ok && counter_ok;
  if (!valid) {
    std::fprintf(
        stderr,
        "persistent: final callbacks=%llu check=%u reason=%s terminal=%u "
        "gpu=%llu prefix=%llu submit=%llu epoch=%llu backend_cb=%llu "
        "host_cb=%llu final=%llu queues=%llu diag=%llu/%llu/%llu/%llu/%u/%u "
        "submits=%llu/%llu\n",
        static_cast<unsigned long long>(
            wait.callback_count.load(std::memory_order_acquire)),
        static_cast<unsigned>(wait.final.check.ok), wait.final.check.reason,
        static_cast<unsigned>(wait.final.terminal),
        static_cast<unsigned long long>(
            wait.final.evidence.gpu_completed_coordinates),
        static_cast<unsigned long long>(wait.final.evidence.completed_prefix),
        static_cast<unsigned long long>(
            wait.final.evidence.native_submit_count),
        static_cast<unsigned long long>(
            wait.final.evidence.epoch_native_submit_count),
        static_cast<unsigned long long>(
            wait.final.evidence.backend_epoch_callback_count),
        static_cast<unsigned long long>(
            wait.final.evidence.host_epoch_callback_count),
        static_cast<unsigned long long>(
            wait.final.evidence.final_callback_count),
        static_cast<unsigned long long>(wait.final.evidence.queue_calls),
        static_cast<unsigned long long>(diagnostics.native_submit_count),
        static_cast<unsigned long long>(diagnostics.epoch_native_submit_count),
        static_cast<unsigned long long>(diagnostics.queue_commit_count),
        static_cast<unsigned long long>(diagnostics.final_callback_count),
        static_cast<unsigned>(diagnostics.submitted),
        static_cast<unsigned>(diagnostics.final_sent),
        static_cast<unsigned long long>(primary_before.command_submit_count),
        static_cast<unsigned long long>(primary_after.command_submit_count));
  }
  return valid;
}

} // namespace rund_node_test_pipeline_metal_persistent

#endif
