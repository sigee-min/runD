#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

bool HandlePersistentDeviceLoss(
    const PreparedCase &test_case,
    const accel::PersistentResidencySlidingSubmitResult &submit_result,
    const rund::AccelCheck &submitted,
    accel::PersistentResidencySlidingControl &control) noexcept {
  const auto &wait = test_case.wait;
  accel::VulkanResidencySlidingDiagnostics primary_after{};
  accel::VulkanResidencySlidingDiagnostics alternate_after{};
  const accel::PersistentResidencySlidingCapability quarantined_capability =
      accel::VulkanResidencyPersistentCapability(
          std::span<const accel::PersistentResidencySlidingRole>{
              wait.request.roles.data(), wait.request.width},
          wait.request.coordinate_count, wait.request.memory);
  std::uint64_t queue_after = 0u;
  {
    std::lock_guard lock{test_case.native->adapter->mutex};
    queue_after = test_case.native->adapter->command_submit_count;
  }
  const bool valid =
      !submitted.ok && submitted.reason != nullptr &&
      std::strcmp(submitted.reason, "compute_device_lost") == 0 &&
      submit_result.event ==
          accel::PersistentResidencySlidingSubmitEvent::AcceptedUnknown &&
      control.active && control.quarantined && control.native != nullptr &&
      control.accepted_end == 0u && control.native_submit_count == 0u &&
      control.queue_calls == 0u && control.chunk_submit_count == 0u &&
      wait.callback_count.load(std::memory_order_acquire) == 0u &&
      !wait.final.check.ok && !quarantined_capability.check.ok &&
      accel::InspectVulkanResidencySliding(test_case.primary, primary_after) &&
      accel::InspectVulkanResidencySliding(test_case.alternate,
                                           alternate_after) &&
      primary_after.gate_quarantined && primary_after.adapter_quarantined &&
      alternate_after.gate_quarantined && alternate_after.adapter_quarantined &&
      primary_after.gpu_result_read_count ==
          test_case.primary_before.gpu_result_read_count &&
      queue_after == test_case.queue_before;
  if (!valid) {
    std::fprintf(
        stderr,
        "Vulkan persistent submit loss: submit=%s active=%u quarantine=%u "
        "event=%u end=%llu submit=%llu queue=%llu chunks=%llu "
        "callbacks=%llu final=%s/%u capability=%s gates=%u/%u adapters=%u/%u "
        "reads=%llu/%llu queue=%llu/%llu\n",
        submitted.reason, static_cast<unsigned>(control.active),
        static_cast<unsigned>(control.quarantined),
        static_cast<unsigned>(submit_result.event),
        static_cast<unsigned long long>(control.accepted_end),
        static_cast<unsigned long long>(control.native_submit_count),
        static_cast<unsigned long long>(control.queue_calls),
        static_cast<unsigned long long>(control.chunk_submit_count),
        static_cast<unsigned long long>(
            wait.callback_count.load(std::memory_order_acquire)),
        wait.final.check.reason, static_cast<unsigned>(wait.final.terminal),
        quarantined_capability.check.reason,
        static_cast<unsigned>(primary_after.gate_quarantined),
        static_cast<unsigned>(primary_after.adapter_quarantined),
        static_cast<unsigned>(alternate_after.gate_quarantined),
        static_cast<unsigned>(alternate_after.adapter_quarantined),
        static_cast<unsigned long long>(
            test_case.primary_before.gpu_result_read_count),
        static_cast<unsigned long long>(primary_after.gpu_result_read_count),
        static_cast<unsigned long long>(test_case.queue_before),
        static_cast<unsigned long long>(queue_after));
  }
  return valid;
}

bool HandlePersistentUnknown(const PreparedCase &test_case,
                             accel::PersistentResidencySlidingControl &control,
                             const std::uint64_t submit_probe_ns) noexcept {
  const auto &wait = test_case.wait;
  accel::PersistentResidencySlidingServiceIdentity service_identity{};
  const bool identified = accel::persistent_sliding_service_identity(
      wait.request, 0u, service_identity);
  const std::uint64_t publication_before =
      test_case.output_backing->write_count();
  const rund::AccelCheck failed =
      identified ? test_case.lowering.backend.service.fail_service(
                       control,
                       accel::PersistentResidencySlidingServiceFailure{
                           .identity = service_identity,
                           .check = {false, "compute_device_lost"},
                           .terminal = accel::NativeTerminal::UnknownMayWrite,
                           .may_write = true})
                 : rund::AccelCheck{false, "identity"};
  accel::VulkanResidencySlidingDiagnostics primary_after{};
  accel::VulkanResidencySlidingDiagnostics alternate_after{};
  accel::PersistentResidencySlidingControl retry_control{};
  const auto retry = test_case.lowering.backend.service.submit_result(
      wait.request, retry_control);
  const rund::AccelCheck late_signal =
      accel::SignalVulkanResidencyPersistentReady(
          control, accel::PersistentResidencySlidingReadySignal{
                       .identity = service_identity});
  std::uint64_t queue_after = 0u;
  {
    std::lock_guard lock{test_case.native->adapter->mutex};
    queue_after = test_case.native->adapter->command_submit_count;
  }
  const accel::PersistentResidencySlidingCapability quarantined_capability =
      accel::VulkanResidencyPersistentCapability(
          std::span<const accel::PersistentResidencySlidingRole>{
              wait.request.roles.data(), wait.request.width},
          wait.request.coordinate_count, wait.request.memory);
  const bool valid =
      failed.ok && wait.callback_count.load(std::memory_order_acquire) == 1u &&
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
      wait.final.evidence.first_failure_coordinate == 0u &&
      wait.final.evidence.completed_ns != 0u &&
      wait.final.evidence.completed_ns <=
          accel::MonotonicNanoseconds() - submit_probe_ns &&
      PersistentRawControlCleared(test_case.primary) && !control.active &&
      control.quarantined && !retry.status.ok &&
      retry.event == accel::PersistentResidencySlidingSubmitEvent::Declined &&
      !late_signal.ok && !quarantined_capability.check.ok &&
      accel::InspectVulkanResidencySliding(test_case.primary, primary_after) &&
      accel::InspectVulkanResidencySliding(test_case.alternate,
                                           alternate_after) &&
      primary_after.gpu_result_read_count ==
          test_case.primary_before.gpu_result_read_count &&
      primary_after.gate_quarantined && primary_after.adapter_quarantined &&
      alternate_after.gate_quarantined && alternate_after.adapter_quarantined &&
      queue_after == test_case.queue_before + 1u &&
      test_case.output_backing->write_count() == publication_before &&
      wait.callback_count.load(std::memory_order_acquire) == 1u;
  if (!valid) {
    std::fprintf(
        stderr,
        "Vulkan persistent Unknown: fail=%s callbacks=%llu check=%u term=%u "
        "submit=%llu queue=%llu service=%llu active=%u quarantine=%u "
        "retry=%s signal=%s capability=%s gates=%u/%u adapters=%u/%u "
        "reads=%llu/%llu native_queue=%llu/%llu publication=%llu/%llu\n",
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
        late_signal.reason, quarantined_capability.check.reason,
        static_cast<unsigned>(primary_after.gate_quarantined),
        static_cast<unsigned>(primary_after.adapter_quarantined),
        static_cast<unsigned>(alternate_after.gate_quarantined),
        static_cast<unsigned>(alternate_after.adapter_quarantined),
        static_cast<unsigned long long>(
            test_case.primary_before.gpu_result_read_count),
        static_cast<unsigned long long>(primary_after.gpu_result_read_count),
        static_cast<unsigned long long>(test_case.queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned long long>(publication_before),
        static_cast<unsigned long long>(
            test_case.output_backing->write_count()));
  }
  return valid;
}

bool HandlePersistentKnown(const PreparedCase &test_case,
                           const PersistentRunOptions &options,
                           accel::PersistentResidencySlidingControl &control,
                           const std::uint64_t submit_probe_ns) {
  constexpr const char *failure_reason = "compute_backend_failed";
  const std::uint64_t failure_coordinate =
      options.failed_admission_only ? 0u : 2u;
  const bool known_suffix_failure =
      options.known_failure || options.failed_admission_only;
  const auto &wait = test_case.wait;
  for (std::uint64_t coordinate = 0u; coordinate < options.coordinate_count;
       ++coordinate) {
    accel::PersistentResidencySlidingServiceIdentity service_identity{};
    const bool identified = accel::persistent_sliding_service_identity(
        wait.request, coordinate, service_identity);
    const bool failed_suffix =
        known_suffix_failure && coordinate >= failure_coordinate;
    const rund::AccelCheck failed =
        identified && options.known_failure && coordinate == failure_coordinate
            ? test_case.lowering.backend.service.fail_service(
                  control,
                  accel::PersistentResidencySlidingServiceFailure{
                      .identity = service_identity,
                      .check = {false, failure_reason},
                      .terminal = accel::NativeTerminal::Known,
                      .may_write = false})
            : rund::AccelCheck{true, "ok"};
    const rund::AccelCheck signaled =
        identified && failed.ok
            ? accel::SignalVulkanResidencyPersistentReady(
                  control,
                  accel::PersistentResidencySlidingReadySignal{
                      .identity = service_identity,
                      .admission = failed_suffix
                                       ? rund::AccelCheck{false, failure_reason}
                                       : rund::AccelCheck{true, "ok"}})
            : rund::AccelCheck{false, "identity"};
    accel::PersistentResidencySlidingDoneObservation observation{};
    const rund::AccelCheck waited =
        signaled.ok ? accel::WaitVulkanResidencyPersistentDone(
                          control,
                          accel::PersistentResidencySlidingDoneWait{
                              .identity = service_identity},
                          observation)
                    : rund::AccelCheck{false, "not-signaled"};
    const rund::AccelCheck acknowledged =
        waited.ok && observation.terminal == accel::NativeTerminal::Known &&
                observation.completed
            ? accel::AcknowledgeVulkanResidencyPersistentDone(
                  control,
                  accel::PersistentResidencySlidingAcknowledgeDone{
                      .identity = service_identity,
                      .service_check =
                          failed_suffix
                              ? rund::AccelCheck{false, failure_reason}
                              : rund::AccelCheck{true, "ok"}})
            : rund::AccelCheck{false, "not-acknowledged"};
    if (!identified || !failed.ok || !signaled.ok || !waited.ok ||
        observation.check.ok == failed_suffix ||
        observation.terminal != accel::NativeTerminal::Known ||
        observation.dispatched == failed_suffix || !observation.completed ||
        observation.may_write == failed_suffix || !acknowledged.ok) {
      std::fprintf(
          stderr,
          "Vulkan persistent: coordinate=%llu signal=%s wait=%s result=%s "
          "term=%u dispatch=%u complete=%u write=%u ack=%s\n",
          static_cast<unsigned long long>(coordinate), signaled.reason,
          waited.reason, observation.check.reason,
          static_cast<unsigned>(observation.terminal),
          static_cast<unsigned>(observation.dispatched),
          static_cast<unsigned>(observation.completed),
          static_cast<unsigned>(observation.may_write), acknowledged.reason);
      return false;
    }
  }
  std::uint64_t queue_after = 0u;
  {
    std::lock_guard lock{test_case.native->adapter->mutex};
    queue_after = test_case.native->adapter->command_submit_count;
  }
  const std::uint64_t expected_failed =
      known_suffix_failure ? options.coordinate_count - failure_coordinate : 0u;
  const std::uint64_t expected_first =
      known_suffix_failure ? failure_coordinate
                           : accel::PersistentSlidingNoCoordinate;
  const std::uint64_t expected_service_first =
      options.known_failure ? failure_coordinate
                            : accel::PersistentSlidingNoCoordinate;
  const bool valid =
      wait.callback_count.load(std::memory_order_acquire) == 1u &&
      accel::persistent_sliding_final_valid(wait.request, wait.final) &&
      wait.final.check.ok == !known_suffix_failure &&
      (!known_suffix_failure ||
       std::strcmp(wait.final.check.reason, failure_reason) == 0) &&
      wait.final.terminal == accel::NativeTerminal::Known &&
      wait.final.evidence.gpu_completed_coordinates ==
          options.coordinate_count &&
      wait.final.evidence.completed_prefix ==
          (known_suffix_failure ? failure_coordinate
                                : options.coordinate_count) &&
      wait.final.evidence.suppressed_first == expected_first &&
      wait.final.evidence.suppressed_count == expected_failed &&
      wait.final.evidence.native_submit_count == 1u &&
      wait.final.evidence.epoch_native_submit_count == 0u &&
      wait.final.evidence.backend_epoch_callback_count == 0u &&
      wait.final.evidence.host_epoch_callback_count == 0u &&
      wait.final.evidence.backing_signal_count == options.coordinate_count &&
      wait.final.evidence.backing_wait_count == options.coordinate_count &&
      wait.final.evidence.backing_acknowledgement_count ==
          options.coordinate_count &&
      wait.final.evidence.backing_service_failure_count ==
          static_cast<std::uint64_t>(options.known_failure) &&
      wait.final.evidence.failed_admission_count == expected_failed &&
      wait.final.evidence.first_failed_admission_coordinate == expected_first &&
      wait.final.evidence.first_service_failure_coordinate ==
          expected_service_first &&
      wait.final.evidence.final_callback_count == 1u &&
      wait.final.evidence.queue_calls == 1u &&
      wait.final.evidence.completed_ns != 0u &&
      wait.final.evidence.completed_ns <=
          accel::MonotonicNanoseconds() - submit_probe_ns &&
      wait.final.evidence.first_failure_coordinate == expected_first &&
      queue_after == test_case.queue_before + 1u;
  if (!valid) {
    std::fprintf(
        stderr,
        "Vulkan persistent: final callbacks=%llu check=%u reason=%s gpu=%llu "
        "submit=%llu epoch=%llu backend=%llu host=%llu signal=%llu wait=%llu "
        "ack=%llu final=%llu queue=%llu native_queue=%llu/%llu\n",
        static_cast<unsigned long long>(
            wait.callback_count.load(std::memory_order_acquire)),
        static_cast<unsigned>(wait.final.check.ok), wait.final.check.reason,
        static_cast<unsigned long long>(
            wait.final.evidence.gpu_completed_coordinates),
        static_cast<unsigned long long>(
            wait.final.evidence.native_submit_count),
        static_cast<unsigned long long>(
            wait.final.evidence.epoch_native_submit_count),
        static_cast<unsigned long long>(
            wait.final.evidence.backend_epoch_callback_count),
        static_cast<unsigned long long>(
            wait.final.evidence.host_epoch_callback_count),
        static_cast<unsigned long long>(
            wait.final.evidence.backing_signal_count),
        static_cast<unsigned long long>(wait.final.evidence.backing_wait_count),
        static_cast<unsigned long long>(
            wait.final.evidence.backing_acknowledgement_count),
        static_cast<unsigned long long>(
            wait.final.evidence.final_callback_count),
        static_cast<unsigned long long>(wait.final.evidence.queue_calls),
        static_cast<unsigned long long>(test_case.queue_before),
        static_cast<unsigned long long>(queue_after));
  }
  return valid;
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
