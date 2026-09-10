#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_persistent {

bool ValidatePersistentBeforeSubmit(PreparedCase &test_case,
                                    const bool submit_device_lost) noexcept {
  const auto &request = test_case.wait.request;
  const auto ticket = request.ticket;
  if (ticket == nullptr || ticket->cell == nullptr) {
    std::fprintf(stderr, "Vulkan persistent: ticket not cold\n");
    return false;
  }
  const auto committed_before = ticket->cell->committed;
  std::uint64_t token_queue_before = 0u;
  {
    std::lock_guard lock{test_case.native->adapter->mutex};
    token_queue_before = test_case.native->adapter->command_submit_count;
  }
  accel::PersistentResidencySlidingRequest token_replay = request;
  ++token_replay.token;
  accel::PersistentResidencySlidingControl token_control{};
  const auto token_result = test_case.lowering.backend.service.submit_result(
      token_replay, token_control);
  std::uint64_t token_queue_after = 0u;
  {
    std::lock_guard lock{test_case.native->adapter->mutex};
    token_queue_after = test_case.native->adapter->command_submit_count;
  }
  const auto *const selection = test_case.native->residency.get();
  const bool token_valid =
      !token_result.status.ok && token_result.status.reason != nullptr &&
      std::strcmp(token_result.status.reason,
                  "accel_kernel_pipeline_invalid") == 0 &&
      token_result.event ==
          accel::PersistentResidencySlidingSubmitEvent::Declined &&
      !ticket->active && !ticket->pending.has_value() &&
      accel::persistent_sliding_request_equal(ticket->cell->committed,
                                              committed_before) &&
      PersistentControlPristine(token_control) && selection != nullptr &&
      !selection->persistent.active &&
      selection->persistent.control == nullptr &&
      !selection->persistent.final_sent && !selection->persistent.quarantined &&
      token_queue_after == token_queue_before;
  if (!token_valid) {
    std::fprintf(stderr,
                 "Vulkan persistent: token replay result=%s active=%u "
                 "pending=%u control=%u run=%u/%u/%u queue=%llu/%llu\n",
                 token_result.status.reason,
                 static_cast<unsigned>(ticket->active),
                 static_cast<unsigned>(ticket->pending.has_value()),
                 static_cast<unsigned>(token_control.active),
                 static_cast<unsigned>(selection != nullptr &&
                                       selection->persistent.active),
                 static_cast<unsigned>(selection != nullptr &&
                                       selection->persistent.final_sent),
                 static_cast<unsigned>(selection != nullptr &&
                                       selection->persistent.quarantined),
                 static_cast<unsigned long long>(token_queue_before),
                 static_cast<unsigned long long>(token_queue_after));
    return false;
  }
  accel::PersistentResidencySlidingRequest replay = request;
  replay.plan_identity ^= 1u;
  accel::PersistentResidencySlidingControl replay_control{};
  const auto replay_result =
      test_case.lowering.backend.service.submit_result(replay, replay_control);
  if (replay_result.status.ok ||
      replay_result.event !=
          accel::PersistentResidencySlidingSubmitEvent::Declined) {
    std::fprintf(stderr,
                 "Vulkan persistent: accepted prepared-request replay\n");
    return false;
  }
  {
    std::lock_guard lock{test_case.native->adapter->mutex};
    test_case.queue_before = test_case.native->adapter->command_submit_count;
  }
  if (!accel::InspectVulkanResidencySliding(test_case.primary,
                                            test_case.primary_before)) {
    std::fprintf(stderr, "Vulkan persistent: pre-inspect\n");
    return false;
  }
  if (submit_device_lost && !test_case.native->adapter->device_loss_fault.arm(
                                accel::SubmitKind::Work)) {
    return false;
  }
  return true;
}

} // namespace rund_node_test_pipeline_vulkan_persistent

#endif
