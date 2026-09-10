#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck AcknowledgeVulkanResidencyPersistentDone(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingAcknowledgeDone &acknowledgement) noexcept {
  using namespace vulkan_persistent_detail;
  std::unique_lock control_lock{control.gate};
  VulkanResidencyPersistentRun *const run = active_run(control);
  if (run == nullptr || !control.active || control.quarantined ||
      acknowledgement.identity.plan_identity != control.plan_identity ||
      acknowledgement.identity.token != control.token ||
      acknowledgement.identity.generation != control.generation ||
      acknowledgement.identity.coordinate !=
          control.next_acknowledgement_coordinate ||
      acknowledgement.identity.coordinate >=
          persistent_sliding_accepted_end(control) ||
      acknowledgement.identity.coordinate >=
          control.next_observation_coordinate ||
      (!acknowledgement.service_check.ok &&
       (acknowledgement.service_check.reason == nullptr ||
        acknowledgement.service_check.reason[0] == '\0'))) {
    return invalid();
  }
  PersistentResidencySlidingServiceIdentity expected{};
  if (!persistent_sliding_service_identity(
          control, acknowledgement.identity.coordinate, expected) ||
      !persistent_sliding_same_service_identity(expected,
                                                acknowledgement.identity)) {
    return invalid();
  }
  bool last = false;
  {
    std::lock_guard run_lock{run->gate};
    const std::size_t cell_index = static_cast<std::size_t>(
        acknowledgement.identity.coordinate % VulkanResidencyWindowCapacity);
    VulkanResidencyPersistentCell &cell = run->cells[cell_index];
    if (!run->active || run->final_sent || !cell.observed ||
        cell.acknowledged ||
        cell.coordinate != acknowledgement.identity.coordinate) {
      return invalid();
    }
    cell.acknowledged = true;
    if (!acknowledgement.service_check.ok && control.first_failure.ok) {
      control.first_failure = acknowledgement.service_check;
    }
    const std::uint64_t end = persistent_sliding_accepted_end(control);
    const std::uint64_t acknowledged =
        control.next_acknowledgement_coordinate + 1u;
    last = acknowledged == end &&
           (end == run->request.coordinate_count || control.service_failed ||
            control.failed_admission_count != 0u);
  }
  ++control.next_acknowledgement_coordinate;
  ++control.backing_acknowledgement_count;
  control_lock.unlock();
  if (last) {
    close_known(*run);
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
