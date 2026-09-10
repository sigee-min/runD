#include "../internal.hpp"

#include "../../generated_indirect/internal.hpp"
#include "../../generated_indirect/map.hpp"

namespace rund::node::accel::detail::vulkan_persistent_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

PersistentResidencySlidingSubmitResult
submit_out(const rund::AccelCheck status,
           const PersistentResidencySlidingSubmitEvent event) noexcept {
  return PersistentResidencySlidingSubmitResult{status, event};
}

void reset_control_unlocked(
    PersistentResidencySlidingControl &control) noexcept {
  control.native.reset();
  control.capability = {};
  control.plan_identity = 0u;
  control.token = 0u;
  control.generation = 0u;
  control.coordinate_count = 0u;
  control.mode = PersistentResidencySlidingMode::OneSubmit;
  control.chunk_first = 0u;
  control.chunk_span = 0u;
  control.accepted_end = 0u;
  control.chunk_submit_count = 0u;
  control.next_ready_coordinate = 0u;
  control.next_wait_coordinate = 0u;
  control.next_observation_coordinate = 0u;
  control.next_acknowledgement_coordinate = 0u;
  control.native_submit_count = 0u;
  control.queue_calls = 0u;
  control.epoch_native_submit_count = 0u;
  control.backend_epoch_callback_count = 0u;
  control.backing_wait_count = 0u;
  control.backing_signal_count = 0u;
  control.backing_acknowledgement_count = 0u;
  control.backing_service_failure_count = 0u;
  control.failed_admission_count = 0u;
  control.first_failed_admission_coordinate = PersistentSlidingNoCoordinate;
  control.first_service_failure_coordinate = PersistentSlidingNoCoordinate;
  control.width = 0u;
  control.active = false;
  control.service_failed = false;
  control.quarantined = false;
}

void destroy_attempt_roles(
    const PersistentResidencySlidingRequest &request) noexcept {
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    auto *const pipeline =
        static_cast<VulkanPipeline *>(request.roles[slot].prepared.get());
    if (pipeline != nullptr && pipeline->residency != nullptr) {
      const auto access =
          vulkan_generated_indirect_detail::map_access(*pipeline->residency);
      if (access.valid && access.selected) {
        vulkan_generated_indirect_detail::destroy_roles(*pipeline->residency);
      }
    }
  }
}

void clear_attempt_claims(const PersistentResidencySlidingRequest &request,
                          VulkanResidencyPersistentRun *const run) noexcept {
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    auto *const pipeline =
        static_cast<VulkanPipeline *>(request.roles[slot].prepared.get());
    if (pipeline == nullptr || pipeline->residency == nullptr) {
      continue;
    }
    VulkanResidencyPersistentRun *expected = run;
    static_cast<void>(
        pipeline->residency->active_persistent.compare_exchange_strong(
            expected, nullptr, std::memory_order_acq_rel,
            std::memory_order_acquire));
  }
}

#endif

} // namespace rund::node::accel::detail::vulkan_persistent_detail
