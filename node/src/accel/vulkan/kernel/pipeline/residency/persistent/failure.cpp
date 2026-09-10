#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace vulkan_persistent_detail {

void quarantine_unknown_locked(
    PersistentResidencySlidingControl &control, Owner *const owner,
    VulkanResidencyPersistentRun *const run, VulkanAdapter *const adapter,
    const rund::AccelCheck failure) noexcept {
  persistent_sliding_mark_unknown_locked(control, failure);
  if (owner != nullptr) {
    owner->quarantine_ticket = true;
  }
  if (run != nullptr) {
    run->quarantined = true;
  }
  if (adapter != nullptr) {
    adapter->residency_quarantined.store(true, std::memory_order_release);
  }
  const PersistentResidencySlidingRequest *request = nullptr;
  if (run != nullptr && run->request.width != 0u) {
    request = &run->request;
  } else if (owner != nullptr) {
    request = &owner->prepared;
  }
  if (request == nullptr) {
    return;
  }
  persistent_sliding_quarantine_ticket(request->ticket);
  for (std::size_t slot = 0u; slot < request->width; ++slot) {
    auto *const pipeline =
        static_cast<VulkanPipeline *>(request->roles[slot].prepared.get());
    if (pipeline == nullptr || pipeline->residency == nullptr) {
      continue;
    }
    pipeline->residency->sliding.quarantined.store(
        true, std::memory_order_release);
    if (pipeline->adapter != nullptr) {
      pipeline->adapter->residency_quarantined.store(
          true, std::memory_order_release);
    }
  }
}

} // namespace vulkan_persistent_detail

void QuarantineVulkanResidencyPersistentUnknown(
    PersistentResidencySlidingControl &control) noexcept {
  using namespace vulkan_persistent_detail;
  std::unique_lock control_lock{control.gate};
  const std::shared_ptr<Owner> owner = owner_of(control.native);
  VulkanResidencyPersistentRun *const run = active_run(control);
  std::unique_lock<std::mutex> run_lock;
  if (run != nullptr) {
    run_lock = std::unique_lock{run->gate};
  }
  VulkanPipeline *pipeline = nullptr;
  if (run != nullptr && run->request.width != 0u) {
    pipeline = static_cast<VulkanPipeline *>(
        run->request.roles[0u].prepared.get());
  } else if (owner != nullptr && owner->prepared.width != 0u) {
    pipeline = static_cast<VulkanPipeline *>(
        owner->prepared.roles[0u].prepared.get());
  }
  std::unique_lock<std::mutex> adapter_lock;
  if (pipeline != nullptr && pipeline->adapter != nullptr) {
    adapter_lock = std::unique_lock{pipeline->adapter->mutex};
  }
  quarantine_unknown_locked(control, owner.get(), run,
                            pipeline == nullptr ? nullptr : pipeline->adapter,
                            rund::AccelCheck{false, "compute_device_lost"});
  control.active = false;
}

rund::AccelCheck FailVulkanResidencyPersistentService(
    PersistentResidencySlidingControl &control,
    const PersistentResidencySlidingServiceFailure &failure) noexcept {
  using namespace vulkan_persistent_detail;
  if (failure.check.ok || failure.check.reason == nullptr ||
      failure.check.reason[0] == '\0') {
    return invalid();
  }
  std::unique_lock control_lock{control.gate};
  const std::shared_ptr<Owner> owner = owner_of(control.native);
  VulkanResidencyPersistentRun *const run = active_run(control);
  if (run == nullptr || !control.active || control.service_failed ||
      control.quarantined ||
      failure.identity.plan_identity != control.plan_identity ||
      failure.identity.token != control.token ||
      failure.identity.generation != control.generation) {
    return invalid();
  }
  PersistentResidencySlidingServiceIdentity expected{};
  if (!persistent_sliding_service_identity(control, failure.identity.coordinate,
                                           expected) ||
      !persistent_sliding_same_service_identity(expected, failure.identity)) {
    return invalid();
  }
  const bool next_input =
      failure.identity.coordinate == control.next_ready_coordinate;
  const bool completed_output =
      control.next_observation_coordinate != 0u &&
      failure.identity.coordinate + 1u == control.next_observation_coordinate;
  if (!next_input && !completed_output) {
    return invalid();
  }
  control.service_failed = true;
  const bool unknown =
      failure.terminal == NativeTerminal::UnknownMayWrite;
  control.first_service_failure_coordinate = failure.identity.coordinate;
  ++control.backing_service_failure_count;
  {
    std::unique_lock run_lock{run->gate};
    VulkanPipeline *pipeline = nullptr;
    if (run->request.width != 0u) {
      pipeline = static_cast<VulkanPipeline *>(
          run->request.roles[0u].prepared.get());
    }
    std::unique_lock<std::mutex> adapter_lock;
    if (pipeline != nullptr && pipeline->adapter != nullptr) {
      adapter_lock = std::unique_lock{pipeline->adapter->mutex};
    }
    if (unknown) {
      quarantine_unknown_locked(
          control, owner.get(), run,
          pipeline == nullptr ? nullptr : pipeline->adapter, failure.check);
    }
    if (control.first_failure.ok) {
      control.first_failure = failure.check;
    }
  }
  control_lock.unlock();
  if (unknown) {
    publish_final(*run);
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
