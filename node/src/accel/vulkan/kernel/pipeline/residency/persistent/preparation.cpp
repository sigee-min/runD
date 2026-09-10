#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

const PersistentResidencySlidingServiceOps &
VulkanResidencyPersistentServiceOps() noexcept {
  static const PersistentResidencySlidingServiceOps service{
      .submit_result = vulkan_persistent_detail::submit_result,
      .wait_done = WaitVulkanResidencyPersistentDone,
      .signal_ready = SignalVulkanResidencyPersistentReady,
      .ack_done = AcknowledgeVulkanResidencyPersistentDone,
      .fail_service = FailVulkanResidencyPersistentService,
      .quarantine_unknown = QuarantineVulkanResidencyPersistentUnknown,
  };
  return service;
}

PersistentResidencySlidingPreparation PrepareVulkanResidencyPersistent(
    const PersistentResidencySlidingRequest &request) noexcept {
  PersistentResidencySlidingCapability capability =
      vulkan_persistent_detail::capability_for(
          std::span<const PersistentResidencySlidingRole>{request.roles.data(),
                                                          request.width},
          request.coordinate_count, request.memory, request.mode);
  PersistentResidencySlidingRequest validated = request;
  if (request.width != 0u) {
    validated.lowering = request.roles[0u].prepared;
  }
  if (!persistent_sliding_request_valid(capability, validated)) {
    return PersistentResidencySlidingPreparation{.capability = capability};
  }
  std::shared_ptr<vulkan_persistent_detail::Owner> owner;
  try {
    owner = std::make_shared<vulkan_persistent_detail::Owner>();
  } catch (const std::bad_alloc &) {
    capability.check = {false, "compute_pipeline_capacity"};
    return PersistentResidencySlidingPreparation{.capability = capability};
  }
  owner->capability = capability;
  owner->owner_nonce = request.owner_nonce;
  owner->prepared = request;
  owner->prepared.lowering.reset();
  owner->first =
      static_cast<VulkanPipeline *>(request.roles[0u].prepared.get());
  owner->run = &owner->first->residency->persistent;
  return PersistentResidencySlidingPreparation{
      .capability = capability,
      .service = VulkanResidencyPersistentServiceOps(),
      .lowering = std::move(owner),
      .encoded_coordinate_count = request.coordinate_count,
      .encoded_intermediate_bytes = capability.transient_bytes,
  };
}

#endif

} // namespace rund::node::accel::detail
