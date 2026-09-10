#include "internal.hpp"

#include "../generated_indirect/internal.hpp"
#include "../generated_indirect/map.hpp"
#include "../mode.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace vulkan_persistent_detail {

PersistentResidencySlidingSubmitResult
submit_result(const PersistentResidencySlidingRequest &request,
              PersistentResidencySlidingControl &control) noexcept {
  for (std::size_t slot = 0u; slot < request.width; ++slot) {
    const auto *const pipeline =
        static_cast<const VulkanPipeline *>(request.roles[slot].prepared.get());
    if (pipeline != nullptr && pipeline->residency != nullptr) {
      const auto access =
          vulkan_generated_indirect_detail::map_access(*pipeline->residency);
      if (!access.valid ||
          (access.selected && (!access.active || !access.descriptor_ready))) {
        return submit_out(invalid());
      }
    }
  }
  const std::shared_ptr<Owner> owner = owner_of(request.lowering);
  const PersistentResidencySlidingCapability capability = capability_for(
      std::span<const PersistentResidencySlidingRole>{request.roles.data(),
                                                      request.width},
      request.coordinate_count, request.memory, request.mode);
  if (owner == nullptr ||
      !persistent_sliding_request_valid(capability, request)) {
    return submit_out(invalid());
  }
  if (request.mode == PersistentResidencySlidingMode::BackendChunked &&
      !persistent_sliding_chunk_valid(capability, request)) {
    return submit_out(invalid());
  }
  auto *const first_pipeline =
      static_cast<VulkanPipeline *>(request.roles[0u].prepared.get());
  VulkanResidencySelection *const first_selection =
      first_pipeline == nullptr ? nullptr : first_pipeline->residency.get();
  if (!ValidVulkanPipeline(first_pipeline) || first_selection == nullptr ||
      first_pipeline->adapter == nullptr) {
    return submit_out(invalid());
  }
  VulkanAdapter &adapter = *first_pipeline->adapter;
  VulkanResidencyPersistentRun &run = first_selection->persistent;
  std::unique_lock control_lock{control.gate, std::try_to_lock};
  if (!control_lock.owns_lock()) {
    return submit_out(invalid());
  }
  std::unique_lock run_lock{run.gate, std::try_to_lock};
  if (!run_lock.owns_lock()) {
    return submit_out({false, "compute_pipeline_busy"});
  }
  const bool first_submit = !run.active;
  if (first_submit && !stage_rearm(*owner, request)) {
    return submit_out(invalid());
  }
  if (first_submit && !persistent_sliding_activate_control_locked(
                          capability, request, control)) {
    abort_rearm(*owner);
    return submit_out(invalid());
  }
  if (!first_submit && !persistent_sliding_accept_chunk_locked(
                           owner->capability, request, control)) {
    return submit_out(invalid());
  }
  if ((first_submit ? run.active : !run.active) || run.quarantined ||
      owner->run != &run || owner->first != request.roles[0u].prepared.get()) {
    abort_rearm(*owner);
    if (first_submit && control.accepted_end == 0u) {
      reset_control_unlocked(control);
    }
    return submit_out({false, "compute_pipeline_busy"});
  }
  std::unique_lock adapter_lock{adapter.mutex};
  if (adapter.residency_quarantined.load(std::memory_order_acquire)) {
    abort_rearm(*owner);
    if (first_submit && control.accepted_end == 0u) {
      reset_control_unlocked(control);
    }
    return submit_out({false, "compute_device_lost"});
  }
  if (!first_submit) {
    return submit_continuation(request, control, *owner, adapter, run);
  }
  return submit_initial(request, control, capability, *owner, adapter, run);
}

} // namespace vulkan_persistent_detail

#endif

} // namespace rund::node::accel::detail
