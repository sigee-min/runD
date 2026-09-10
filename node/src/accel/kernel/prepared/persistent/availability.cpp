#include "../interface/api.hpp"

#include "../model.hpp"

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool
valid_mode(const PersistentResidencySlidingMode mode) noexcept {
  return mode == PersistentResidencySlidingMode::OneSubmit ||
         mode == PersistentResidencySlidingMode::BackendChunked;
}

[[nodiscard]] bool valid_memory(const ResidencySlidingMemory memory) noexcept {
  return memory == ResidencySlidingMemory::HostCoherent ||
         memory == ResidencySlidingMemory::NoncoherentIntegratedCopy ||
         memory == ResidencySlidingMemory::NoncoherentSplitTransfer;
}

[[nodiscard]] PersistentResidencySlidingCapability
failure(const ResidencySlidingMemory memory,
        const PersistentResidencySlidingMode mode,
        const char *const reason = "accel_kernel_pipeline_invalid") noexcept {
  return PersistentResidencySlidingCapability{
      .check = {false, reason}, .memory = memory, .mode = mode};
}

[[nodiscard]] const prepared::PipelineState *
state_of(const PreparedKernelPipeline &pipeline) noexcept {
  return pipeline.ok && pipeline.owner != nullptr
             ? static_cast<const prepared::PipelineState *>(
                   pipeline.owner.get())
             : nullptr;
}

} // namespace

PersistentResidencySlidingCapability
QueryPreparedKernelPipelinePersistentSlidingCapability(
    const std::span<const PreparedResidencyPersistentSlidingRole> roles,
    const std::uint64_t coordinate_count, const ResidencySlidingMemory memory,
    const PersistentResidencySlidingMode mode) noexcept {
  if (roles.empty() || roles.size() > PersistentResidencySlidingCapacity ||
      coordinate_count == 0u || !valid_memory(memory) || !valid_mode(mode)) {
    return failure(memory, mode);
  }
  const prepared::PipelineState *const first = state_of(roles[0u].pipeline);
  if (first == nullptr || first->backend == nullptr) {
    return failure(memory, mode);
  }
  if (first->ops == nullptr) {
    return failure(memory, mode);
  }
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    const PreparedResidencyPersistentSlidingRole &role = roles[slot];
    const prepared::PipelineState *const state = state_of(role.pipeline);
    if (state == nullptr || state->ops != first->ops ||
        state->backend == nullptr || role.slot != slot ||
        role.local_count == 0u ||
        role.local_count > ResidencyWindowLocalCapacity ||
        role.first_control_generation == 0u ||
        role.control_generation_stride == 0u ||
        role.first_descriptor_generation == 0u ||
        role.descriptor_generation_stride == 0u ||
        role.pipeline.owner == nullptr) {
      return failure(memory, mode);
    }
    for (std::size_t prior = 0u; prior < slot; ++prior) {
      if (roles[prior].pipeline.owner.get() == role.pipeline.owner.get()) {
        return failure(memory, mode);
      }
    }
  }
  if (first->ops->query_persistent_sliding_capability == nullptr ||
      first->ops->prepare_persistent_sliding == nullptr) {
    // A valid prepared backend that omits a product hook explicitly declines
    // this route. It is different from a malformed PipelineState above.
    return failure(memory, mode, "compute_backend_unsupported");
  }
  const PersistentResidencySlidingCapability capability =
      first->ops->query_persistent_sliding_capability(roles, coordinate_count,
                                                      memory, mode);
  if (capability.mode != mode || capability.memory != memory) {
    // Backend mode/memory disagreement is a malformed capability result, not
    // a clean unsupported probe. Keep the requested shape visible to callers.
    return failure(memory, mode);
  }
  if (capability.check.ok &&
      (capability.width != roles.size() ||
       !persistent_sliding_product_capable(capability))) {
    return failure(memory, mode);
  }
  if (!capability.check.ok && capability.check.reason == nullptr) {
    return failure(memory, mode);
  }
  return capability;
}

} // namespace rund::node::accel::detail
