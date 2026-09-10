#include "internal.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::prepared::sliding {

rund::AccelCheck
admit(const rund::AccelContext &context,
      const std::span<const PreparedResidencySlidingRole> roles,
      const ResidencySlidingMemory memory, Admission &admission) noexcept {
  admission = {};
  BackendResidencySlidingCapability combined{};
  combined.check = {true, "ok"};
  combined.memory = memory;
  combined.max_slots = static_cast<std::uint8_t>(ResidencySlidingCapacity);
  combined.callbacks_async = true;
  combined.descriptor_release_acquire = true;
  combined.retained_bytes = sizeof(State) + sizeof(Service);

  const BackendOps *ops = nullptr;
  for (std::size_t slot = 0u; slot < roles.size(); ++slot) {
    const PreparedResidencySlidingRole &role = roles[slot];
    auto *const pipeline =
        static_cast<prepared::PipelineState *>(role.pipeline.owner.get());
    if (!role.pipeline.ok || role.slot != slot || pipeline == nullptr ||
        role.first_control_generation == 0u ||
        role.control_generation_stride == 0u ||
        !prepared::ValidPipeline(context, *pipeline) ||
        pipeline->ops == nullptr || pipeline->backend == nullptr ||
        pipeline->ops->seed_prepared_pipeline_generation == nullptr ||
        pipeline->ops->submit_prepared_sliding == nullptr ||
        pipeline->ops->prepared_sliding_capability == nullptr ||
        (ops != nullptr && ops != pipeline->ops)) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    for (std::size_t prior = 0u; prior < slot; ++prior) {
      if (admission.pipelines[prior] == pipeline ||
          admission.pipelines[prior]->backend.get() ==
              pipeline->backend.get()) {
        return {false, "accel_kernel_pipeline_invalid"};
      }
    }

    const BackendResidencySlidingCapability capability =
        pipeline->ops->prepared_sliding_capability(pipeline->backend, memory);
    if (!valid_memory(capability, memory) ||
        capability.max_slots < roles.size() ||
        (slot != 0u && !same_capability(combined, capability)) ||
        !::rund::kernel::checked::add(combined.retained_bytes,
                                     capability.retained_bytes,
                                     combined.retained_bytes) ||
        !::rund::kernel::checked::add(combined.transient_bytes,
                                     capability.transient_bytes,
                                     combined.transient_bytes)) {
      return capability.check.ok
                 ? rund::AccelCheck{false, "accel_kernel_pipeline_invalid"}
                 : capability.check;
    }
    if (slot == 0u) {
      combined.memory = capability.memory;
      combined.max_slots = capability.max_slots;
      combined.callbacks_async = capability.callbacks_async;
      combined.descriptor_release_acquire =
          capability.descriptor_release_acquire;
      combined.flush_before_frontier = capability.flush_before_frontier;
      combined.integrated_copy = capability.integrated_copy;
      combined.distinct_transfer_queue = capability.distinct_transfer_queue;
    }
    admission.pipelines[slot] = pipeline;
    ops = pipeline->ops;
  }
  admission.capability = combined;
  admission.role_count = roles.size();
  return {true, "ok"};
}

} // namespace rund::node::accel::detail::prepared::sliding
