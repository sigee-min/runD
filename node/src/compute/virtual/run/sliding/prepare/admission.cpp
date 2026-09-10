#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

bool admit_preparation(VirtualPipelineState &state,
                       const VirtualRunProjection &run,
                       residency::execution::SealResult &sealed,
                       node::accel::detail::PersistentResidencySlidingMode &mode)
    noexcept {
  mode = node::accel::detail::PersistentResidencySlidingMode::OneSubmit;
  sealed = seal_virtual_execution(state, run);
  const bool eligible =
      sealed && sealed.plan.epoch_count() >= 2u &&
      sealed.plan.input_sources_materializable() &&
      state.geometry.route == VirtualRoute::Pointwise &&
      state.pipeline != nullptr && state.alternate_pipeline != nullptr &&
      state.pipeline->device != nullptr &&
      state.pipeline->device == state.alternate_pipeline->device &&
      state.pipeline->device->ops != nullptr;
  if (!eligible) {
    return false;
  }
  const VirtualBufferState *const input = virtual_input(state);
  if (input != nullptr && input->backing != nullptr &&
      VirtualBackingAccess::resident(*input->backing) == nullptr &&
      input->backing->tier() == VirtualBackingTier::Persistent &&
      input->backing->max_parallel_reads() >= 2u) {
    mode = node::accel::detail::PersistentResidencySlidingMode::BackendChunked;
  }
  return true;
}

} // namespace rund::compute::detail::sliding_product_detail
