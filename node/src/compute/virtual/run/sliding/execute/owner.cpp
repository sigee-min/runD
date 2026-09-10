#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

std::shared_ptr<SlidingProductOwner> validate_execution_owner(
    VirtualPipelineState &state,
    const VirtualExecutionSlidingPrepared &prepared) noexcept {
  const auto owner =
      std::static_pointer_cast<SlidingProductOwner>(prepared.owner);
  if (!prepared || owner == nullptr || owner->run == nullptr ||
      owner->plan_owner == nullptr ||
      owner->plan.identity() != prepared.plan.identity() ||
      state.pipeline == nullptr || state.alternate_pipeline == nullptr ||
      state.pipeline->device == nullptr ||
      state.pipeline->device->ops == nullptr ||
      state.pipeline->residency_pool == nullptr ||
      state.pipeline->residency_pool !=
          state.alternate_pipeline->residency_pool) {
    return {};
  }
  return owner;
}

} // namespace rund::compute::detail::sliding_product_detail
