#include "../internal.hpp"

#include <new>

namespace rund::compute::detail::sliding_product_detail {

Status allocate_preparation_owner(
    const residency::execution::Plan &plan, storage::Reservation &&capacity,
    std::shared_ptr<SlidingProductOwner> &owner) noexcept {
  try {
    owner = std::make_shared<SlidingProductOwner>();
    owner->capacity = std::move(capacity);
    owner->run = std::make_shared<SlidingProductRun>();
    owner->plan = plan;
    owner->plan_owner =
        std::make_shared<const residency::execution::Plan>(owner->plan);
  } catch (const std::bad_alloc &) {
    owner.reset();
    return Status::fail(Reason::PipelineMemoryBudget);
  }
  return Status::success();
}

Status snapshot_preparation_pipelines(VirtualPipelineState &state,
                                      SlidingProductOwner &owner) noexcept {
  owner.pipelines = {state.pipeline, state.alternate_pipeline};
  return snapshot_pipeline_execution(owner.pipelines, owner.snapshot);
}

} // namespace rund::compute::detail::sliding_product_detail
