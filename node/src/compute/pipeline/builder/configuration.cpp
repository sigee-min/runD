#include <rund/compute/pipeline.hpp>

#include <utility>

namespace rund::compute {

PipelineBuilder &
PipelineBuilder::profile(const PipelineProfile profile) & noexcept {
  detail::configure_pipeline_profile(state_, profile);
  return *this;
}

PipelineBuilder &&
PipelineBuilder::profile(const PipelineProfile profile) && noexcept {
  static_cast<PipelineBuilder &>(*this).profile(profile);
  return std::move(*this);
}

PipelineBuilder &PipelineBuilder::budget(const MemoryBudget limit) & noexcept {
  detail::configure_pipeline_budget(state_, limit);
  return *this;
}

PipelineBuilder &&
PipelineBuilder::budget(const MemoryBudget limit) && noexcept {
  static_cast<PipelineBuilder &>(*this).budget(limit);
  return std::move(*this);
}

Result<PipelinePlan> PipelineBuilder::plan() const noexcept {
  return detail::plan_pipeline(state_);
}

} // namespace rund::compute
