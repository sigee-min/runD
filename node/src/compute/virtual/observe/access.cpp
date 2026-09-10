#include "local.hpp"

#include <rund/compute/pipeline/runtime.hpp>

#include <mutex>

namespace rund::compute::detail {

Stats virtual_pipeline_stats(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return state->stats;
}

PipelinePlan virtual_pipeline_plan(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept {
  if (state == nullptr) {
    return {};
  }
  std::lock_guard lock{state->gate};
  return pipeline_plan(state->pipeline);
}

} // namespace rund::compute::detail
