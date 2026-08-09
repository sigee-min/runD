#pragma once

#include <accel/check.hpp>

#include "pipeline.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool
BindMetalScanPlanShape(const rund::kernel::ScanPlan &plan,
                       const RangePrefixExec &prefix_execution,
                       MetalScanEncodeState &state) {
  if (!MetalScanPrefixMatchesPlan(plan, prefix_execution)) {
    return false;
  }
  state.prefix_execution = prefix_execution;
  state.element_count = prefix_execution.stage(0u).element_count;
  state.block_size = plan.block_size;
  state.block_count = MetalScanStageGroups(prefix_execution, 0u);
  state.block_threads = static_cast<NSUInteger>(prefix_execution.width());
  state.prefix_threads = static_cast<NSUInteger>(prefix_execution.width());
  return true;
}

[[nodiscard]] inline rund::AccelCheck
CheckMetalScanThreadShape(MetalAdapter &adapter,
                          const MetalScanEncodeState &state) {
  if (!state.prefix_execution.has_value() ||
      MetalScanPipelineCount(*state.prefix_execution) == 0u ||
      state.block_count == 0u || state.block_size > kMetalScanMaxBlockSize ||
      state.block_threads == 0u ||
      state.block_threads > [state.block maxTotalThreadsPerThreadgroup] ||
      (ScanPrefixHasOffset(*state.prefix_execution) &&
       (state.block_threads > [state.offset maxTotalThreadsPerThreadgroup] ||
        state.prefix_threads > [state.prefix maxTotalThreadsPerThreadgroup]))) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
