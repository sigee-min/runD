#pragma once

#include <accel/check.hpp>

#include "pipeline.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void BindMetalScanPlanShape(const rund::kernel::ScanPlan &plan,
                                   MetalScanEncodeState &state) {
  state.element_count = plan.element_count;
  state.block_size = plan.block_size;
  state.block_count = plan.block_count;
  state.block_threads = static_cast<NSUInteger>(kMetalScanWidth);
  state.prefix_threads = static_cast<NSUInteger>(kMetalScanWidth);
}

[[nodiscard]] inline rund::AccelCheck
CheckMetalScanThreadShape(MetalAdapter &adapter,
                          const rund::kernel::ScanPlan &plan,
                          const MetalScanEncodeState &state) {
  if (state.block_size > kMetalScanMaxBlockSize || state.block_threads == 0u ||
      state.block_threads > [state.block maxTotalThreadsPerThreadgroup] ||
      (plan.pass_count == 2u &&
       (state.block_threads > [state.offset maxTotalThreadsPerThreadgroup] ||
        state.prefix_threads >
            [state.prefix maxTotalThreadsPerThreadgroup]))) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
