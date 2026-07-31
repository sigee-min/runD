#pragma once

#include <accel/check.hpp>

#include "../state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck CheckMetalScanEncodeInputs(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *const input_buffer,
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const command_encoder) {
  if (adapter.device == nullptr || input_buffer == nullptr ||
      output_buffer == nullptr || totals_buffer == nullptr ||
      status_buffer == nullptr || command_encoder == nullptr) {
    SetMetalLastError(adapter, "accel_metal_unavailable");
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  if (!ScanShapeOk(desc, plan)) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
