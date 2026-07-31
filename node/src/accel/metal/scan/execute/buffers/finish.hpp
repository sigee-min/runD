#pragma once

#include <accel/check.hpp>

#include "../../../../scan/count.hpp"
#include "encode.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck FinishMetalScanDirectBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanPlan &plan,
    const bool record_dispatches, const MetalScanDirectBuffers &buffers,
    const CommandRun &command, const rund::AccelCheck encoded) {
  const rund::AccelCheck submit = FinishCommand(adapter, command, encoded);
  if (submit.ok && record_dispatches) {
    RecordMetalDispatches(adapter, EncodedScanDispatchCount(plan));
  }
  if (!submit.ok) {
    return submit;
  }
  const rund::kernel::u32 flags = MetalScanStatusFlags(buffers.status);
  if (flags != 0u) {
    const char *const reason = (flags & 2u) != 0u
                                   ? "compute_bounded_count_invalid"
                                   : "compute_scan_sum_overflow";
    SetMetalLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
