#pragma once

#include <accel/check.hpp>

#include <rund/counter.hpp>
#include "command.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
FinishVulkanScanDirect(VulkanAdapter &adapter, VulkanScanDirectState &state,
                       const bool record_dispatches) {
  if (record_dispatches) {
    const auto *const scan =
        static_cast<const VulkanScanEncodeResources *>(state.resources.get());
    ::rund::detail::counter::Accumulate(
        adapter.dispatch_count, scan == nullptr ? 0u : scan->dispatch_count);
  }
  const rund::kernel::u32 flags = VulkanScanStatusFlags(state.status);
  CleanupVulkanScanDirect(adapter, state);
  if (flags != 0u) {
    const char *const reason = (flags & 2u) != 0u
                                   ? "compute_bounded_count_invalid"
                                   : "compute_scan_sum_overflow";
    SetVulkanLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  SetVulkanLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
