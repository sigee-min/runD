#pragma once

#include <kernel/program/compute/scan/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr rund::kernel::u64
EncodedScanDispatchCount(const rund::kernel::ScanPlan &plan) noexcept {
  return plan.pass_count == 2u ? 3u : (plan.pass_count == 1u ? 1u : 0u);
}

[[nodiscard]] constexpr rund::kernel::u64
EncodedScanDeferredOffsetDispatchCount(
    const rund::kernel::ScanPlan &plan) noexcept {
  return plan.pass_count == 2u ? 2u : (plan.pass_count == 1u ? 1u : 0u);
}

} // namespace rund::node::accel::detail
