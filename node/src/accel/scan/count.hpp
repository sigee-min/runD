#pragma once

#include "prefix.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] constexpr rund::kernel::u64
EncodedScanDispatchCount(const rund::kernel::ScanPlan &plan) noexcept {
  const RangePrefixExec prefix = PlanScanPrefixExecution(plan);
  return prefix.ok() ? prefix.stage_count() : 0u;
}

[[nodiscard]] constexpr rund::kernel::u64
EncodedScanDeferredOffsetDispatchCount(
    const rund::kernel::ScanPlan &plan) noexcept {
  const RangePrefixExec prefix = PlanScanPrefixExecution(plan);
  return !prefix.ok() ? 0u : (ScanPrefixHasOffset(prefix) ? 2u : 1u);
}

} // namespace rund::node::accel::detail
