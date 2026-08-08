#pragma once

#include "../range_aggregate/model.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/model.hpp>

#include <limits>
#include <optional>

namespace rund::node::accel::detail {

// Native Scan's source shape is fixed by its portable hierarchy. This is a
// source capability, not a RangeAggregate candidate policy: Scan keeps its
// visible-prefix and overflow semantics while sharing physical stage and
// temporary derivation with RangeAggregate PrefixDifference.
inline constexpr rund::kernel::u32 kScanPrefixWorkgroupWidth = 128u;

[[nodiscard]] constexpr RangeAggregatePrefixExecution
PlanScanPrefixExecution(const rund::kernel::ScanPlan &plan) noexcept {
  const bool known_element = (plan.element == rund::kernel::ScanElement::U32 &&
                              plan.element_bytes == 4u) ||
                             (plan.element == rund::kernel::ScanElement::U64 &&
                              plan.element_bytes == 8u);
  const bool known_op = plan.op == rund::kernel::ScanOp::ExclusiveSum ||
                        plan.op == rund::kernel::ScanOp::InclusiveSum;
  if (!plan.ok || !known_op || !known_element || plan.element_count == 0u ||
      plan.block_size == 0u || plan.block_count == 0u ||
      (plan.pass_count != 1u && plan.pass_count != 2u)) {
    return PlanRangeAggregateFlatPrefix(0u, 0u, kScanPrefixWorkgroupWidth, 0u);
  }
  const rund::kernel::u64 expected_blocks =
      plan.element_count / plan.block_size +
      static_cast<rund::kernel::u64>(plan.element_count % plan.block_size !=
                                     0u);
  if (expected_blocks != plan.block_count ||
      plan.pass_count != (expected_blocks == 1u ? 1u : 2u)) {
    return PlanRangeAggregateFlatPrefix(0u, 0u, kScanPrefixWorkgroupWidth, 0u);
  }
  rund::kernel::u64 payload_bytes = 0u;
  rund::kernel::u64 totals_bytes = 0u;
  if (plan.element_bytes > std::numeric_limits<rund::kernel::u32>::max() ||
      !rund::kernel::checked::mul(plan.element_count, plan.element_bytes,
                                  payload_bytes) ||
      !rund::kernel::checked::mul(plan.block_count, plan.element_bytes,
                                  totals_bytes) ||
      plan.temp_bytes != payload_bytes) {
    return PlanRangeAggregateFlatPrefix(0u, 0u, kScanPrefixWorkgroupWidth, 0u);
  }
  const RangeAggregatePrefixExecution prefix = PlanRangeAggregateFlatPrefix(
      plan.element_count, plan.block_count, kScanPrefixWorkgroupWidth,
      static_cast<rund::kernel::u32>(plan.element_bytes));
  if (!prefix.ok() ||
      prefix.disposition() !=
          RangeAggregatePrefixDisposition::FlatBlockTotals ||
      prefix.stage_count() != (plan.pass_count == 1u ? 1u : 3u) ||
      prefix.stage(0u).groups != plan.block_count ||
      prefix.temporary_count() != 1u ||
      prefix.temporary(0u).bytes != totals_bytes) {
    return PlanRangeAggregateFlatPrefix(0u, 0u, kScanPrefixWorkgroupWidth, 0u);
  }
  return prefix;
}

[[nodiscard]] constexpr std::optional<rund::kernel::u64>
ScanPrefixTotalsBytes(const rund::kernel::ScanPlan &plan) noexcept {
  const RangeAggregatePrefixExecution prefix = PlanScanPrefixExecution(plan);
  return prefix.ok() && prefix.temporary_count() == 1u
             ? std::optional<rund::kernel::u64>{prefix.temporary(0u).bytes}
             : std::nullopt;
}

[[nodiscard]] constexpr std::optional<rund::kernel::u64>
ScanPrefixPayloadBytes(const rund::kernel::ScanPlan &plan) noexcept {
  return PlanScanPrefixExecution(plan).ok()
             ? std::optional<rund::kernel::u64>{plan.temp_bytes}
             : std::nullopt;
}

} // namespace rund::node::accel::detail
