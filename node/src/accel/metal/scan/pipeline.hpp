#pragma once

#include "../../scan/metal.hpp"
#include "../../scan/prefix.hpp"
#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr bool
MetalScanPrefixMatchesPlan(const rund::kernel::ScanPlan &plan,
                           const RangePrefixExec &execution) noexcept {
  const RangePrefixExec expected = PlanScanPrefixExecution(plan);
  if (!expected.ok() || !execution.ok() ||
      expected.disposition() != execution.disposition() ||
      expected.width() != execution.width() ||
      expected.stage_count() != execution.stage_count() ||
      expected.temporary_count() != execution.temporary_count()) {
    return false;
  }
  for (std::size_t index = 0u; index < expected.stage_count(); ++index) {
    if (expected.stage(index) != execution.stage(index)) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < expected.temporary_count(); ++index) {
    if (expected.temporary(index) != execution.temporary(index)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr std::size_t
MetalScanPipelineCount(const RangePrefixExec &execution) noexcept {
  return execution.ok() &&
                 execution.disposition() == RangePrefixKind::FlatBlockTotals &&
                 (execution.stage_count() == 1u ||
                  execution.stage_count() == 3u)
             ? execution.stage_count()
             : 0u;
}

[[nodiscard]] constexpr rund::kernel::u64
MetalScanStageGroups(const RangePrefixExec &execution,
                     const std::size_t index) noexcept {
  return MetalScanPipelineCount(execution) != 0u &&
                 index < execution.stage_count()
             ? execution.stage(index).groups
             : 0u;
}

[[nodiscard]] bool CompileMetalScanPipelines(
    MetalAdapter &adapter, rund::kernel::ScanElement element,
    const RangePrefixExec &prefix_execution, std::shared_ptr<void> &block,
    std::shared_ptr<void> &prefix, std::shared_ptr<void> &offset);

// Compact retains its fixed executable tuple. Native Scan and Partition use
// the RangePrefixExec overload above as their sole tuple-cardinality authority.
[[nodiscard]] bool CompileMetalScanPipelines(MetalAdapter &adapter,
                                             rund::kernel::ScanElement element,
                                             std::shared_ptr<void> &block,
                                             std::shared_ptr<void> &prefix,
                                             std::shared_ptr<void> &offset);

[[nodiscard]] bool CompileMetalScanFlagPipelines(MetalAdapter &adapter,
                                                 std::shared_ptr<void> &block,
                                                 std::shared_ptr<void> &prefix,
                                                 std::shared_ptr<void> &offset);

} // namespace rund::node::accel::detail
