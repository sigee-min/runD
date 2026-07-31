#include "local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

#include <chrono>

namespace rund::kernel {

FoldResult FoldStrictOrderedSlots(const std::span<const u64> values,
                                  const FoldOperation operation,
                                  const StrictFloatReductionPolicy policy) {
  const auto start = std::chrono::steady_clock::now();
  if (!IsSupportedFoldOperation(operation)) {
    const auto end = std::chrono::steady_clock::now();
    return FoldResult{
        .ok = false,
        .reason = "unsupported_fold_operation",
        .slot_count = static_cast<u32>(values.size()),
        .operation = operation,
        .fold_cost_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
        .fold_cost_measured = true,
    };
  }
  if (FoldOperationAllowsFloatingPoint(operation) && !StrictFloatReductionValid(operation, policy)) {
    const auto end = std::chrono::steady_clock::now();
    return FoldResult{
        .ok = false,
        .reason = policy.valid ? "floating_point_law_mismatch" : "floating_point_fold_forbidden",
        .slot_count = static_cast<u32>(values.size()),
        .operation = operation,
        .fold_cost_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
        .fold_cost_measured = true,
    };
  }
  if (values.empty()) {
    const auto end = std::chrono::steady_clock::now();
    return FoldResult{
        .ok = true,
        .reason = "pass",
        .value = 0u,
        .slot_count = 0u,
        .operation = operation,
        .fold_cost_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
        .fold_cost_measured = true,
    };
  }
  FoldGraph graph{};
  const FoldGraphBuild build = BuildFoldGraph(graph,
                                              static_cast<u32>(values.size()),
                                              operation,
                                              policy,
                                              AllocationPolicy::AllowGrowth);
  if (!build.ok) {
    const auto end = std::chrono::steady_clock::now();
    return FoldResult{
        .ok = false,
        .reason = build.reason,
        .slot_count = static_cast<u32>(values.size()),
        .operation = operation,
        .fold_cost_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
        .fold_cost_measured = true,
    };
  }
  FoldSlots scratch{};
  const FoldResult result = FoldGraphReduce(ViewFoldGraph(graph), values, scratch, AllocationPolicy::AllowGrowth);
  const auto end = std::chrono::steady_clock::now();
  return FoldResult{
      .ok = result.ok,
      .reason = result.reason,
      .value = result.value,
      .slot_count = static_cast<u32>(values.size()),
      .operation = operation,
      .fold_cost_ns = static_cast<u64>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
      .fold_cost_measured = true,
  };
}

FoldResult FoldOrderedSlots(const std::span<const u64> values, const FoldOperation operation) {
  return FoldStrictOrderedSlots(values, operation, StrictFloatReductionPolicy{});
}

} // namespace rund::kernel
