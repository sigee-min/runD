#include "../local.hpp"
#include "plan/worker.hpp"

#include <cstddef>

namespace rund::kernel::dispatch::detail {

Result ValidatePlan(const Plan& plan) {
  if (plan.packet_count == 0u) {
    return FailResult(plan, "invalid_packet_count");
  }
  if (plan.execution_width == 0u) {
    return FailResult(plan, "invalid_execution_width");
  }
  if (plan.partition_count == 0u) {
    return FailResult(plan, "partitions_empty");
  }
  if (plan.partitions == nullptr) {
    return FailResult(plan, "partitions_missing");
  }
  if (plan.dispatch == nullptr) {
    return FailResult(plan, "dispatch_missing");
  }
  if (plan.require_no_allocation && !plan.no_allocation) {
    return FailResult(plan, "no_allocation_contract_mismatch");
  }
  if (WorkerStatsSinksTooSmall(plan)) {
    return FailResult(plan, "worker_stats_capacity_exceeded");
  }
  if (plan.ordered_packet_indices != nullptr && plan.ordered_packet_count != plan.packet_count) {
    return FailResult(plan, "ordered_packet_count_mismatch");
  }
  const bool over_partitioned = plan.partition_count > plan.execution_width;
  const u32 partition_slot_limit = over_partitioned ? plan.partition_count : plan.execution_width;

  u32 expected_begin = 0u;
  for (u32 index = 0u; index < plan.partition_count; ++index) {
    const Partition& partition = plan.partitions[index];
    if (partition.worker_index >= partition_slot_limit) {
      return FailResult(plan, "partition_worker_mismatch");
    }
    if (partition.begin >= partition.end ||
        partition.begin != expected_begin ||
        partition.end > plan.packet_count) {
      return FailResult(plan, "partition_coverage_mismatch");
    }
    expected_begin = partition.end;
  }

  if (expected_begin != plan.packet_count) {
    return FailResult(plan, "partition_coverage_mismatch");
  }
  if (plan.execution_width > 1u || plan.worker_backend) {
    if (!plan.worker_backend) {
      return FailResult(plan, "pool_missing");
    }
    const WorkerBackendCapabilities capabilities = InspectBackend(plan.worker_backend, plan.execution_width);
    if (const char* const failure = ValidateCommonBackendCapabilities(capabilities)) {
      return FailResult(plan, failure);
    }
    if (over_partitioned && !capabilities.supports_claim_free_static_tiles) {
      return FailResult(plan, "partition_count_exceeds_width");
    }
    if (!over_partitioned && !capabilities.supports_claim_free_static_tiles) {
      return FailResult(plan, "static_tile_backend_required");
    }
    if (!over_partitioned && !capabilities.supports_static_partitions) {
      return FailResult(plan, "pool_policy_mismatch");
    }
    if (WorkerStatsWouldAllocate(plan, capabilities)) {
      return FailResult(plan, "worker_stats_would_allocate");
    }
  }

  return Result{
      .ok = true,
      .failure_reason = "pass",
      .telemetry = BuildBaseTelemetry(plan),
  };
}

} // namespace rund::kernel::dispatch::detail
