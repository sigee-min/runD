#include "../../local.hpp"

#include <algorithm>
#include <limits>
#include <math32/math32.hpp>

namespace rund::kernel::dispatch::detail {
namespace {

void PopulatePartitionSizeTelemetry(const Partition* partitions,
                                    const u32 partition_count,
                                    Telemetry& telemetry) {
  if (partitions == nullptr || partition_count == 0u) {
    return;
  }
  u32 min_size = std::numeric_limits<u32>::max();
  u32 max_size = 0u;
  u64 total_size = 0u;
  for (u32 index = 0u; index < partition_count; ++index) {
    const u32 size = partitions[index].size();
    min_size = std::min<u32>(min_size, size);
    max_size = std::max<u32>(max_size, size);
    total_size += size;
  }
  telemetry.min_partition_size = min_size == std::numeric_limits<u32>::max() ? 0u : min_size;
  telemetry.max_partition_size = max_size;
  const u32 mean_size = static_cast<u32>(total_size / partition_count);
  telemetry.partition_size_imbalance_milli =
      mean_size == 0u || max_size <= mean_size
          ? 0u
          : math32::detail::ScalarSatMilliRatio(max_size - mean_size, mean_size);
  telemetry.min_partition_work_units = telemetry.min_partition_size;
  telemetry.max_partition_work_units = telemetry.max_partition_size;
  telemetry.work_imbalance_milli = telemetry.partition_size_imbalance_milli;
  telemetry.work_imbalance_measured = true;
}

} // namespace

Telemetry BuildBaseTelemetry(const Plan& plan) {
  Telemetry telemetry{
      .packet_count = plan.packet_count,
      .execution_width = plan.execution_width,
      .useful_width = plan.useful_width != 0u
                           ? plan.useful_width
                           : (plan.partition_count < plan.execution_width ? plan.partition_count : plan.execution_width),
      .partition_count = plan.partition_count,
      .worker_slot_count = plan.worker_slot_count != 0u
                               ? plan.worker_slot_count
                               : (plan.partition_count < plan.execution_width ? plan.partition_count : plan.execution_width),
      .fold_slot_count = plan.fold_slot_count != 0u ? plan.fold_slot_count : plan.partition_count,
      .alignment_packets = plan.alignment_packets,
      .packets_per_partition_max = plan.packets_per_partition_max,
      .placement = plan.placement,
      .no_allocation = plan.no_allocation,
  };
  PopulatePartitionSizeTelemetry(plan.partitions, plan.partition_count, telemetry);
  return telemetry;
}

} // namespace rund::kernel::dispatch::detail
