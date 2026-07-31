#include <kernel/internal/schedule/builder.hpp>

#include "../work/units.hpp"
#include "local.hpp"

#include <algorithm>

namespace rund::kernel::internal {

PartitionBuild
BuildBalancedPartitions(std::vector<Partition> &out_partitions,
                        const std::span<const u64> packet_work_units,
                        const PartitionRequest &request) {
  out_partitions.clear();
  if (!schedule::planner::ValidPacketWorkUnits(packet_work_units, request)) {
    return schedule::planner::FailPartitionBuild("invalid_packet_work_units");
  }
  const u32 partition_count =
      schedule::planner::ResolvePartitionCount(request, request.packet_count);
  if (partition_count == 0u) {
    return schedule::planner::FailPartitionBuild("invalid_partition_count");
  }
  if (out_partitions.capacity() < partition_count) {
    out_partitions.reserve(partition_count);
  }
  out_partitions.clear();

  u64 remaining_work = 0u;
  for (const u64 work_units : packet_work_units) {
    remaining_work = rund::math32::detail::ScalarSatAdd(
        remaining_work, schedule_detail::NormalizeWorkUnits(work_units));
  }

  u32 begin = 0u;
  u32 max_packets = 0u;
  for (u32 partition_index = 0u; partition_index < partition_count;
       ++partition_index) {
    const bool is_last_partition = partition_index + 1u == partition_count;
    const u32 remaining_partitions = partition_count - partition_index;
    u32 end = request.packet_count;
    u64 partition_work = 0u;
    if (!is_last_partition) {
      const u32 max_end = request.packet_count - (remaining_partitions - 1u);
      const u64 target_work =
          (remaining_work + remaining_partitions - 1u) / remaining_partitions;
      end = begin;
      while (end < max_end) {
        partition_work = rund::math32::detail::ScalarSatAdd(
            partition_work,
            schedule_detail::NormalizeWorkUnits(packet_work_units[end]));
        end += 1u;
        if (partition_work >= target_work) {
          break;
        }
      }
      remaining_work =
          rund::math32::detail::ScalarSatSub(remaining_work, partition_work);
    }
    if (begin >= end || end > request.packet_count) {
      out_partitions.clear();
      return schedule::planner::FailPartitionBuild(
          "partition_coverage_mismatch");
    }
    max_packets = std::max<u32>(max_packets, end - begin);
    out_partitions.push_back(Partition{
        .worker_index = partition_index,
        .begin = begin,
        .end = end,
    });
    begin = end;
  }
  if (begin != request.packet_count) {
    out_partitions.clear();
    return schedule::planner::FailPartitionBuild("partition_coverage_mismatch");
  }
  return PartitionBuild{
      .ok = true,
      .reason = "pass",
      .partition_count = static_cast<u32>(out_partitions.size()),
      .alignment_packets = 1u,
      .packets_per_partition_max = max_packets,
      .worker_slot_count = std::min<u32>(
          schedule::planner::ClampKernelWidth(request.execution_width),
          static_cast<u32>(out_partitions.size())),
      .fold_slot_count = static_cast<u32>(out_partitions.size()),
      .useful_width = std::min<u32>(
          schedule::planner::ClampKernelWidth(request.execution_width),
          static_cast<u32>(out_partitions.size())),
      .placement = PlacementPolicy::ContiguousBalanced,
  };
}

} // namespace rund::kernel::internal
