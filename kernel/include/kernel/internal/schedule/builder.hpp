#pragma once

#include <kernel/schedule/planner/build.hpp>

namespace rund::kernel::internal {

PartitionBuild BuildPartitions(std::vector<Partition>& out_partitions,
                               const PartitionRequest& request);
PartitionBuild BuildBalancedPartitions(std::vector<Partition>& out_partitions,
                                       std::span<const u64> packet_work_units,
                                       const PartitionRequest& request);
PartitionBuild BuildCapacityWeightedPartitions(std::vector<Partition>& out_partitions,
                                               const PartitionRequest& request);
PartitionBuild BuildSchedule(Schedule& out_schedule,
                             const PartitionRequest& request);
PartitionBuild CompileSchedule(Schedule& out_schedule,
                               const ScheduleCompileRequest& request);

} // namespace rund::kernel::internal
