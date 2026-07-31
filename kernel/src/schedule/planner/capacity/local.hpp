#pragma once

#include <kernel/internal/schedule/builder.hpp>

namespace rund::kernel::schedule::planner::capacity {

bool BuildMinimaxUnitWidths(std::vector<Partition>& out_widths,
                            const PartitionRequest& request,
                            u32 partition_count,
                            u32 partition_units);
bool BuildMinimaxWorkPartitions(std::vector<Partition>& out_partitions,
                                const PartitionRequest& request,
                                u32 partition_count,
                                u32& out_max_packets);

} // namespace rund::kernel::schedule::planner::capacity
