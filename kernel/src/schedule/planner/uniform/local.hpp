#pragma once

#include <kernel/internal/schedule/builder.hpp>

#include <vector>

namespace rund::kernel::internal::uniform_detail {

void BuildUnitPartitions(std::vector<Partition>& out_partitions,
                         u32 partition_units,
                         u32 partition_count,
                         u32 alignment_packets,
                         AllocationPolicy allocation);
void BuildUnitPartitionsInto(Partition* out_partitions,
                             u32 partition_units,
                             u32 partition_count,
                             u32 alignment_packets);
bool ValidatePartitionCoverage(const Partition* partitions,
                               u32 partition_count,
                               u32 packet_count,
                               u32& max_packets);

} // namespace rund::kernel::internal::uniform_detail
