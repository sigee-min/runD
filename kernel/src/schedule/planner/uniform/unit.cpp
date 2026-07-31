#include "local.hpp"

#include <algorithm>

namespace rund::kernel::internal::uniform_detail {

void BuildUnitPartitions(std::vector<Partition>& out_partitions,
                         const u32 partition_units,
                         const u32 partition_count,
                         const u32 alignment_packets,
                         const AllocationPolicy allocation) {
  out_partitions.clear();
  if (partition_units == 0u || partition_count == 0u) {
    return;
  }
  if (allocation == AllocationPolicy::AllowGrowth && out_partitions.capacity() < partition_count) {
    out_partitions.reserve(partition_count);
  }
  out_partitions.resize(partition_count);
  BuildUnitPartitionsInto(out_partitions.data(), partition_units, partition_count, alignment_packets);
}

void BuildUnitPartitionsInto(Partition* const out_partitions,
                             const u32 partition_units,
                             const u32 partition_count,
                             const u32 alignment_packets) {
  if (out_partitions == nullptr || partition_units == 0u || partition_count == 0u) {
    return;
  }
  const u32 base = partition_units / partition_count;
  const u32 remainder = partition_units % partition_count;
  u32 unit_begin = 0u;
  for (u32 slot = 0u; slot < partition_count; ++slot) {
    const u32 unit_width = base + (slot < remainder ? 1u : 0u);
    const u32 unit_end = unit_begin + unit_width;
    out_partitions[slot] = Partition{
        .worker_index = slot,
        .begin = unit_begin * alignment_packets,
        .end = unit_end * alignment_packets,
    };
    unit_begin = unit_end;
  }
}

bool ValidatePartitionCoverage(const Partition* const partitions,
                               const u32 partition_count,
                               const u32 packet_count,
                               u32& max_packets) {
  if (partitions == nullptr || partition_count == 0u ||
      partitions[0].begin != 0u || partitions[partition_count - 1u].end != packet_count) {
    return false;
  }
  max_packets = 0u;
  u32 expected_begin = 0u;
  for (u32 index = 0u; index < partition_count; ++index) {
    const Partition& partition = partitions[index];
    if (partition.begin != expected_begin || partition.begin >= partition.end ||
        partition.end > packet_count) {
      return false;
    }
    max_packets = std::max<u32>(max_packets, partition.end - partition.begin);
    expected_begin = partition.end;
  }
  return true;
}

} // namespace rund::kernel::internal::uniform_detail
