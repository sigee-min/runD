#include "heap.hpp"

#include <algorithm>
#include <vector>

namespace rund::kernel::workspace_detail {
namespace {

bool PartitionLoadHeapLess(const std::vector<u64>& partition_loads,
                           const u32 lhs_partition,
                           const u32 rhs_partition) {
  const u64 lhs_load = partition_loads[lhs_partition];
  const u64 rhs_load = partition_loads[rhs_partition];
  if (lhs_load != rhs_load) {
    return lhs_load < rhs_load;
  }
  return lhs_partition < rhs_partition;
}

void SiftDownPartitionLoadHeap(std::vector<u32>& partition_heap,
                               const std::vector<u64>& partition_loads,
                               const u32 heap_size,
                               u32 root) {
  while (true) {
    const u32 left = root * 2u + 1u;
    if (left >= heap_size) {
      return;
    }
    const u32 right = left + 1u;
    u32 least_child = left;
    if (right < heap_size &&
        PartitionLoadHeapLess(partition_loads, partition_heap[right], partition_heap[left])) {
      least_child = right;
    }
    if (!PartitionLoadHeapLess(partition_loads, partition_heap[least_child], partition_heap[root])) {
      return;
    }
    std::swap(partition_heap[root], partition_heap[least_child]);
    root = least_child;
  }
}

void InitializePartitionLoadHeap(std::vector<u32>& partition_heap,
                                 const std::vector<u64>& partition_loads) {
  const u32 heap_size = static_cast<u32>(partition_heap.size());
  for (u32 partition = 0u; partition < heap_size; ++partition) {
    partition_heap[partition] = partition;
  }
  for (u32 parent = heap_size / 2u; parent > 0u; --parent) {
    SiftDownPartitionLoadHeap(partition_heap, partition_loads, heap_size, parent - 1u);
  }
}

} // namespace

void AssignWeightedPacketPartitionsHeap(Workspace& workspace,
                                        const std::span<const u64> resolved_work_units) {
  std::vector<u32>& partition_heap = workspace.partition_write_offsets;
  InitializePartitionLoadHeap(partition_heap, workspace.partition_loads);
  const u32 heap_size = static_cast<u32>(partition_heap.size());

  for (u32 ordered_index = 0u; ordered_index < workspace.ordered_packet_indices.size(); ++ordered_index) {
    const u32 packet_index = workspace.ordered_packet_indices[ordered_index];
    const u32 partition = partition_heap[0u];
    workspace.packet_partition_indices[packet_index] = partition;
    workspace.partition_counts[partition] += 1u;
    workspace.partition_loads[partition] =
        rund::math32::detail::ScalarSatAdd(workspace.partition_loads[partition], resolved_work_units[packet_index]);
    SiftDownPartitionLoadHeap(partition_heap, workspace.partition_loads, heap_size, 0u);
  }
}

} // namespace rund::kernel::workspace_detail
