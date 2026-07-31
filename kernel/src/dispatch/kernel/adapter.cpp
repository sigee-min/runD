#include "local.hpp"

namespace rund::kernel::dispatch::detail {

Partition MapPartitionPackets(const DispatchAdapter& adapter,
                              const Partition& partition) {
  if (adapter.ordered_packet_indices == nullptr) {
    return partition;
  }
  Partition mapped = partition;
  mapped.packet_indices = adapter.ordered_packet_indices + partition.begin;
  mapped.packet_index_count = partition.size();
  return mapped;
}

void InvokeMappedPartition(void* const raw_context, const Partition& partition) {
  auto* const adapter = static_cast<DispatchAdapter*>(raw_context);
  if (adapter == nullptr || adapter->dispatch == nullptr) {
    return;
  }
  adapter->dispatch(adapter->context, MapPartitionPackets(*adapter, partition));
}

} // namespace rund::kernel::dispatch::detail
