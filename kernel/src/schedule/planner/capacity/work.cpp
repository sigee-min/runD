#include "local.hpp"

#include "../../work/units.hpp"
#include "../local.hpp"

#include <algorithm>

namespace rund::kernel::schedule::planner::capacity {
namespace {

constexpr u64 kRatioScale = 1000000u;

u32 CapacityForSlot(const PartitionRequest& request, const u32 slot) {
  const u32 width = std::max<u32>(1u, request.execution_width);
  return request.worker_capacity_milli[slot % width];
}

u64 WorkAtLimit(const u64 limit, const u32 capacity) {
  const u64 whole = limit / kRatioScale;
  const u64 rem = limit % kRatioScale;
  const u64 base = rund::math32::detail::ScalarSatMul(whole, capacity);
  const u64 extra = rund::math32::detail::ScalarSatMul(rem, capacity) / kRatioScale;
  return rund::math32::detail::ScalarSatAdd(base, extra);
}

u64 PacketWork(const PartitionRequest& request, const u32 packet) {
  return schedule_detail::NormalizeWorkUnits(request.packet_work_units[packet]);
}

u64 TotalWork(const PartitionRequest& request) {
  u64 total = 0u;
  for (u32 packet = 0u; packet < request.packet_count; ++packet) {
    total = rund::math32::detail::ScalarSatAdd(total, PacketWork(request, packet));
  }
  return total;
}

bool ConsumeSlot(const PartitionRequest& request,
                 const u32 slot,
                 const u32 partition_count,
                 const u64 limit,
                 u32& packet) {
  const u32 remaining_slots = partition_count - slot - 1u;
  const u32 max_end = request.packet_count - remaining_slots;
  const u64 allowed = WorkAtLimit(limit, CapacityForSlot(request, slot));
  u64 work = 0u;
  const u32 begin = packet;
  while (packet < max_end) {
    const u64 next = PacketWork(request, packet);
    const u64 candidate = rund::math32::detail::ScalarSatAdd(work, next);
    if (packet != begin && candidate > allowed) {
      break;
    }
    if (candidate > allowed) {
      return false;
    }
    work = candidate;
    packet += 1u;
  }
  return packet != begin;
}

bool Feasible(const PartitionRequest& request,
              const u32 partition_count,
              const u64 limit) {
  u32 packet = 0u;
  for (u32 slot = 0u; slot < partition_count; ++slot) {
    if (!ConsumeSlot(request, slot, partition_count, limit, packet)) {
      return false;
    }
  }
  return packet == request.packet_count;
}

u64 MinimaxLimit(const PartitionRequest& request, const u32 partition_count) {
  u64 low = 0u;
  u64 high = rund::math32::detail::ScalarSatMul(TotalWork(request), kRatioScale);
  while (low < high) {
    const u64 mid = low + ((high - low) / 2u);
    if (Feasible(request, partition_count, mid)) {
      high = mid;
    } else {
      low = mid + 1u;
    }
  }
  return low;
}

} // namespace

bool BuildMinimaxWorkPartitions(std::vector<Partition>& out_partitions,
                                const PartitionRequest& request,
                                const u32 partition_count,
                                u32& out_max_packets) {
  const u64 limit = MinimaxLimit(request, partition_count);
  out_partitions.clear();
  out_max_packets = 0u;
  u32 begin = 0u;
  for (u32 slot = 0u; slot < partition_count; ++slot) {
    u32 end = begin;
    if (!ConsumeSlot(request, slot, partition_count, limit, end)) {
      out_partitions.clear();
      return false;
    }
    out_max_packets = std::max<u32>(out_max_packets, end - begin);
    out_partitions.push_back(Partition{.worker_index = slot, .begin = begin, .end = end});
    begin = end;
  }
  return begin == request.packet_count;
}

} // namespace rund::kernel::schedule::planner::capacity
