#include "eight.hpp"

namespace rund::kernel::workspace_detail {

void AssignEightPartitions(Workspace &workspace,
                           const std::span<const u64> resolved_work_units) {
  u64 load0 = 0u;
  u64 load1 = 0u;
  u64 load2 = 0u;
  u64 load3 = 0u;
  u64 load4 = 0u;
  u64 load5 = 0u;
  u64 load6 = 0u;
  u64 load7 = 0u;
  u32 count0 = 0u;
  u32 count1 = 0u;
  u32 count2 = 0u;
  u32 count3 = 0u;
  u32 count4 = 0u;
  u32 count5 = 0u;
  u32 count6 = 0u;
  u32 count7 = 0u;

  for (const u32 packet_index : workspace.ordered_packet_indices) {
    u32 partition = 0u;
    u64 min_load = load0;
    if (load1 < min_load) {
      min_load = load1;
      partition = 1u;
    }
    if (load2 < min_load) {
      min_load = load2;
      partition = 2u;
    }
    if (load3 < min_load) {
      min_load = load3;
      partition = 3u;
    }
    if (load4 < min_load) {
      min_load = load4;
      partition = 4u;
    }
    if (load5 < min_load) {
      min_load = load5;
      partition = 5u;
    }
    if (load6 < min_load) {
      min_load = load6;
      partition = 6u;
    }
    if (load7 < min_load) {
      partition = 7u;
    }

    workspace.packet_partition_indices[packet_index] = partition;
    const u64 packet_work = resolved_work_units[packet_index];
    switch (partition) {
    case 0u:
      ++count0;
      load0 = rund::math32::detail::ScalarSatAdd(load0, packet_work);
      break;
    case 1u:
      ++count1;
      load1 = rund::math32::detail::ScalarSatAdd(load1, packet_work);
      break;
    case 2u:
      ++count2;
      load2 = rund::math32::detail::ScalarSatAdd(load2, packet_work);
      break;
    case 3u:
      ++count3;
      load3 = rund::math32::detail::ScalarSatAdd(load3, packet_work);
      break;
    case 4u:
      ++count4;
      load4 = rund::math32::detail::ScalarSatAdd(load4, packet_work);
      break;
    case 5u:
      ++count5;
      load5 = rund::math32::detail::ScalarSatAdd(load5, packet_work);
      break;
    case 6u:
      ++count6;
      load6 = rund::math32::detail::ScalarSatAdd(load6, packet_work);
      break;
    default:
      ++count7;
      load7 = rund::math32::detail::ScalarSatAdd(load7, packet_work);
      break;
    }
  }

  workspace.partition_loads[0u] = load0;
  workspace.partition_loads[1u] = load1;
  workspace.partition_loads[2u] = load2;
  workspace.partition_loads[3u] = load3;
  workspace.partition_loads[4u] = load4;
  workspace.partition_loads[5u] = load5;
  workspace.partition_loads[6u] = load6;
  workspace.partition_loads[7u] = load7;
  workspace.partition_counts[0u] = count0;
  workspace.partition_counts[1u] = count1;
  workspace.partition_counts[2u] = count2;
  workspace.partition_counts[3u] = count3;
  workspace.partition_counts[4u] = count4;
  workspace.partition_counts[5u] = count5;
  workspace.partition_counts[6u] = count6;
  workspace.partition_counts[7u] = count7;
}

} // namespace rund::kernel::workspace_detail
