#include <kernel/schedule/planner/view.hpp>

namespace rund::kernel {

ScheduleView ViewSchedule(const Schedule& schedule) {
  return ScheduleView{
      .packet_count = schedule.packet_count,
      .execution_width = schedule.execution_width,
      .intent = schedule.intent,
      .placement = schedule.placement,
      .alignment_packets = schedule.alignment_packets,
      .packets_per_partition_max = schedule.packets_per_partition_max,
      .worker_slot_count = schedule.worker_slot_count,
      .fold_slot_count = schedule.fold_slot_count,
      .useful_width = schedule.useful_width,
      .no_allocation = schedule.no_allocation,
      .partitions = schedule.partitions.data(),
      .partition_count = static_cast<u32>(schedule.partitions.size()),
      .ordered_packet_indices = nullptr,
      .ordered_packet_count = 0u,
  };
}

} // namespace rund::kernel
