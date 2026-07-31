#include "../local.hpp"

namespace rund::kernel::internal {

Result ExecuteSchedule(const SchedulePlan& plan) {
  if (plan.schedule.packet_count == 0u) {
    return dispatch::detail::FailResult(Plan{}, "schedule_missing");
  }
  const Plan kernel_plan{
      .packet_count = plan.schedule.packet_count,
      .execution_width = plan.schedule.execution_width,
      .placement = plan.schedule.placement,
      .alignment_packets = plan.schedule.alignment_packets,
      .packets_per_partition_max = plan.schedule.packets_per_partition_max,
      .worker_slot_count = plan.schedule.worker_slot_count,
      .fold_slot_count = plan.schedule.fold_slot_count,
      .useful_width = plan.schedule.useful_width,
      .no_allocation = plan.schedule.no_allocation,
      .partitions = plan.schedule.partitions,
      .partition_count = plan.schedule.partition_count,
      .ordered_packet_indices = plan.schedule.ordered_packet_indices,
      .ordered_packet_count = plan.schedule.ordered_packet_count,
      .worker_backend = plan.worker_backend,
      .context = plan.context,
      .dispatch = plan.dispatch,
      .collect_worker_stats = plan.collect_worker_stats,
      .worker_stats_sink = plan.worker_stats_sink,
      .worker_start_offset_ns_sink = plan.worker_start_offset_ns_sink,
      .worker_elapsed_ns_sink = plan.worker_elapsed_ns_sink,
      .worker_tail_wait_ns_sink = plan.worker_tail_wait_ns_sink,
      .require_no_allocation = plan.require_no_allocation,
  };
  return Execute(kernel_plan);
}

} // namespace rund::kernel::internal
