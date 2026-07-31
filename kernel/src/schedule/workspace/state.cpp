#include "local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

namespace rund::kernel {

void ResetWorkspace(Workspace& workspace) {
  // Reset the schedule metadata without replacing its owning vectors.  A
  // prepared bounded collective recompiles the schedule for its resident
  // logical count on every tick; replacing Schedule here discarded the
  // cold-reserved partition capacity and forced ReserveWorkspace to allocate
  // again on the warm path.
  workspace.schedule.packet_count = 0u;
  workspace.schedule.execution_width = 1u;
  workspace.schedule.intent = PartitionIntent::StaticWidth;
  workspace.schedule.placement = PlacementPolicy::Uniform;
  workspace.schedule.alignment_packets = 1u;
  workspace.schedule.packets_per_partition_max = 0u;
  workspace.schedule.worker_slot_count = 0u;
  workspace.schedule.fold_slot_count = 0u;
  workspace.schedule.useful_width = 0u;
  workspace.schedule.no_allocation = false;
  workspace.schedule.partitions.clear();
  ResetFoldSlots(workspace.fold_slots);
  ResetFoldGraph(workspace.fold_graph);
  workspace.program = KernelProgram{};
  workspace.packet_work_units.clear();
  workspace.ordered_packet_indices.clear();
  workspace.packet_partition_indices.clear();
  workspace.ordered_packet_scratch.clear();
  workspace.partition_loads.clear();
  workspace.partition_counts.clear();
  workspace.partition_offsets.clear();
  workspace.partition_write_offsets.clear();
  workspace.worker_stats_partitions_per_worker.clear();
  workspace.worker_stats_start_offset_ns.clear();
  workspace.worker_stats_elapsed_ns.clear();
  workspace.worker_stats_tail_wait_ns.clear();
  workspace.telemetry = Telemetry{};
  workspace.last_failure_reason = "not_run";
}

void ClearPacketWorkUnits(Workspace& workspace) {
  workspace.packet_work_units.clear();
}

bool ReservePacketWorkUnits(Workspace& workspace, const u32 packet_capacity) {
  try {
    workspace.packet_work_units.reserve(packet_capacity);
  } catch (...) {
    return false;
  }
  return true;
}

bool AppendPacketWorkUnit(Workspace& workspace, const u64 work_units) {
  try {
    workspace.packet_work_units.push_back(work_units);
  } catch (...) {
    return false;
  }
  return true;
}

std::span<const u64> ViewPacketWorkUnits(const Workspace& workspace) {
  return std::span<const u64>(workspace.packet_work_units.data(), workspace.packet_work_units.size());
}

std::span<const u32> ViewOrderedPacketIndices(const Workspace& workspace) {
  return std::span<const u32>(workspace.ordered_packet_indices.data(), workspace.ordered_packet_indices.size());
}

ScheduleView ViewSchedule(const Workspace& workspace) {
  ScheduleView view = ViewSchedule(workspace.schedule);
  if (workspace.ordered_packet_indices.size() == static_cast<std::size_t>(view.packet_count)) {
    view.ordered_packet_indices = workspace.ordered_packet_indices.data();
    view.ordered_packet_count = static_cast<u32>(workspace.ordered_packet_indices.size());
  }
  return view;
}

} // namespace rund::kernel
