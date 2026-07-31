#include "../local.hpp"

#include <chrono>

namespace rund::kernel::workspace_detail {

void RecordCompileOutcome(Workspace& workspace,
                          const ScheduleCompileRequest& request,
                          const PartitionBuild& build,
                          const std::span<const u64> resolved_work_units,
                          const std::chrono::steady_clock::time_point start) {
  const auto end = std::chrono::steady_clock::now();
  const WorkspaceReservation required = ScheduleWorkspaceReservation(request);
  const bool checked = request.allocation == AllocationPolicy::NoGrowth;
  const auto capacity_start = std::chrono::steady_clock::now();
  const KernelProgramCapacityProof proof =
      BuildKernelProgramCapacityProof(workspace, required, checked, build.reason);
  const auto capacity_end = std::chrono::steady_clock::now();
  ScheduleView schedule_view = build.ok ? ViewSchedule(workspace.schedule) : ScheduleView{};
  if (build.ok &&
      workspace.ordered_packet_indices.size() == static_cast<std::size_t>(schedule_view.packet_count)) {
    schedule_view.ordered_packet_indices = workspace.ordered_packet_indices.data();
    schedule_view.ordered_packet_count = static_cast<u32>(workspace.ordered_packet_indices.size());
  }
  const auto placement_start = std::chrono::steady_clock::now();
  const KernelProgramPlacementMetadata placement =
      build.ok
          ? BuildKernelProgramPlacementMetadata(
                request,
                schedule_view,
                resolved_work_units,
                std::span<const u64>(workspace.partition_loads.data(), workspace.partition_loads.size()))
          : KernelProgramPlacementMetadata{};
  const auto placement_end = std::chrono::steady_clock::now();
  workspace.telemetry = Telemetry{
      .packet_count = request.packet_count,
      .execution_width = request.execution_width,
      .useful_width = build.useful_width,
      .partition_count = build.partition_count,
      .worker_slot_count = build.worker_slot_count,
      .fold_slot_count = build.fold_slot_count,
      .alignment_packets = build.alignment_packets,
      .packets_per_partition_max = build.packets_per_partition_max,
      .placement = build.placement,
      .no_allocation = build.no_allocation,
  };
  workspace.telemetry.compile_cost_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  workspace.telemetry.compile_cost_measured = true;
  workspace.telemetry.schedule_compile_cost_ns = workspace.telemetry.compile_cost_ns;
  workspace.telemetry.schedule_compile_cost_measured = true;
  workspace.telemetry.capacity_check_cost_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(capacity_end - capacity_start).count());
  workspace.telemetry.capacity_check_cost_measured = true;
  workspace.telemetry.placement_cost_ns = static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(placement_end - placement_start).count());
  workspace.telemetry.placement_cost_measured = true;
  AttachKernelProgramCapacityTelemetry(workspace.telemetry, proof);
  AttachKernelProgramPlacementTelemetry(workspace.telemetry, placement);
  workspace.program = KernelProgram{
      .schedule = schedule_view,
      .fold_graph = FoldGraphView{},
      .ordered_packet_indices = schedule_view.ordered_packet_indices,
      .ordered_packet_count = schedule_view.ordered_packet_count,
      .capacity = proof,
      .placement = placement,
      .telemetry_schema =
          BuildKernelProgramTelemetrySchema(workspace.telemetry, proof, placement, schedule_view),
      .ok = build.ok,
      .reason = build.reason,
  };
}

} // namespace rund::kernel::workspace_detail
