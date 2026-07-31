#include "timing/local.hpp"

namespace rund::kernel::program_detail {
namespace {

u64 ElapsedNs(const TimePoint start) {
  const TimePoint end = Now();
  return static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

} // namespace

TimePoint Now() {
  return std::chrono::steady_clock::now();
}

void RecordProgramCompileCost(Workspace& workspace, const TimePoint start) {
  workspace.telemetry.program_compile_cost_ns = ElapsedNs(start);
  workspace.telemetry.program_compile_cost_measured = true;
}

void RecordCapacityCheckCost(Workspace& workspace, const TimePoint start) {
  workspace.telemetry.capacity_check_cost_ns = ElapsedNs(start);
  workspace.telemetry.capacity_check_cost_measured = true;
}

void RecordScheduleCompileCost(Workspace& workspace, const TimePoint start) {
  workspace.telemetry.schedule_compile_cost_ns = ElapsedNs(start);
  workspace.telemetry.schedule_compile_cost_measured = true;
}

void RecordFoldGraphCompileCost(Workspace& workspace, const TimePoint start) {
  workspace.telemetry.fold_graph_compile_cost_ns = ElapsedNs(start);
  workspace.telemetry.fold_graph_compile_cost_measured = true;
}

void RecordPlacementCost(Workspace& workspace, const TimePoint start) {
  workspace.telemetry.placement_cost_ns = ElapsedNs(start);
  workspace.telemetry.placement_cost_measured = true;
}

} // namespace rund::kernel::program_detail
