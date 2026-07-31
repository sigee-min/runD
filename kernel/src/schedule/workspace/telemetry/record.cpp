#include "../local.hpp"

#include "../../../program/model/local.hpp"

#include <chrono>

namespace rund::kernel {

void RecordTelemetry(Workspace &workspace, const Result &result) {
  const auto telemetry_start = std::chrono::steady_clock::now();
  const u64 compile_cost_ns = workspace.telemetry.compile_cost_ns;
  const bool compile_cost_measured = workspace.telemetry.compile_cost_measured;
  const u64 schedule_compile_cost_ns =
      workspace.telemetry.schedule_compile_cost_ns;
  const bool schedule_compile_cost_measured =
      workspace.telemetry.schedule_compile_cost_measured;
  const u64 program_compile_cost_ns =
      workspace.telemetry.program_compile_cost_ns;
  const bool program_compile_cost_measured =
      workspace.telemetry.program_compile_cost_measured;
  const u64 capacity_check_cost_ns = workspace.telemetry.capacity_check_cost_ns;
  const bool capacity_check_cost_measured =
      workspace.telemetry.capacity_check_cost_measured;
  const u64 placement_cost_ns = workspace.telemetry.placement_cost_ns;
  const bool placement_cost_measured =
      workspace.telemetry.placement_cost_measured;
  const u64 fold_graph_compile_cost_ns =
      workspace.telemetry.fold_graph_compile_cost_ns;
  const bool fold_graph_compile_cost_measured =
      workspace.telemetry.fold_graph_compile_cost_measured;
  const bool strict_fp_software_reference =
      workspace.telemetry.strict_fp_software_reference;
  const bool strict_fp_backend_supported =
      workspace.telemetry.strict_fp_backend_supported;
  workspace.telemetry = result.telemetry;
  if (compile_cost_measured) {
    workspace.telemetry.compile_cost_ns = compile_cost_ns;
    workspace.telemetry.compile_cost_measured = true;
  }
  if (schedule_compile_cost_measured) {
    workspace.telemetry.schedule_compile_cost_ns = schedule_compile_cost_ns;
    workspace.telemetry.schedule_compile_cost_measured = true;
  }
  if (program_compile_cost_measured) {
    workspace.telemetry.program_compile_cost_ns = program_compile_cost_ns;
    workspace.telemetry.program_compile_cost_measured = true;
  }
  if (capacity_check_cost_measured) {
    workspace.telemetry.capacity_check_cost_ns = capacity_check_cost_ns;
    workspace.telemetry.capacity_check_cost_measured = true;
  }
  if (placement_cost_measured) {
    workspace.telemetry.placement_cost_ns = placement_cost_ns;
    workspace.telemetry.placement_cost_measured = true;
  }
  if (fold_graph_compile_cost_measured) {
    workspace.telemetry.fold_graph_compile_cost_ns = fold_graph_compile_cost_ns;
    workspace.telemetry.fold_graph_compile_cost_measured = true;
  }
  workspace.telemetry.strict_fp_software_reference =
      strict_fp_software_reference;
  workspace.telemetry.strict_fp_backend_supported = strict_fp_backend_supported;
  AttachKernelProgramCapacityTelemetry(workspace.telemetry,
                                       workspace.program.capacity);
  AttachKernelProgramPlacementTelemetry(workspace.telemetry,
                                        workspace.program.placement);
  const auto telemetry_end = std::chrono::steady_clock::now();
  workspace.telemetry.telemetry_update_cost_ns =
      static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           telemetry_end - telemetry_start)
                           .count());
  workspace.telemetry.telemetry_update_cost_measured = true;
  workspace.program.telemetry_schema =
      program_detail::BuildProgramTelemetrySchema(
          workspace.telemetry, workspace.program.capacity,
          workspace.program.placement, workspace.program.schedule);
  workspace.last_failure_reason = result.failure_reason;
}

} // namespace rund::kernel
