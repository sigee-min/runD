#include "local.hpp"
#include "../model/local.hpp"
#include "../strict/local.hpp"
#include "../timing/local.hpp"

namespace rund::kernel::program_detail {

void StoreProgramSuccess(Workspace &workspace,
                         const KernelProgramCompileRequest &request,
                         const ScheduleCompileRequest &schedule_request,
                         const CurrentKernelProgramViews &current_views,
                         const KernelProgramCapacityProof &proof,
                         const KernelProgramPlacementMetadata &placement,
                         const WorkerBackendCapabilities &capabilities,
                         const TimePoint program_start) {
  workspace.program = BuildProgramView(
      workspace, current_views,
      BuildDispatchContract(request, schedule_request, capabilities, proof),
      proof, placement, capabilities, request.physical_tile_policy, true,
      "pass");
  AttachKernelProgramCapacityTelemetry(workspace.telemetry,
                                       workspace.program.capacity);
  AttachKernelProgramPlacementTelemetry(workspace.telemetry, placement);
  AttachStrictFloatTelemetry(workspace.telemetry, request.fold_operation,
                             request.strict_float_reduction, capabilities);
  RecordProgramCompileCost(workspace, program_start);
  workspace.program.telemetry_schema = BuildProgramTelemetrySchema(
      workspace.telemetry, workspace.program.capacity, placement,
      workspace.program.schedule);
  workspace.last_failure_reason = "pass";
}

} // namespace rund::kernel::program_detail
