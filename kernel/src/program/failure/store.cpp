#include "local.hpp"
#include "../model/local.hpp"
#include "../strict/local.hpp"
#include "../timing/local.hpp"

#include <kernel/internal/program/contract.hpp>

namespace rund::kernel::program_detail {

void StoreProgramFailure(Workspace& workspace,
                         const ScheduleCompileRequest& schedule_request,
                         const CurrentKernelProgramViews& current_views,
                         const WorkspaceReservation& required,
                         const KernelProgramCompileRequest& request,
                         const WorkerBackendCapabilities& capabilities,
                         const char* const reason,
                         const TimePoint program_start) {
  const bool no_growth_required = internal::ProgramRequiresNoGrowth(request, schedule_request);
  const KernelProgramCapacityProof proof =
      BuildKernelProgramCapacityProof(workspace, required, no_growth_required, reason);
  AttachKernelProgramCapacityTelemetry(workspace.telemetry, proof);
  AttachKernelProgramPlacementTelemetry(workspace.telemetry, current_views.placement);
  AttachStrictFloatTelemetry(workspace.telemetry,
                             request.fold_operation,
                             request.strict_float_reduction,
                             capabilities);
  workspace.program = BuildProgramView(workspace,
                                       current_views,
                                       BuildDispatchContract(request, schedule_request, capabilities, proof),
                                       proof,
                                       current_views.placement,
                                       capabilities,
                                       request.physical_tile_policy,
                                       false,
                                       reason);
  RecordProgramCompileCost(workspace, program_start);
  workspace.program.telemetry_schema =
      BuildKernelProgramTelemetrySchema(workspace.telemetry,
                                        workspace.program.capacity,
                                        workspace.program.placement,
                                        workspace.program.schedule);
  workspace.last_failure_reason = reason;
}

} // namespace rund::kernel::program_detail
