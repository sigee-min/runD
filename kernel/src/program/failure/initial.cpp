#include "local.hpp"
#include "../model/local.hpp"
#include "../strict/local.hpp"
#include "../timing/local.hpp"

namespace rund::kernel::program_detail {

void StoreInitialCapacityFailure(Workspace& workspace,
                                 const ScheduleCompileRequest& schedule_request,
                                 const KernelProgramCompileRequest& request,
                                 const KernelProgramCapacityProof& proof,
                                 const WorkerBackendCapabilities& capabilities,
                                 const TimePoint program_start) {
  const u64 capacity_check_cost_ns = workspace.telemetry.capacity_check_cost_ns;
  const bool capacity_check_cost_measured = workspace.telemetry.capacity_check_cost_measured;
  workspace.telemetry = Telemetry{
      .packet_count = schedule_request.packet_count,
      .execution_width = schedule_request.execution_width,
      .placement = schedule_request.placement,
      .no_allocation = true,
  };
  workspace.telemetry.capacity_check_cost_ns = capacity_check_cost_ns;
  workspace.telemetry.capacity_check_cost_measured = capacity_check_cost_measured;
  AttachKernelProgramCapacityTelemetry(workspace.telemetry, proof);
  AttachStrictFloatTelemetry(workspace.telemetry,
                             request.fold_operation,
                             request.strict_float_reduction,
                             capabilities);
  RecordProgramCompileCost(workspace, program_start);
  const CurrentKernelProgramViews current_views{};
  workspace.program = BuildProgramView(workspace,
                                       current_views,
                                       KernelProgramDispatchContract{
                                           .require_no_allocation = request.require_no_allocation,
                                           .collect_worker_stats = request.collect_worker_stats,
                                           .require_dispatch_backend = request.require_dispatch_backend,
                                       },
                                       proof,
                                       KernelProgramPlacementMetadata{},
                                       capabilities,
                                       request.physical_tile_policy,
                                       false,
                                       proof.reason);
  workspace.last_failure_reason = proof.reason;
}

} // namespace rund::kernel::program_detail
