#include "compile/local.hpp"
#include "dispatch/local.hpp"
#include "failure/local.hpp"
#include "timing/local.hpp"

#include <kernel/internal/program/contract.hpp>
#include <kernel/reduction/fold/graph/api.hpp>

#include <span>

namespace rund::kernel {
namespace {

program_detail::CurrentKernelProgramViews
CaptureCurrentScheduleViews(const Workspace &workspace,
                            const ScheduleCompileRequest &schedule_request) {
  program_detail::CurrentKernelProgramViews views{};
  views.schedule = ViewSchedule(workspace);
  views.ordered_packet_indices = views.schedule.ordered_packet_indices;
  views.ordered_packet_count = views.schedule.ordered_packet_count;
  if (schedule_request.placement == PlacementPolicy::WeightedStable) {
    const std::span<const u64> packet_work_units =
        ViewPacketWorkUnits(workspace);
    if (packet_work_units.size() ==
        static_cast<std::size_t>(schedule_request.packet_count)) {
      views.resolved_work_units = packet_work_units;
    }
  }
  views.placement = workspace.program.placement;
  return views;
}

program_detail::CurrentKernelProgramViews
WithCurrentFoldGraph(program_detail::CurrentKernelProgramViews views,
                     const Workspace &workspace) {
  views.fold_graph = ViewFoldGraph(workspace.fold_graph);
  return views;
}

program_detail::CurrentKernelProgramViews
WithCurrentPlacement(program_detail::CurrentKernelProgramViews views,
                     const KernelProgramPlacementMetadata &placement) {
  views.placement = placement;
  return views;
}

} // namespace

KernelProgramBuild
CompileKernelProgram(Workspace &workspace,
                     const KernelProgramCompileRequest &request) {
  ++workspace.program_generation;
  const program_detail::TimePoint program_start = program_detail::Now();
  const WorkerBackendCapabilities early_capabilities = InspectWorkerBackend(
      request.worker_backend, request.schedule.execution_width);

  const ScheduleCompileRequest schedule_request =
      program_detail::BuildProgramScheduleRequest(request, early_capabilities);
  const WorkspaceReservation required =
      program_detail::BuildProgramWorkspaceReservation(request,
                                                       schedule_request);
  const bool no_growth_required =
      internal::ProgramRequiresNoGrowth(request, schedule_request);
  const KernelProgramCapacityProof proof = program_detail::CheckProgramCapacity(
      workspace, required, no_growth_required,
      "program_workspace_capacity_exceeded");
  if (no_growth_required && !proof.satisfied) {
    program_detail::StoreInitialCapacityFailure(
        workspace, schedule_request, request, proof, early_capabilities,
        program_start);
    return program_detail::BuildInitialCapacityFailureResult(workspace, proof);
  }

  const PartitionBuild schedule_build =
      program_detail::CompileProgramSchedule(workspace, schedule_request);
  if (!schedule_build.ok) {
    const program_detail::CurrentKernelProgramViews empty_views{};
    program_detail::StoreProgramFailure(
        workspace, schedule_request, empty_views, required, request,
        early_capabilities, schedule_build.reason, program_start);
    return program_detail::BuildProgramFailureResult(
        workspace, schedule_build.reason, schedule_build);
  }
  const program_detail::CurrentKernelProgramViews schedule_views =
      CaptureCurrentScheduleViews(workspace, schedule_request);

  const auto fail_program =
      [&](const char *const reason,
          const WorkerBackendCapabilities &capabilities,
          const program_detail::CurrentKernelProgramViews &current_views) {
        program_detail::StoreProgramFailure(
            workspace, schedule_request, current_views, required, request,
            capabilities, reason, program_start);
      };

  const FoldGraphBuild fold_build = program_detail::CompileProgramFoldGraph(
      workspace, schedule_build, request, schedule_request.allocation);
  if (!fold_build.ok) {
    fail_program(fold_build.reason, early_capabilities, schedule_views);
    return program_detail::BuildProgramFailureResult(
        workspace, fold_build.reason, schedule_build, fold_build);
  }
  const program_detail::CurrentKernelProgramViews fold_views =
      WithCurrentFoldGraph(schedule_views, workspace);

  const WorkerBackendCapabilities capabilities = InspectWorkerBackend(
      request.worker_backend, schedule_request.execution_width);
  const char *const dispatch_failure =
      program_detail::ValidateDispatchRequirements(request, schedule_request,
                                                   workspace, capabilities);
  if (dispatch_failure != nullptr) {
    fail_program(dispatch_failure, capabilities, fold_views);
    return program_detail::BuildProgramFailureResult(
        workspace, dispatch_failure, schedule_build, fold_build);
  }

  const KernelProgramPlacementMetadata placement =
      program_detail::BuildTimedProgramPlacement(workspace, schedule_request,
                                                 fold_views, capabilities);
  const program_detail::CurrentKernelProgramViews success_views =
      WithCurrentPlacement(fold_views, placement);
  const KernelProgramCapacityProof final_proof =
      program_detail::CheckProgramCapacity(workspace, required,
                                           no_growth_required, "pass");
  program_detail::StoreProgramSuccess(workspace, request, schedule_request,
                                      success_views, final_proof, placement,
                                      capabilities, program_start);
  return program_detail::BuildProgramSuccessResult(workspace, schedule_build,
                                                   fold_build);
}

} // namespace rund::kernel
