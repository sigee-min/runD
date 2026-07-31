#pragma once

#include "../state.hpp"

#include <kernel/program/build.hpp>

namespace rund::kernel::program_detail {

ScheduleCompileRequest
BuildProgramScheduleRequest(const KernelProgramCompileRequest &request,
                            const WorkerBackendCapabilities &capabilities);
WorkspaceReservation BuildProgramWorkspaceReservation(
    const KernelProgramCompileRequest &request,
    const ScheduleCompileRequest &schedule_request);
KernelProgramCapacityProof
CheckProgramCapacity(Workspace &workspace, const WorkspaceReservation &required,
                     bool require_no_allocation, const char *reason);
PartitionBuild
CompileProgramSchedule(Workspace &workspace,
                       const ScheduleCompileRequest &schedule_request);
FoldGraphBuild CompileProgramFoldGraph(
    Workspace &workspace, const PartitionBuild &schedule_build,
    const KernelProgramCompileRequest &request, AllocationPolicy allocation);
KernelProgramPlacementMetadata
BuildTimedProgramPlacement(Workspace &workspace,
                           const ScheduleCompileRequest &schedule_request,
                           const CurrentKernelProgramViews &current_views,
                           const WorkerBackendCapabilities &capabilities);
KernelProgramBuild
BuildInitialCapacityFailureResult(Workspace &workspace,
                                  const KernelProgramCapacityProof &proof);
KernelProgramBuild
BuildProgramFailureResult(Workspace &workspace, const char *reason,
                          const PartitionBuild &schedule_build,
                          const FoldGraphBuild &fold_build = FoldGraphBuild{});
KernelProgramBuild
BuildProgramSuccessResult(Workspace &workspace,
                          const PartitionBuild &schedule_build,
                          const FoldGraphBuild &fold_build);

} // namespace rund::kernel::program_detail
