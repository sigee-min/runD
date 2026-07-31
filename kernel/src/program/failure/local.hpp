#pragma once

#include "../state.hpp"

#include <kernel/program/request.hpp>

namespace rund::kernel::program_detail {

void StoreInitialCapacityFailure(Workspace& workspace,
                                 const ScheduleCompileRequest& schedule_request,
                                 const KernelProgramCompileRequest& request,
                                 const KernelProgramCapacityProof& proof,
                                 const WorkerBackendCapabilities& capabilities,
                                 TimePoint program_start);

void StoreProgramFailure(Workspace& workspace,
                         const ScheduleCompileRequest& schedule_request,
                         const CurrentKernelProgramViews& current_views,
                         const WorkspaceReservation& required,
                         const KernelProgramCompileRequest& request,
                         const WorkerBackendCapabilities& capabilities,
                         const char* reason,
                         TimePoint program_start);

void StoreProgramSuccess(Workspace& workspace,
                         const KernelProgramCompileRequest& request,
                         const ScheduleCompileRequest& schedule_request,
                         const CurrentKernelProgramViews& current_views,
                         const KernelProgramCapacityProof& proof,
                         const KernelProgramPlacementMetadata& placement,
                         const WorkerBackendCapabilities& capabilities,
                         TimePoint program_start);

} // namespace rund::kernel::program_detail
