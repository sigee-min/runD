#pragma once

#include "../state.hpp"

#include <kernel/program/request.hpp>

namespace rund::kernel::program_detail {

KernelProgram
BuildProgramView(const Workspace &workspace,
                 const CurrentKernelProgramViews &current_views,
                 KernelProgramDispatchContract dispatch,
                 const KernelProgramCapacityProof &capacity,
                 const KernelProgramPlacementMetadata &placement,
                 const WorkerBackendCapabilities &backend_capabilities,
                 const KernelProgramPhysicalTilePolicy &physical_tile_policy,
                 bool ok, const char *reason);

KernelProgramDispatchContract
BuildDispatchContract(const KernelProgramCompileRequest &request,
                      const ScheduleCompileRequest &schedule_request,
                      const WorkerBackendCapabilities &capabilities,
                      const KernelProgramCapacityProof &proof);

KernelProgramTelemetrySchema BuildProgramTelemetrySchema(
    const Telemetry &telemetry, const KernelProgramCapacityProof &capacity,
    const KernelProgramPlacementMetadata &placement, ScheduleView schedule);

} // namespace rund::kernel::program_detail
