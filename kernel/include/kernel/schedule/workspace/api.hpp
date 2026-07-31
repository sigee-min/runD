#pragma once

#include <kernel/dispatch/kernel.hpp>
#include <kernel/schedule/workspace/capacity.hpp>
#include <kernel/schedule/workspace/state.hpp>

#include <span>

namespace rund::kernel {

struct KernelProgramCompileRequest;

void ResetWorkspace(Workspace& workspace);
bool ReserveWorkspace(Workspace& workspace, const WorkspaceReservation& reservation);
WorkspaceCapacity GetWorkspaceCapacity(const Workspace& workspace);
WorkspaceReservation ScheduleWorkspaceReservation(const ScheduleCompileRequest& request);
WorkspaceReservation KernelProgramWorkspaceReservation(const KernelProgramCompileRequest& request);
bool WorkspaceSatisfiesReservation(const Workspace& workspace, const WorkspaceReservation& reservation);
bool WorkspaceSatisfiesSchedule(const Workspace& workspace, const ScheduleCompileRequest& request);
KernelProgramCapacitySet ToKernelProgramCapacitySet(const WorkspaceReservation& reservation);
KernelProgramCapacitySet ToKernelProgramCapacitySet(const WorkspaceCapacity& capacity);
KernelProgramCapacityProof BuildKernelProgramCapacityProof(const Workspace& workspace,
                                                           const WorkspaceReservation& required,
                                                           bool checked,
                                                           const char* reason);
KernelProgramPlacementMetadata BuildKernelProgramPlacementMetadata(const ScheduleCompileRequest& request,
                                                                   ScheduleView schedule,
                                                                   std::span<const u64> resolved_work_units,
                                                                   std::span<const u64> partition_work_units,
                                                                   WorkerBackendCapabilities backend_capabilities = {});
KernelProgramPlacementMetadata BuildKernelProgramPlacementMetadata(const ScheduleCompileRequest& request,
                                                                   ScheduleView schedule,
                                                                   std::span<const u64> resolved_work_units,
                                                                   WorkerBackendCapabilities backend_capabilities = {});
KernelProgramPlacementMetadata BuildKernelProgramPlacementMetadata(const ScheduleCompileRequest& request,
                                                                   ScheduleView schedule,
                                                                   WorkerBackendCapabilities backend_capabilities = {});
void AttachKernelProgramCapacityTelemetry(Telemetry& telemetry,
                                          const KernelProgramCapacityProof& proof);
void AttachKernelProgramPlacementTelemetry(Telemetry& telemetry,
                                           const KernelProgramPlacementMetadata& metadata);
KernelProgramTelemetrySchema BuildKernelProgramTelemetrySchema(
    const Telemetry& telemetry,
    const KernelProgramCapacityProof& capacity,
    const KernelProgramPlacementMetadata& placement,
    ScheduleView schedule = ScheduleView{});
void ClearPacketWorkUnits(Workspace& workspace);
bool ReservePacketWorkUnits(Workspace& workspace, u32 packet_capacity);
bool AppendPacketWorkUnit(Workspace& workspace, u64 work_units);
std::span<const u64> ViewPacketWorkUnits(const Workspace& workspace);
std::span<const u32> ViewOrderedPacketIndices(const Workspace& workspace);

ScheduleView ViewSchedule(const Workspace& workspace);
void RecordTelemetry(Workspace& workspace, const Result& result);

} // namespace rund::kernel
