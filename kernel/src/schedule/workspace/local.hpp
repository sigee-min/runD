#pragma once

#include <kernel/schedule/workspace.hpp>
#include <math32/math32.hpp>

#include <chrono>
#include <cstddef>
#include <span>

namespace rund::kernel::workspace_detail {

struct WorkspaceRetentionPlan final {
  WorkspaceReservation reservation{};
  u64 bytes{};
  bool ok = false;
  const char *reason = "workspace_retention_not_planned";
};

WorkspaceReservation NormalizeReservation(WorkspaceReservation reservation);
[[nodiscard]] WorkspaceRetentionPlan
PlanWorkspaceRetention(WorkspaceReservation reservation) noexcept;
[[nodiscard]] u64 WorkspaceRetainedBytes(const Workspace &workspace) noexcept;
u32 MinimumCapacityMargin(const WorkspaceReservation &required,
                          const WorkspaceCapacity &available);
bool ReservationSatisfiedByCapacity(const WorkspaceCapacity &capacity,
                                    const WorkspaceReservation &reservation);

bool HasPacketHints(const ScheduleCompileRequest &request);
bool PacketHintsValid(const ScheduleCompileRequest &request);
bool HasWorkUnits(const ScheduleCompileRequest &request);
u64 PacketWorkUnitsAt(const ScheduleCompileRequest &request, u32 packet);

enum class WeightedWorkSourceKind {
  Invalid,
  Hints,
  ExternalWorkUnits,
  WorkspacePacketWorkUnits,
};

struct WeightedWorkSourcePlan {
  WeightedWorkSourceKind kind = WeightedWorkSourceKind::Invalid;
  const char *reason = "invalid_packet_work_units";
};

WeightedWorkSourcePlan
ClassifyWeightedWorkSource(const Workspace &workspace,
                           const ScheduleCompileRequest &request);
bool MaterializeWeightedWorkUnits(Workspace &workspace,
                                  const ScheduleCompileRequest &request,
                                  WeightedWorkSourcePlan plan);
std::span<const u64> ViewResolvedWeightedWorkUnits(const Workspace &workspace,
                                                   u32 packet_count);

void RecordCompileOutcome(Workspace &workspace,
                          const ScheduleCompileRequest &request,
                          const PartitionBuild &build,
                          std::span<const u64> resolved_work_units,
                          std::chrono::steady_clock::time_point start);
bool HasWeightedNoGrowthCapacity(const Workspace &workspace,
                                 const PartitionProjection &projection);
PartitionBuild
FailWeightedNoGrowthSchedule(Workspace &workspace,
                             const PartitionProjection &projection,
                             const char *reason);
PartitionBuild
CompileWeightedNoGrowthSchedule(Workspace &workspace,
                                const ScheduleCompileRequest &request,
                                const PartitionProjection &projection);

} // namespace rund::kernel::workspace_detail
