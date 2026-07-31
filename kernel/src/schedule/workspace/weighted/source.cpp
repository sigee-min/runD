#include "local.hpp"
#include "../../work/units.hpp"

#include <functional>

namespace rund::kernel::workspace_detail {
namespace {

bool HasExactWorkUnits(const ScheduleCompileRequest& request) {
  return request.packet_work_units.size() == static_cast<std::size_t>(request.packet_count);
}

bool HasExactHints(const ScheduleCompileRequest& request) {
  return request.packet_hints.size() == static_cast<std::size_t>(request.packet_count);
}

bool Overlaps(const u64* const lhs_begin,
              const std::size_t lhs_size,
              const u64* const rhs_begin,
              const std::size_t rhs_size) {
  if (lhs_begin == nullptr || rhs_begin == nullptr || lhs_size == 0u || rhs_size == 0u) {
    return false;
  }
  const u64* const lhs_end = lhs_begin + lhs_size;
  const u64* const rhs_end = rhs_begin + rhs_size;
  const std::less<const u64*> less{};
  return less(lhs_begin, rhs_end) && less(rhs_begin, lhs_end);
}

bool OverlapsOwnedCapacity(const std::span<const u64> span, const std::vector<u64>& storage) {
  return Overlaps(span.data(), span.size(), storage.data(), storage.capacity());
}

} // namespace

WeightedWorkSourcePlan ClassifyWeightedWorkSource(const Workspace& workspace,
                                                  const ScheduleCompileRequest& request) {
  const std::span<const u64> work_units = request.packet_work_units;
  const bool exact_work_units = HasExactWorkUnits(request);
  const bool exact_hints = HasExactHints(request);

  if (request.packet_count == 0u) {
    return WeightedWorkSourcePlan{.kind = WeightedWorkSourceKind::Hints, .reason = "pass"};
  }
  if (!work_units.empty() && work_units.size() > static_cast<std::size_t>(request.packet_count)) {
    return WeightedWorkSourcePlan{.reason = "invalid_packet_work_units"};
  }

  if (!work_units.empty()) {
    const bool aliases_packet_work_units = OverlapsOwnedCapacity(work_units, workspace.packet_work_units);
    const bool aliases_partition_loads = OverlapsOwnedCapacity(work_units, workspace.partition_loads);
    const bool aliases_fold_slots = OverlapsOwnedCapacity(work_units, workspace.fold_slots.values);
    if (aliases_partition_loads || aliases_fold_slots) {
      return WeightedWorkSourcePlan{.reason = "invalid_packet_work_units"};
    }
    if (aliases_packet_work_units) {
      if (exact_work_units &&
          work_units.data() == workspace.packet_work_units.data() &&
          workspace.packet_work_units.size() == static_cast<std::size_t>(request.packet_count)) {
        return WeightedWorkSourcePlan{
            .kind = WeightedWorkSourceKind::WorkspacePacketWorkUnits,
            .reason = "pass",
        };
      }
      return WeightedWorkSourcePlan{.reason = "invalid_packet_work_units"};
    }
  }

  if (exact_work_units) {
    return WeightedWorkSourcePlan{.kind = WeightedWorkSourceKind::ExternalWorkUnits, .reason = "pass"};
  }
  if (exact_hints) {
    return WeightedWorkSourcePlan{.kind = WeightedWorkSourceKind::Hints, .reason = "pass"};
  }
  return WeightedWorkSourcePlan{.reason = "invalid_packet_work_units"};
}

bool MaterializeWeightedWorkUnits(Workspace& workspace,
                                  const ScheduleCompileRequest& request,
                                  const WeightedWorkSourcePlan plan) {
  if (workspace.packet_work_units.size() != static_cast<std::size_t>(request.packet_count)) {
    return false;
  }

  switch (plan.kind) {
    case WeightedWorkSourceKind::Hints:
      for (u32 packet = 0u; packet < request.packet_count; ++packet) {
        workspace.packet_work_units[packet] =
            request.packet_hints.empty()
                ? 1u
                : schedule_detail::NormalizeWorkUnits(request.packet_hints[packet].work_units);
      }
      return true;
    case WeightedWorkSourceKind::ExternalWorkUnits:
      for (u32 packet = 0u; packet < request.packet_count; ++packet) {
        workspace.packet_work_units[packet] =
            schedule_detail::NormalizeWorkUnits(request.packet_work_units[packet]);
      }
      return true;
    case WeightedWorkSourceKind::WorkspacePacketWorkUnits:
      for (u32 packet = 0u; packet < request.packet_count; ++packet) {
        workspace.packet_work_units[packet] =
            schedule_detail::NormalizeWorkUnits(workspace.packet_work_units[packet]);
      }
      return true;
    case WeightedWorkSourceKind::Invalid:
      return false;
  }
  return false;
}

std::span<const u64> ViewResolvedWeightedWorkUnits(const Workspace& workspace,
                                                   const u32 packet_count) {
  if (workspace.packet_work_units.size() != static_cast<std::size_t>(packet_count)) {
    return {};
  }
  return std::span<const u64>(workspace.packet_work_units.data(), workspace.packet_work_units.size());
}

} // namespace rund::kernel::workspace_detail
