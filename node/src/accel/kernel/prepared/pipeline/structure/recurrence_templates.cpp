#include "../reservation.hpp"
#include "../structure.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::add;
using ::rund::kernel::checked::mul;

[[nodiscard]] bool same_program_recurrence_authority(
    const PreparedKernelProgramRoute &left_route,
    const PreparedKernelProgramRoute &right_route) noexcept {
  const rund::AccelKernel *const left = left_route.kernel;
  const rund::AccelKernel *const right = right_route.kernel;
  return left != nullptr && right != nullptr && left->owner != nullptr &&
         left->owner == right->owner && left->kernel_id == right->kernel_id &&
         left->context_id == right->context_id &&
         left->graph_id_hi == right->graph_id_hi &&
         left->graph_id_lo == right->graph_id_lo &&
         left->node_count == right->node_count && left->api == right->api &&
         left->scalar == right->scalar && left->domain == right->domain;
}

[[nodiscard]] bool program_recurrence_template_capacities(
    const KernelExecution &execution,
    const std::span<const PreparedKernelProgramRoute> routes,
    const std::size_t route_index,
    const MapRecurrencePreparationPlan &route_plan,
    MapRecurrenceTemplateCapacities &capacities) noexcept {
  capacities = {};
  if (route_index >= routes.size() || !route_plan.ok) {
    return false;
  }
  const PreparedKernelProgramRoute &route = routes[route_index];
  for (const PreparedKernelProgramRoute &candidate : routes) {
    if (candidate.map_recurrence_history_group_count >
            candidate.map_recurrence_group_count ||
        (candidate.route_copies != 1u && candidate.route_copies != 2u)) {
      return false;
    }
    if (!same_program_recurrence_authority(route, candidate)) {
      continue;
    }
    const MapRecurrencePreparationPlan candidate_plan =
        PlanMapRecurrencePreparation(execution, candidate);
    if (!candidate_plan.ok) {
      return false;
    }
    std::uint64_t terminal = 0u;
    std::uint64_t history = 0u;
    if (SameMapRecurrenceTemplate(route_plan, candidate_plan, false) &&
        (!mul(candidate_plan.terminal_group_count(), candidate.route_copies,
              terminal) ||
         !accumulate(capacities.terminal, terminal))) {
      return false;
    }
    if (SameMapRecurrenceTemplate(route_plan, candidate_plan, true) &&
        (!mul(candidate_plan.history_group_count, candidate.route_copies,
              history) ||
         !accumulate(capacities.history, history))) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
