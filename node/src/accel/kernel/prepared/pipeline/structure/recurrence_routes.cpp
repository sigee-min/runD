#include "../../run.hpp"
#include "../demand.hpp"
#include "../reservation.hpp"
#include "../structure.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::add;
using ::rund::kernel::checked::mul;

namespace {

[[nodiscard]] PreparedKernelRecurrenceIdentity
recurrence_identity(const BackendRecurrence &recurrence) noexcept {
  PreparedKernelRecurrenceIdentity identity{
      .logical_step = recurrence.logical_step,
      .iteration = recurrence.iteration,
      .bound = recurrence.bound,
      .writes_each_iteration = recurrence.writes_each_iteration,
  };
  const BackendWindow *const window = recurrence.window;
  if (window == nullptr) {
    return identity;
  }
  identity.maximum = window->maximum;
  identity.tile = window->tile;
  identity.expected = window->expected;
  identity.outer_iteration = window->outer_iteration;
  identity.outer_bound = window->outer_bound;
  identity.inner_iteration = window->inner_iteration;
  identity.inner_bound = window->inner_bound;
  identity.route = window->route;
  identity.state = window->state;
  identity.phase = window->phase;
  identity.has_window = true;
  identity.has_terminal = window->has_terminal;
  return identity;
}

} // namespace

[[nodiscard]] bool route_recurrence_shape(
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const void *const route_owner, std::uint64_t &entry_count,
    std::uint64_t &occurrence_count, std::uint64_t &window_count,
    std::uint64_t &nested_group_count,
    std::uint64_t &map_recurrence_group_count,
    std::uint64_t &map_recurrence_history_group_count,
    std::uint64_t &recurrence_hi, std::uint64_t &recurrence_lo) noexcept {
  if (route_owner == nullptr || runs.size() != recurrences.size()) {
    return false;
  }
  SeedPreparedKernelRecurrenceFingerprint(recurrence_hi, recurrence_lo);
  bool top_level_recurrence =
      recurrences.size() > 1u &&
      recurrences.size() <= std::numeric_limits<std::uint32_t>::max();
  const std::uint32_t top_logical =
      top_level_recurrence ? recurrences.front().logical_step : 0u;
  const bool top_history =
      top_level_recurrence && recurrences.front().writes_each_iteration;
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const BackendRecurrence &marker = recurrences[index];
    top_level_recurrence =
        top_level_recurrence && marker.logical_step == top_logical &&
        marker.iteration == index && marker.bound == recurrences.size() &&
        marker.window == nullptr && marker.writes_each_iteration == top_history;
  }
  if (top_level_recurrence && runs.front() != nullptr &&
      runs.front()->owner.get() == route_owner &&
      (!accumulate(map_recurrence_group_count, 1u) ||
       (top_history && !accumulate(map_recurrence_history_group_count, 1u)))) {
    return false;
  }
  for (std::size_t index = 0u; index < recurrences.size();) {
    const BackendWindow *const window = recurrences[index].window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      const PreparedKernelRun *const item = runs[index];
      if (item == nullptr) {
        return false;
      }
      if (item->owner.get() == route_owner &&
          (!accumulate(entry_count, 1u) || !accumulate(occurrence_count, 1u) ||
           (window != nullptr && !accumulate(window_count, 1u)))) {
        return false;
      }
      if (item->owner.get() == route_owner &&
          !MixPreparedKernelRecurrenceFingerprint(
              recurrence_hi, recurrence_lo,
              recurrence_identity(recurrences[index]))) {
        return false;
      }
      ++index;
      continue;
    }
    NestedTemplateGeometry geometry{};
    if (!ProveNestedTemplateGeometry(recurrences, index, geometry) ||
        runs[index] == nullptr) {
      return false;
    }
    const NestedTemplateShape &shape = geometry.shape();
    const NestedTemplateRecurrenceIdentityBase identity_base{
        .logical_step = geometry.logical_step(),
        .maximum = geometry.window()->maximum,
        .tile = geometry.window()->tile,
        .expected = geometry.window()->expected,
        .state = geometry.window()->state,
        .has_terminal = geometry.window()->has_terminal,
    };
    for (std::size_t template_index = shape.first();
         template_index < shape.end(); ++template_index) {
      const PreparedKernelRun *const item = runs[template_index];
      NestedTemplateRouteProjection route{};
      PreparedKernelRecurrenceIdentity identity{};
      if (item == nullptr || !shape.project(template_index, route) ||
          !ProjectNestedRecurrenceIdentity(shape, template_index, identity_base,
                                           identity)) {
        return false;
      }
      if (item->owner.get() != route_owner) {
        continue;
      }
      if (!accumulate(entry_count, 1u) ||
          !accumulate(occurrence_count, route.occurrence_count) ||
          !accumulate(window_count, route.occurrence_count) ||
          !MixPreparedKernelRecurrenceFingerprint(recurrence_hi, recurrence_lo,
                                                  identity)) {
        return false;
      }
    }
    if (runs[index]->owner.get() == route_owner &&
        !accumulate(nested_group_count, 1u)) {
      return false;
    }
    if (shape.action_group_candidate() &&
        runs[shape.action_first()]->owner.get() == route_owner &&
        !accumulate(map_recurrence_group_count, 1u)) {
      return false;
    }
    index = shape.end();
  }
  // A compact terminal bank can be semantically authored yet have zero
  // physical occurrences for a one-iteration bound (for example fold bank
  // two). Its route/template identity still participates in the frozen plan.
  return entry_count != 0u;
}

// Recurrence route state belongs to one prepared stream, while a terminal or
// history template is shared by every structurally equal route in both native
// generation streams. Freeze that complete equivalence-class demand before a
// backend planner can allocate a descriptor arena or another template-private
// owner. The existing route-demand planner remains the single authority for
// duplicate owners and symmetric/transitive template equality.
[[nodiscard]] bool plan_runtime_recurrence_routes(
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const std::uint32_t route_copies,
    const std::span<RuntimeRecurrenceRoutePlan> plans) noexcept {
  if (runs.empty() || runs.size() != recurrences.size() ||
      plans.size() != runs.size() ||
      (route_copies != 1u && route_copies != 2u)) {
    return false;
  }

  std::array<BackendTemplateRouteDemand, PreparedPipelineStepCapacity>
      demands{};
  std::uint64_t unique_route_count = 0u;
  std::uint64_t template_count = 0u;
  if (!plan_backend_template_route_demands(
          runs, route_copies,
          std::span<BackendTemplateRouteDemand>{demands.data(), runs.size()},
          unique_route_count, template_count) ||
      unique_route_count == 0u || template_count == 0u) {
    return false;
  }

  for (RuntimeRecurrenceRoutePlan &plan : plans) {
    plan = {};
  }
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    if (item == nullptr || item->owner == nullptr) {
      return false;
    }
    bool duplicate = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      if (previous != nullptr && previous->owner.get() == item->owner.get()) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }
    RuntimeRecurrenceRoutePlan &plan = plans[index];
    plan.first_owner = true;
    if (!route_recurrence_shape(runs, recurrences, item->owner.get(),
                                plan.entry_count, plan.occurrence_count,
                                plan.window_count, plan.nested_group_count,
                                plan.group_count, plan.history_group_count,
                                plan.fingerprint_hi, plan.fingerprint_lo) ||
        plan.history_group_count > plan.group_count) {
      return false;
    }
  }

  for (std::size_t index = 0u; index < runs.size(); ++index) {
    RuntimeRecurrenceRoutePlan &plan = plans[index];
    if (!plan.first_owner) {
      continue;
    }
    const auto *const state =
        static_cast<const prepared::RunState *>(runs[index]->owner.get());
    if (state == nullptr) {
      return false;
    }
    const MapRecurrencePreparationPlan route_plan =
        PlanMapRecurrencePreparation(state->bound.run, plan.group_count,
                                     plan.history_group_count);
    if (!route_plan.ok) {
      return false;
    }
    for (std::size_t candidate = 0u; candidate < runs.size(); ++candidate) {
      const RuntimeRecurrenceRoutePlan &other = plans[candidate];
      if (!other.first_owner) {
        continue;
      }
      const auto *const other_state =
          static_cast<const prepared::RunState *>(runs[candidate]->owner.get());
      if (other_state == nullptr ||
          other_state->bound.run.ops != state->bound.run.ops) {
        return false;
      }
      const MapRecurrencePreparationPlan candidate_plan =
          PlanMapRecurrencePreparation(other_state->bound.run,
                                       other.group_count,
                                       other.history_group_count);
      if (!candidate_plan.ok) {
        return false;
      }
      if (SameMapRecurrenceTemplate(route_plan, candidate_plan, false) &&
          !accumulate(plan.template_capacities.terminal,
                      candidate_plan.terminal_group_count())) {
        return false;
      }
      if (SameMapRecurrenceTemplate(route_plan, candidate_plan, true) &&
          !accumulate(plan.template_capacities.history,
                      candidate_plan.history_group_count)) {
        return false;
      }
    }
    if (!mul(plan.template_capacities.terminal, route_copies,
             plan.template_capacities.terminal) ||
        !mul(plan.template_capacities.history, route_copies,
             plan.template_capacities.history)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
