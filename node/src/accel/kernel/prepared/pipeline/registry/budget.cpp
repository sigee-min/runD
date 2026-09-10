#include "../../interface/api.hpp"
#include "../../run.hpp"
#include "../registry.hpp"

#include "../recurrence.hpp"
#include "../reservation.hpp"
#include "../structure.hpp"
#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <new>
#include <span>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::mul;

namespace {

[[nodiscard]] bool
same_charged_template(const PreparedKernelTemplateCharge &charged,
                      const BackendOps &ops, const BackendRun &probe,
                      const PreparedKernelTemplateChargeKind kind) noexcept {
  return charged.ops == &ops && charged.kind == kind &&
         charged.probe != nullptr && ops.same_pipeline_template != nullptr &&
         ops.same_pipeline_template(probe, charged.probe->bound.run) &&
         ops.same_pipeline_template(charged.probe->bound.run, probe);
}

[[nodiscard]] bool same_charged_recurrence_template(
    const PreparedKernelTemplateCharge &charged, const BackendOps &ops,
    const MapRecurrencePreparationPlan &probe, const bool history) noexcept {
  const PreparedKernelTemplateChargeKind kind =
      history ? PreparedKernelTemplateChargeKind::RecurrenceHistory
              : PreparedKernelTemplateChargeKind::RecurrenceTerminal;
  if (charged.ops != &ops || charged.kind != kind || charged.probe == nullptr) {
    return false;
  }
  const MapRecurrencePreparationPlan cached = PlanMapRecurrencePreparation(
      charged.probe->bound.run, 1u, history ? 1u : 0u);
  return SameMapRecurrenceTemplate(probe, cached, history);
}

} // namespace

// Reserves the combined public-plan budget before backend/native
// materialization. Route state is charged for every stream. An immutable
// Program template is charged only by the first semantically equal stream in
// this registry. The registry mutex makes concurrent primary/alternate cold
// preparation one transaction with respect to the frozen limit.
[[nodiscard]] rund::AccelCheck
reserve_pipeline_budget(PreparedKernelTemplateRegistry &registry,
                        const std::span<const PreparedKernelRun *const> runs,
                        const std::span<const BackendRecurrence> recurrences,
                        const PreparedKernelPipelineReservation &structure,
                        const PreparedMapRecurrenceReservation &map_recurrence,
                        const std::uint32_t route_copies,
                        PipelineBudgetTransaction &transaction) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || runs.empty() || runs.size() != recurrences.size() ||
      !registry.limit.ok) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }

  std::array<RuntimeRecurrenceRoutePlan, PreparedPipelineStepCapacity>
      recurrence_routes{};
  if (!plan_runtime_recurrence_routes(
          runs, recurrences, route_copies,
          std::span<RuntimeRecurrenceRoutePlan>{recurrence_routes.data(),
                                                runs.size()})) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }

  try {
    if (!transaction.begin(*state, registry)) {
      return rund::AccelCheck{false, "accel_kernel_template_invalid"};
    }
    PreparedKernelPipelineReservation charge{};
    charge.ok = true;
    charge.reason = "ok";
    charge.template_capacity = registry.limit.template_capacity;
    charge.descriptor_set_capacity = registry.limit.descriptor_set_capacity;
    charge.descriptor_capacity = registry.limit.descriptor_capacity;
    charge.host_bytes = sizeof(prepared::PipelineState);
    std::uint64_t state_pointer_bytes = 0u;
    if (!mul(runs.size(), sizeof(std::shared_ptr<prepared::RunState>),
             state_pointer_bytes) ||
        !accumulate(charge.host_bytes, state_pointer_bytes) ||
        !accumulate_reservation(charge, structure)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const PreparedMapRecurrenceReservation recurrence_route =
        map_recurrence_route_reservation(map_recurrence);
    PreparedKernelPipelineReservation recurrence_route_charge{};
    if (!project_map_recurrence_reservation(recurrence_route,
                                            recurrence_route_charge) ||
        !accumulate_reservation(charge, recurrence_route_charge)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    PreparedMapRecurrenceReservation observed_recurrence_route{};
    PreparedMapRecurrenceReservation observed_recurrence_templates{};
    std::array<bool, PreparedPipelineStepCapacity> recurrence_terminal{};
    std::array<bool, PreparedPipelineStepCapacity> recurrence_history{};
    for (std::size_t index = 0u; index < runs.size(); ++index) {
      const PreparedKernelRun *const item = runs[index];
      auto route =
          item == nullptr
              ? std::shared_ptr<prepared::RunState>{}
              : std::static_pointer_cast<prepared::RunState>(item->owner);
      const BackendOps *const ops =
          route == nullptr ? nullptr : route->bound.run.ops;
      if (item == nullptr || !item->ok || route == nullptr || ops == nullptr ||
          ops->plan_pipeline_private == nullptr ||
          ops->same_pipeline_template == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }

      bool duplicate_route = false;
      for (std::size_t prior = 0u; prior < index; ++prior) {
        const PreparedKernelRun *const previous = runs[prior];
        if (previous != nullptr && previous->owner.get() == item->owner.get()) {
          duplicate_route = true;
          break;
        }
      }
      if (duplicate_route) {
        continue;
      }

      PreparedKernelRouteReservation planned{};
      const rund::AccelCheck checked =
          ops->plan_pipeline_private(route->bound.run, planned);
      if (!checked.ok || planned.route_step_count == 0u ||
          planned.template_step_count == 0u ||
          planned.template_capacity == 0u ||
          !accumulate(planned.route_host_bytes, sizeof(prepared::RunState))) {
        return rund::AccelCheck{false, checked.reason == nullptr
                                           ? "compute_pipeline_capacity"
                                           : checked.reason};
      }
      PreparedKernelPipelineReservation route_charge{};
      route_charge.ok = true;
      route_charge.reason = "ok";
      route_charge.host_bytes = planned.route_host_bytes;
      route_charge.native_bytes = planned.route_native_bytes;
      route_charge.route_host_bytes = planned.route_host_bytes;
      route_charge.route_native_bytes = planned.route_native_bytes;
      route_charge.route_count = 1u;
      route_charge.route_step_count = planned.route_step_count;
      route_charge.descriptor_set_count = planned.descriptor_set_count;
      route_charge.descriptor_count = planned.descriptor_count;
      route_charge.native_allocation_count =
          planned.route_native_allocation_count;
      if (!accumulate_reservation(charge, route_charge)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }

      const RuntimeRecurrenceRoutePlan &recurrence_route_plan =
          recurrence_routes[index];
      if (!recurrence_route_plan.first_owner) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      MapRecurrencePreparationPlan recurrence_plan =
          PlanMapRecurrencePreparation(
              route->bound.run, recurrence_route_plan.group_count,
              recurrence_route_plan.history_group_count);
      recurrence_plan.terminal_template_group_capacity =
          recurrence_plan.terminal_group_count() == 0u
              ? 0u
              : recurrence_route_plan.template_capacities.terminal;
      recurrence_plan.history_template_group_capacity =
          recurrence_plan.history_group_count == 0u
              ? 0u
              : recurrence_route_plan.template_capacities.history;
      recurrence_terminal[index] = recurrence_plan.terminal_group_count() != 0u;
      recurrence_history[index] = recurrence_plan.history_group_count != 0u;
      PreparedMapRecurrenceReservation recurrence{};
      const rund::AccelCheck recurrence_checked =
          plan_map_recurrence_reservation(*ops, recurrence_plan, recurrence);
      PreparedMapRecurrenceReservation terminal_reservation{};
      PreparedMapRecurrenceReservation history_reservation{};
      const rund::AccelCheck recurrence_variants_checked =
          recurrence_checked.ok ? verify_map_recurrence_template_variants(
                                      *ops, recurrence_plan, recurrence,
                                      terminal_reservation, history_reservation)
                                : recurrence_checked;
      if (!recurrence_checked.ok || !recurrence_variants_checked.ok ||
          !accumulate_map_recurrence_route(
              observed_recurrence_route,
              map_recurrence_route_reservation(recurrence))) {
        return rund::AccelCheck{
            false, !recurrence_checked.ok
                       ? (recurrence_checked.reason == nullptr
                              ? "compute_pipeline_capacity"
                              : recurrence_checked.reason)
                       : (recurrence_variants_checked.reason == nullptr
                              ? "compute_pipeline_capacity"
                              : recurrence_variants_checked.reason)};
      }

      const auto current_recurrence_template_seen =
          [&](const bool history) noexcept {
            for (std::size_t prior = 0u; prior < index; ++prior) {
              const PreparedKernelRun *const previous = runs[prior];
              const auto *const previous_state =
                  previous == nullptr ? nullptr
                                      : static_cast<const prepared::RunState *>(
                                            previous->owner.get());
              if (!(history ? recurrence_history[prior]
                            : recurrence_terminal[prior]) ||
                  previous_state == nullptr) {
                continue;
              }
              const RuntimeRecurrenceRoutePlan &previous_route_plan =
                  recurrence_routes[prior];
              const MapRecurrencePreparationPlan previous_plan =
                  PlanMapRecurrencePreparation(
                      previous_state->bound.run,
                      previous_route_plan.group_count,
                      previous_route_plan.history_group_count);
              if (SameMapRecurrenceTemplate(recurrence_plan, previous_plan,
                                            history)) {
                return true;
              }
            }
            return false;
          };
      const auto charge_recurrence_template =
          [&](const bool present, const bool history,
              const PreparedMapRecurrenceReservation &reservation) {
            if (!present || current_recurrence_template_seen(history)) {
              return true;
            }
            if (!accumulate_map_recurrence_template(
                    observed_recurrence_templates, reservation)) {
              return false;
            }
            const PreparedKernelTemplateChargeKind kind =
                history ? PreparedKernelTemplateChargeKind::RecurrenceHistory
                        : PreparedKernelTemplateChargeKind::RecurrenceTerminal;
            bool recurrence_template_seen = false;
            for (const PreparedKernelTemplateCharge &charged :
                 state->template_charges) {
              if (same_charged_recurrence_template(charged, *ops,
                                                   recurrence_plan, history)) {
                recurrence_template_seen = true;
                break;
              }
            }
            if (recurrence_template_seen) {
              return true;
            }
            PreparedKernelPipelineReservation recurrence_template_charge{};
            if (!project_map_recurrence_reservation(
                    reservation, recurrence_template_charge) ||
                !accumulate_reservation(charge, recurrence_template_charge) ||
                state->template_charges.size() ==
                    state->template_charges.capacity()) {
              return false;
            }
            state->template_charges.push_back(
                PreparedKernelTemplateCharge{route, ops, kind});
            return true;
          };
      if (!charge_recurrence_template(recurrence_terminal[index], false,
                                      terminal_reservation) ||
          !charge_recurrence_template(recurrence_history[index], true,
                                      history_reservation)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }

      bool template_seen = false;
      for (const PreparedKernelTemplateCharge &charged :
           state->template_charges) {
        if (same_charged_template(charged, *ops, route->bound.run,
                                  PreparedKernelTemplateChargeKind::Program)) {
          template_seen = true;
          break;
        }
      }
      if (template_seen) {
        continue;
      }

      PreparedKernelPipelineReservation template_charge{};
      template_charge.ok = true;
      template_charge.reason = "ok";
      template_charge.host_bytes = planned.template_host_bytes;
      template_charge.native_bytes = planned.template_native_bytes;
      template_charge.template_host_bytes = planned.template_host_bytes;
      template_charge.template_native_bytes = planned.template_native_bytes;
      template_charge.template_source_bytes = planned.template_source_bytes;
      template_charge.source_transient_bytes = planned.source_transient_bytes;
      template_charge.template_count = 1u;
      template_charge.template_step_count = planned.template_step_count;
      template_charge.native_allocation_count =
          planned.template_native_allocation_count;
      if (!accumulate_reservation(charge, template_charge)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      if (state->template_charges.size() ==
          state->template_charges.capacity()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      state->template_charges.push_back(PreparedKernelTemplateCharge{
          route, ops, PreparedKernelTemplateChargeKind::Program});
    }

    if (!same_map_recurrence_reservation(
            observed_recurrence_route,
            map_recurrence_route_reservation(map_recurrence)) ||
        !same_map_recurrence_reservation(
            observed_recurrence_templates,
            map_recurrence_template_reservation(map_recurrence))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    // Source specialization is serialized by the registry transaction. Only
    // the increase over the already consumed high-water can coexist with the
    // retained owners; primary/alternate streams must not add the same
    // transient a second time.
    if (charge.source_transient_bytes >
            state->consumed.source_transient_bytes &&
        !accumulate(charge.host_bytes,
                    charge.source_transient_bytes -
                        state->consumed.source_transient_bytes)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    // Backend cold-finalizer workspace is independent of source emission and
    // is likewise serialized per prepared stream. Charge only a new
    // high-water; no transient allocation becomes a retained registry owner.
    if (charge.host_transient_bytes > state->consumed.host_transient_bytes &&
        !accumulate(charge.host_bytes,
                    charge.host_transient_bytes -
                        state->consumed.host_transient_bytes)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    PreparedKernelPipelineReservation prospective = state->consumed;
    prospective.ok = true;
    prospective.reason = "ok";
    prospective.template_capacity = registry.limit.template_capacity;
    prospective.template_step_capacity = registry.limit.template_step_capacity;
    prospective.descriptor_set_capacity =
        registry.limit.descriptor_set_capacity;
    prospective.descriptor_capacity = registry.limit.descriptor_capacity;
    if (!accumulate_reservation(prospective, charge) ||
        !PreparedKernelPipelineReservationWithin(prospective, registry.limit)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    state->consumed = prospective;
    registry.reservation = prospective;
    return rund::AccelCheck{true, "ok"};
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
}

} // namespace rund::node::accel::detail
