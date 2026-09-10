#include "../interface/api.hpp"

#include "../../../context/internal/support.hpp"
#include "../../backend/pipeline/failure.hpp"
#include "../../backend/template/identity.hpp"
#include "../../recurrence.hpp"
#include "../../recurrence/plan.hpp"
#include "../evidence.hpp"
#include "../model.hpp"
#include "backend.hpp"
#include "demand.hpp"
#include "expand.hpp"
#include "recurrence.hpp"
#include "registry.hpp"
#include "reservation.hpp"
#include "structure.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace rund::node::accel::detail {

namespace {

using ::rund::kernel::checked::add;
using ::rund::kernel::checked::mul;

} // namespace

PreparedKernelPipelineReservation PlanPreparedKernelPipeline(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const PreparedKernelPipelineShape shape,
    PreparedKernelTemplateRegistry *const templates) noexcept {
  (void)templates;
  PreparedKernelPipelineReservation result{};
  if (runs.empty() || runs.size() != recurrences.size() ||
      runs.size() > PreparedPipelineStepCapacity ||
      shape.publication_count > 32u || shape.declared_step_count == 0u ||
      shape.declared_step_count > PreparedPipelineStepCapacity ||
      !valid_publication_shape(shape) ||
      (shape.route_copies != 1u && shape.route_copies != 2u)) {
    result.reason = "accel_kernel_run_invalid";
    return result;
  }

  std::uint64_t unique_route_count = 0u;
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    if (item == nullptr || item->owner == nullptr) {
      result.reason = "accel_kernel_run_invalid";
      return result;
    }
    bool duplicate = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      duplicate = duplicate || (runs[prior] != nullptr &&
                                runs[prior]->owner.get() == item->owner.get());
    }
    unique_route_count += duplicate ? 0u : 1u;
  }
  fingerprint_pipeline_header(result.fingerprint_hi, result.fingerprint_lo,
                              context, shape, unique_route_count);

  std::array<RuntimeRecurrenceRoutePlan, PreparedPipelineStepCapacity>
      recurrence_routes{};
  if (!plan_runtime_recurrence_routes(
          runs, recurrences, shape.route_copies,
          std::span<RuntimeRecurrenceRoutePlan>{recurrence_routes.data(),
                                                runs.size()})) {
    result.reason = "accel_kernel_run_invalid";
    return result;
  }

  const BackendOps *ops = nullptr;
  std::uint64_t common_host_bytes = sizeof(prepared::PipelineState);
  std::uint64_t state_pointer_bytes = 0u;
  if (!mul(runs.size(), sizeof(std::shared_ptr<prepared::RunState>),
           state_pointer_bytes) ||
      !accumulate(common_host_bytes, state_pointer_bytes)) {
    return result;
  }
  result.host_bytes = common_host_bytes;
  result.template_capacity = std::numeric_limits<std::uint64_t>::max();
  std::array<bool, PreparedPipelineStepCapacity> recurrence_terminal{};
  std::array<bool, PreparedPipelineStepCapacity> recurrence_history{};

  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    const BackendOps *const candidate =
        state == nullptr ? nullptr : state->bound.run.ops;
    if (item == nullptr || !item->ok || state == nullptr ||
        !IsPipelinePrivatePreparation(state->mode) || candidate == nullptr ||
        candidate->plan_pipeline_private == nullptr ||
        candidate->same_pipeline_template == nullptr ||
        !prepared::MatchesContext(context, *state) ||
        (ops != nullptr && ops != candidate)) {
      result.reason = "accel_kernel_run_invalid";
      return result;
    }
    ops = candidate;

    const RuntimeRecurrenceRoutePlan &recurrence_route_plan =
        recurrence_routes[index];
    if (!recurrence_route_plan.first_owner) {
      continue;
    }
    const KernelAdmission &admission = state->execution.admission;
    fingerprint_route(
        result.fingerprint_hi, result.fingerprint_lo, admission.kernel_id,
        admission.graph_id_hi, admission.graph_id_lo, admission.node_count,
        admission.api, state->tile_count, state->bound.run.views,
        state->bound.run.scratch, recurrence_route_plan.entry_count,
        recurrence_route_plan.occurrence_count,
        recurrence_route_plan.window_count,
        recurrence_route_plan.nested_group_count,
        recurrence_route_plan.group_count,
        recurrence_route_plan.history_group_count,
        recurrence_route_plan.fingerprint_hi,
        recurrence_route_plan.fingerprint_lo, shape.route_copies);
    const backend_template_plan::MapSpecializationFingerprint map_fingerprint =
        backend_template_plan::runtime_map_specialization_fingerprint(
            state->bound.run);
    if (!map_fingerprint.ok) {
      result.reason = "accel_kernel_run_invalid";
      return result;
    }
    fingerprint_mix(result.fingerprint_hi, map_fingerprint.hi);
    fingerprint_mix(result.fingerprint_lo, map_fingerprint.lo);
    if (!accumulate(result.route_count, 1u)) {
      return result;
    }

    PreparedKernelRouteReservation route{};
    const rund::AccelCheck planned =
        candidate->plan_pipeline_private(state->bound.run, route);
    if (!planned.ok || route.route_step_count == 0u ||
        route.template_step_count == 0u || route.template_capacity == 0u ||
        !accumulate(route.route_host_bytes, sizeof(prepared::RunState))) {
      result.reason = planned.reason == nullptr ? "compute_pipeline_capacity"
                                                : planned.reason;
      return result;
    }
    result.template_capacity =
        std::min(result.template_capacity, route.template_capacity);
    result.template_step_capacity =
        std::min(result.template_step_capacity, route.template_step_capacity);
    if (!accumulate(result.route_host_bytes, route.route_host_bytes) ||
        !accumulate(result.route_native_bytes, route.route_native_bytes) ||
        !accumulate(result.host_bytes, route.route_host_bytes) ||
        !accumulate(result.native_bytes, route.route_native_bytes) ||
        !accumulate(result.route_step_count, route.route_step_count) ||
        !accumulate(result.descriptor_set_count, route.descriptor_set_count) ||
        !accumulate(result.descriptor_count, route.descriptor_count) ||
        !accumulate(result.native_allocation_count,
                    route.route_native_allocation_count)) {
      return result;
    }

    MapRecurrencePreparationPlan recurrence_plan = PlanMapRecurrencePreparation(
        state->bound.run, recurrence_route_plan.group_count,
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
    const rund::AccelCheck recurrence_checked = plan_map_recurrence_reservation(
        *candidate, recurrence_plan, recurrence);
    PreparedMapRecurrenceReservation terminal_recurrence{};
    PreparedMapRecurrenceReservation history_recurrence{};
    const rund::AccelCheck recurrence_variants_checked =
        recurrence_checked.ok ? verify_map_recurrence_template_variants(
                                    *candidate, recurrence_plan, recurrence,
                                    terminal_recurrence, history_recurrence)
                              : recurrence_checked;
    const PreparedMapRecurrenceReservation recurrence_route =
        map_recurrence_route_reservation(recurrence);
    PreparedKernelPipelineReservation recurrence_route_projection{};
    if (!recurrence_checked.ok || !recurrence_variants_checked.ok ||
        !project_map_recurrence_reservation(recurrence_route,
                                            recurrence_route_projection) ||
        !accumulate_reservation(result, recurrence_route_projection)) {
      result.reason = !recurrence_checked.ok
                          ? (recurrence_checked.reason == nullptr
                                 ? "compute_pipeline_capacity"
                                 : recurrence_checked.reason)
                          : (recurrence_variants_checked.reason == nullptr
                                 ? "compute_pipeline_capacity"
                                 : recurrence_variants_checked.reason);
      return result;
    }
    const auto recurrence_variant_seen = [&](const bool history) noexcept {
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
                previous_state->bound.run, previous_route_plan.group_count,
                previous_route_plan.history_group_count);
        if (SameMapRecurrenceTemplate(recurrence_plan, previous_plan,
                                      history)) {
          return true;
        }
      }
      return false;
    };
    const auto accumulate_recurrence_template =
        [&](const bool present, const bool history,
            const PreparedMapRecurrenceReservation &reservation) {
          if (!present || recurrence_variant_seen(history)) {
            return true;
          }
          PreparedKernelPipelineReservation recurrence_template_projection{};
          return project_map_recurrence_reservation(
                     reservation, recurrence_template_projection) &&
                 accumulate_reservation(result, recurrence_template_projection);
        };
    if (!accumulate_recurrence_template(recurrence_terminal[index], false,
                                        terminal_recurrence) ||
        !accumulate_recurrence_template(recurrence_history[index], true,
                                        history_recurrence)) {
      result.reason = "compute_pipeline_capacity";
      return result;
    }

    bool template_seen = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      const auto *const previous_state =
          previous == nullptr
              ? nullptr
              : static_cast<const prepared::RunState *>(previous->owner.get());
      if (previous_state != nullptr &&
          candidate->same_pipeline_template(state->bound.run,
                                            previous_state->bound.run)) {
        template_seen = true;
        break;
      }
    }
    if (template_seen) {
      continue;
    }
    if (!accumulate(result.template_count, 1u) ||
        !accumulate(result.template_host_bytes, route.template_host_bytes) ||
        !accumulate(result.template_native_bytes,
                    route.template_native_bytes) ||
        !accumulate(result.host_bytes, route.template_host_bytes) ||
        !accumulate(result.native_bytes, route.template_native_bytes) ||
        !accumulate(result.template_source_bytes,
                    route.template_source_bytes) ||
        !accumulate(result.template_step_count, route.template_step_count) ||
        !accumulate(result.native_allocation_count,
                    route.template_native_allocation_count)) {
      return result;
    }
    result.source_transient_bytes =
        std::max(result.source_transient_bytes, route.source_transient_bytes);
  }

  result.service_free_direct_host_bytes =
      result.map_recurrence.group_count == 0u ? 0u
                                              : sizeof(ServiceFreeDirectProof);
  if (!accumulate(result.host_bytes, result.service_free_direct_host_bytes) ||
      !accumulate(result.host_bytes, result.source_transient_bytes) ||
      ops == nullptr || result.template_count == 0u ||
      result.template_count > result.template_capacity ||
      result.descriptor_set_count > result.descriptor_set_capacity ||
      result.descriptor_count > result.descriptor_capacity ||
      result.host_bytes > std::numeric_limits<std::size_t>::max()) {
    return result;
  }
  result.ok = true;
  result.reason = "ok";
  return result;
}

} // namespace rund::node::accel::detail
