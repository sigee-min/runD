#include "recurrence.hpp"
#include "reservation.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>

namespace rund::node::accel::detail {
namespace {

using ::rund::kernel::checked::add;
using ::rund::kernel::checked::mul;

[[nodiscard]] bool valid_map_recurrence_reservation(
    const MapRecurrencePreparationPlan &plan,
    const PreparedMapRecurrenceReservation &reservation) noexcept {
  const std::uint64_t expected_templates =
      static_cast<std::uint64_t>(plan.terminal_group_count() != 0u) +
      static_cast<std::uint64_t>(plan.history_group_count != 0u);
  return plan.eligible() && reservation.group_count == plan.group_count &&
         reservation.history_group_count == plan.history_group_count &&
         reservation.template_count == expected_templates &&
         reservation.terminal_template_group_capacity ==
             plan.terminal_template_group_capacity &&
         reservation.history_template_group_capacity ==
             plan.history_template_group_capacity &&
         reservation.route_step_count == plan.group_count &&
         reservation.template_step_count == expected_templates;
}

[[nodiscard]] rund::AccelCheck plan_map_recurrence_template_variant(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    const bool history,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
  const bool present = history ? plan.history_group_count != 0u
                               : plan.terminal_group_count() != 0u;
  if (!plan.ok || !present) {
    return rund::AccelCheck{false, plan.reason == nullptr
                                       ? "compute_pipeline_recurrence_invalid"
                                       : plan.reason};
  }
  MapRecurrencePreparationPlan variant = plan;
  variant.group_count = 1u;
  variant.history_group_count = history ? 1u : 0u;
  if (history) {
    variant.terminal_source = {};
    variant.terminal_template_group_capacity = 0u;
  } else {
    variant.history_source = {};
    variant.history_template_group_capacity = 0u;
  }
  PreparedMapRecurrenceReservation planned{};
  const rund::AccelCheck checked =
      plan_map_recurrence_reservation(ops, variant, planned);
  if (!checked.ok) {
    return checked;
  }
  reservation = map_recurrence_template_reservation(planned);
  if (reservation.template_count != 1u ||
      reservation.template_step_count != 1u) {
    reservation = {};
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace

[[nodiscard]] rund::AccelCheck plan_map_recurrence_reservation(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
  if (!plan.ok) {
    return rund::AccelCheck{false, plan.reason == nullptr
                                       ? "compute_pipeline_recurrence_invalid"
                                       : plan.reason};
  }
  if (!plan.eligible()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (ops.plan_pipeline_recurrence == nullptr) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const rund::AccelCheck checked =
      ops.plan_pipeline_recurrence(plan, reservation);
  if (!checked.ok || !valid_map_recurrence_reservation(plan, reservation)) {
    reservation = {};
    return rund::AccelCheck{false, checked.reason == nullptr
                                       ? "compute_pipeline_capacity"
                                       : checked.reason};
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] bool scale_map_recurrence_route_reservation(
    const PreparedMapRecurrenceReservation &value, const std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept {
  if (copies == 0u) {
    return false;
  }
  scaled = {};
  return mul(value.route_host_bytes, copies, scaled.route_host_bytes) &&
         mul(value.route_native_bytes, copies,
                  scaled.route_native_bytes) &&
         mul(value.group_count, copies, scaled.group_count) &&
         mul(value.history_group_count, copies,
                  scaled.history_group_count) &&
         mul(value.route_step_count, copies, scaled.route_step_count) &&
         mul(value.route_native_allocation_count, copies,
                  scaled.route_native_allocation_count);
}

[[nodiscard]] PreparedMapRecurrenceReservation
map_recurrence_route_reservation(
    const PreparedMapRecurrenceReservation &value) noexcept {
  return PreparedMapRecurrenceReservation{
      .route_host_bytes = value.route_host_bytes,
      .route_native_bytes = value.route_native_bytes,
      .group_count = value.group_count,
      .history_group_count = value.history_group_count,
      .route_step_count = value.route_step_count,
      .route_native_allocation_count = value.route_native_allocation_count,
  };
}

[[nodiscard]] PreparedMapRecurrenceReservation
map_recurrence_template_reservation(
    const PreparedMapRecurrenceReservation &value) noexcept {
  return PreparedMapRecurrenceReservation{
      .template_host_bytes = value.template_host_bytes,
      .template_native_bytes = value.template_native_bytes,
      .template_source_bytes = value.template_source_bytes,
      .source_transient_bytes = value.source_transient_bytes,
      .template_count = value.template_count,
      .terminal_template_group_capacity =
          value.terminal_template_group_capacity,
      .history_template_group_capacity = value.history_template_group_capacity,
      .template_step_count = value.template_step_count,
      .descriptor_set_count = value.descriptor_set_count,
      .descriptor_count = value.descriptor_count,
      .template_native_allocation_count =
          value.template_native_allocation_count,
  };
}

[[nodiscard]] bool same_map_recurrence_reservation(
    const PreparedMapRecurrenceReservation &left,
    const PreparedMapRecurrenceReservation &right) noexcept {
  return left.route_host_bytes == right.route_host_bytes &&
         left.route_native_bytes == right.route_native_bytes &&
         left.template_host_bytes == right.template_host_bytes &&
         left.template_native_bytes == right.template_native_bytes &&
         left.template_source_bytes == right.template_source_bytes &&
         left.source_transient_bytes == right.source_transient_bytes &&
         left.group_count == right.group_count &&
         left.history_group_count == right.history_group_count &&
         left.template_count == right.template_count &&
         left.terminal_template_group_capacity ==
             right.terminal_template_group_capacity &&
         left.history_template_group_capacity ==
             right.history_template_group_capacity &&
         left.route_step_count == right.route_step_count &&
         left.template_step_count == right.template_step_count &&
         left.descriptor_set_count == right.descriptor_set_count &&
         left.descriptor_count == right.descriptor_count &&
         left.route_native_allocation_count ==
             right.route_native_allocation_count &&
         left.template_native_allocation_count ==
             right.template_native_allocation_count;
}

[[nodiscard]] bool accumulate_map_recurrence_template(
    PreparedMapRecurrenceReservation &target,
    const PreparedMapRecurrenceReservation &value) noexcept {
  target.source_transient_bytes =
      std::max(target.source_transient_bytes, value.source_transient_bytes);
  return accumulate(target.template_host_bytes, value.template_host_bytes) &&
         accumulate(target.template_native_bytes,
                    value.template_native_bytes) &&
         accumulate(target.template_source_bytes,
                    value.template_source_bytes) &&
         accumulate(target.template_count, value.template_count) &&
         accumulate(target.terminal_template_group_capacity,
                    value.terminal_template_group_capacity) &&
         accumulate(target.history_template_group_capacity,
                    value.history_template_group_capacity) &&
         accumulate(target.template_step_count, value.template_step_count) &&
         accumulate(target.descriptor_set_count, value.descriptor_set_count) &&
         accumulate(target.descriptor_count, value.descriptor_count) &&
         accumulate(target.template_native_allocation_count,
                    value.template_native_allocation_count);
}

[[nodiscard]] bool accumulate_map_recurrence_route(
    PreparedMapRecurrenceReservation &target,
    const PreparedMapRecurrenceReservation &value) noexcept {
  return accumulate(target.route_host_bytes, value.route_host_bytes) &&
         accumulate(target.route_native_bytes, value.route_native_bytes) &&
         accumulate(target.group_count, value.group_count) &&
         accumulate(target.history_group_count, value.history_group_count) &&
         accumulate(target.route_step_count, value.route_step_count) &&
         accumulate(target.route_native_allocation_count,
                    value.route_native_allocation_count);
}

[[nodiscard]] rund::AccelCheck verify_map_recurrence_template_variants(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    const PreparedMapRecurrenceReservation &combined,
    PreparedMapRecurrenceReservation &terminal,
    PreparedMapRecurrenceReservation &history) noexcept {
  terminal = {};
  history = {};
  PreparedMapRecurrenceReservation observed{};
  if (plan.terminal_group_count() != 0u) {
    const rund::AccelCheck checked =
        plan_map_recurrence_template_variant(ops, plan, false, terminal);
    if (!checked.ok ||
        !accumulate_map_recurrence_template(observed, terminal)) {
      return checked.ok ? rund::AccelCheck{false, "compute_pipeline_capacity"}
                        : checked;
    }
  }
  if (plan.history_group_count != 0u) {
    const rund::AccelCheck checked =
        plan_map_recurrence_template_variant(ops, plan, true, history);
    if (!checked.ok || !accumulate_map_recurrence_template(observed, history)) {
      return checked.ok ? rund::AccelCheck{false, "compute_pipeline_capacity"}
                        : checked;
    }
  }
  return same_map_recurrence_reservation(
             observed, map_recurrence_template_reservation(combined))
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_pipeline_capacity"};
}

// Projects the auditable recurrence subset into the generic preparation
// totals exactly once. Callers then use the ordinary reservation accumulator;
// no backend may maintain a second hidden recurrence budget.
[[nodiscard]] bool project_map_recurrence_reservation(
    const PreparedMapRecurrenceReservation &recurrence,
    PreparedKernelPipelineReservation &projection) noexcept {
  projection = {};
  projection.ok = true;
  projection.reason = "ok";
  projection.map_recurrence = recurrence;
  projection.source_transient_bytes = recurrence.source_transient_bytes;
  return add(recurrence.route_host_bytes, recurrence.template_host_bytes,
             projection.host_bytes) &&
         add(recurrence.route_native_bytes, recurrence.template_native_bytes,
             projection.native_bytes) &&
         accumulate(projection.route_host_bytes, recurrence.route_host_bytes) &&
         accumulate(projection.route_native_bytes,
                    recurrence.route_native_bytes) &&
         accumulate(projection.template_host_bytes,
                    recurrence.template_host_bytes) &&
         accumulate(projection.template_native_bytes,
                    recurrence.template_native_bytes) &&
         accumulate(projection.template_source_bytes,
                    recurrence.template_source_bytes) &&
         accumulate(projection.route_count, recurrence.group_count) &&
         accumulate(projection.template_count, recurrence.template_count) &&
         accumulate(projection.route_step_count, recurrence.route_step_count) &&
         accumulate(projection.template_step_count,
                    recurrence.template_step_count) &&
         accumulate(projection.descriptor_set_count,
                    recurrence.descriptor_set_count) &&
         accumulate(projection.descriptor_count, recurrence.descriptor_count) &&
         accumulate(projection.native_allocation_count,
                    recurrence.route_native_allocation_count) &&
         accumulate(projection.native_allocation_count,
                    recurrence.template_native_allocation_count);
}

bool ScalePreparedMapRecurrenceRoutesForContract(
    const PreparedMapRecurrenceReservation &reservation,
    const std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept {
  return scale_map_recurrence_route_reservation(reservation, copies, scaled);
}

} // namespace rund::node::accel::detail
