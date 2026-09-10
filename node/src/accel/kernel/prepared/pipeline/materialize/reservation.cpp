#include "../../interface/api.hpp"
#include "../../run.hpp"
#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::add;

namespace {

[[nodiscard]] bool locate_template_step_capacity_failure(
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const std::uint64_t capacity,
    PreparedPipelineFailureContext &failure) noexcept {
  std::uint64_t admitted_steps = 0u;
  for (std::size_t index = 0u; index < runs.size(); ++index) {
    const PreparedKernelRun *const item = runs[index];
    const auto *const state =
        item == nullptr
            ? nullptr
            : static_cast<const prepared::RunState *>(item->owner.get());
    if (state == nullptr || state->bound.run.ops == nullptr ||
        state->bound.run.ops->same_pipeline_template == nullptr) {
      return false;
    }
    bool duplicate_route = false;
    bool template_seen = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      const PreparedKernelRun *const previous = runs[prior];
      const auto *const previous_state =
          previous == nullptr
              ? nullptr
              : static_cast<const prepared::RunState *>(previous->owner.get());
      duplicate_route =
          duplicate_route || (previous != nullptr && item != nullptr &&
                              previous->owner.get() == item->owner.get());
      template_seen =
          template_seen || (previous_state != nullptr &&
                            state->bound.run.ops->same_pipeline_template(
                                state->bound.run, previous_state->bound.run));
    }
    if (duplicate_route || template_seen) {
      continue;
    }
    const std::uint64_t step_count = state->bound.run.step_count;
    if (admitted_steps > capacity || step_count > capacity - admitted_steps) {
      const std::uint64_t crossing =
          admitted_steps >= capacity ? 0u : capacity - admitted_steps;
      const std::size_t step_index =
          crossing < step_count ? static_cast<std::size_t>(crossing) : 0u;
      if (index <= std::numeric_limits<std::uint32_t>::max() &&
          index < recurrences.size()) {
        failure.compact_template_node_route(static_cast<std::uint32_t>(index),
                                            recurrences[index],
                                            state->bound.run, step_index);
      }
      return true;
    }
    admitted_steps += step_count;
  }
  return false;
}

} // namespace

const char *admit_pipeline_resources(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    const std::span<const BackendPublish> publications,
    const std::uint32_t declared_step_count,
    const std::uint32_t generation_stride, const bool profile_steps,
    PreparedKernelTemplateRegistry *const templates,
    PipelineMaterializationDraft &draft,
    PipelineBudgetTransaction &budget_transaction) noexcept {
  draft.failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  if (templates == nullptr || !templates->limit.ok) {
    return "accel_kernel_template_invalid";
  }
  PreparedKernelTemplateRegistry &registry = *templates;
  const PreparedKernelPipelineShape runtime_shape =
      runtime_pipeline_shape(publications, recurrences, declared_step_count,
                             generation_stride, profile_steps);
  PreparedKernelPipelineReservation &reservation = draft.reservation;
  reservation = PlanPreparedKernelPipeline(context, runs, recurrences,
                                           runtime_shape, &registry);
  if (!reservation.ok) {
    return reservation.reason;
  }
  if (reservation.template_step_count > reservation.template_step_capacity) {
    (void)locate_template_step_capacity_failure(
        runs, recurrences, reservation.template_step_capacity, draft.failure);
    return PreparedPipelineTemplateStepCapacityReasonKey;
  }
  if (reservation.fingerprint_hi != registry.limit.fingerprint_hi ||
      reservation.fingerprint_lo != registry.limit.fingerprint_lo) {
    return "accel_kernel_template_invalid";
  }
  const PreparedKernelPipelineReservation structure =
      plan_pipeline_structure(recurrences);
  PreparedKernelPipelineReservation backend_structure{};
  backend_structure.occurrence_count = structure.occurrence_count;
  backend_structure.window_count = structure.window_count;
  backend_structure.nested_group_count = structure.nested_group_count;
  const rund::AccelCheck backend_planned =
      structure.ok ? plan_runtime_backend_structure(
                         context, runs, recurrences, publications,
                         profile_steps ? declared_step_count : 0u,
                         profile_steps ? structure.occurrence_count : 0u,
                         backend_structure)
                   : rund::AccelCheck{false, structure.reason};
  backend_structure.occurrence_count = 0u;
  backend_structure.window_count = 0u;
  backend_structure.nested_group_count = 0u;
  PreparedKernelPipelineReservation charged_structure = structure;
  if (!structure.ok || !backend_planned.ok ||
      !accumulate_reservation(charged_structure, backend_structure) ||
      !accumulate_reservation(reservation, charged_structure) ||
      !accumulate(reservation.host_bytes, reservation.host_transient_bytes)) {
    return !structure.ok ? structure.reason
                         : (backend_planned.reason == nullptr
                                ? "compute_pipeline_capacity"
                                : backend_planned.reason);
  }
  if (!PreparedKernelPipelineReservationWithin(reservation, registry.limit)) {
    return "compute_pipeline_capacity";
  }

  std::uint64_t demand_route_count = 0u;
  std::uint64_t demand_template_count = 0u;
  std::uint64_t complete_route_count = 0u;
  std::uint64_t complete_template_count = 0u;
  if (!plan_backend_template_route_demands(
          runs, generation_stride,
          std::span<BackendTemplateRouteDemand>{
              draft.template_route_demands.data(), runs.size()},
          demand_route_count, demand_template_count) ||
      !add(demand_route_count, reservation.map_recurrence.group_count,
           complete_route_count) ||
      !add(demand_template_count, reservation.map_recurrence.template_count,
           complete_template_count) ||
      complete_route_count != reservation.route_count ||
      complete_template_count != reservation.template_count) {
    return "accel_kernel_template_invalid";
  }

  const rund::AccelCheck registry_bound =
      BindPreparedKernelTemplateRegistry(context.api, context.id, registry);
  if (!registry_bound.ok) {
    return registry_bound.reason;
  }
  const rund::AccelCheck budget = reserve_pipeline_budget(
      registry, runs, recurrences, charged_structure,
      reservation.map_recurrence, generation_stride, budget_transaction);
  if (!budget.ok) {
    return budget.reason;
  }
  return nullptr;
}

} // namespace rund::node::accel::detail
