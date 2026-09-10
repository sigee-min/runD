#include "internal.hpp"

#include "../../../../kernel/backend/template/identity.hpp"
#include "../../../../kernel/recurrence/plan.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
SameRecurrenceSourcePlan(const MapRecurrenceSourcePlan &left,
                         const MapRecurrenceSourcePlan &right) noexcept {
  return left.exact_source_bytes == right.exact_source_bytes &&
         left.source_upper_bytes == right.source_upper_bytes &&
         left.source_storage_upper_bytes == right.source_storage_upper_bytes &&
         left.metadata_storage_upper_bytes ==
             right.metadata_storage_upper_bytes &&
         left.history == right.history && left.ok == right.ok;
}

[[nodiscard]] bool
SameIdentityLayout(const PreparedKernelProgramBindingIdentity &identity,
                   const rund::kernel::ResidentBufferRef *const ref,
                   const std::uint64_t alignment,
                   const std::uint64_t count_divisor) noexcept {
  return ref != nullptr && alignment != 0u && count_divisor != 0u &&
         ref->count % count_divisor == 0u &&
         identity.offset_bytes % alignment == ref->offset_bytes % alignment &&
         identity.element_bytes == ref->element_bytes &&
         identity.stride_bytes == ref->stride_bytes &&
         identity.count == ref->count / count_divisor &&
         identity.usage == ref->usage;
}

} // namespace

std::span<const std::uint64_t>
RecurrenceHistoryPitches(const MapRecurrence &recurrence) noexcept {
  return recurrence.history == nullptr ? std::span<const std::uint64_t>{}
                                       : recurrence.history->pitches();
}

bool ValidRecurrence(const MapRecurrence &recurrence) noexcept {
  const std::span<const std::uint64_t> pitches =
      RecurrenceHistoryPitches(recurrence);
  return recurrence.ready() && recurrence.first != nullptr &&
         recurrence.last != nullptr && recurrence.first->step != nullptr &&
         recurrence.canonical_artifact != nullptr &&
         recurrence.canonical_artifact == &recurrence.first->step->artifact &&
         recurrence.canonical_artifact->ok && recurrence.source_plan.ok &&
         recurrence.source_plan.history == !pitches.empty() &&
         (!recurrence.source_plan.history ||
          pitches.size() == recurrence.plan.output_buffer_count) &&
         recurrence.plan.ok &&
         recurrence.plan.api == rund::kernel::ComputeApi::Vulkan &&
         recurrence.plan.input_buffer_count ==
             recurrence.bindings.resident_inputs.count &&
         recurrence.plan.output_buffer_count ==
             recurrence.bindings.resident_outputs.count &&
         recurrence.bindings.resident_inputs.has_refs() &&
         recurrence.bindings.resident_inputs.has_handles() &&
         recurrence.bindings.resident_outputs.has_refs() &&
         recurrence.bindings.resident_outputs.has_handles() &&
         recurrence.windows != nullptr && recurrence.window_count != 0u &&
         recurrence.window_count == recurrence.plan.dispatch_count &&
         recurrence.iterations >= 2u &&
         recurrence.iterations <= std::numeric_limits<std::uint32_t>::max() &&
         !recurrence.first->control.active();
}

bool RuntimeRecurrenceMatchesPreparedPlan(
    const MapRecurrencePreparationPlan &planned,
    const MapRecurrence &recurrence, const std::uint64_t route_group_count,
    const std::uint64_t route_history_group_count,
    const std::uint64_t alignment) noexcept {
  if (!planned.eligible() || planned.authority != recurrence.first->step ||
      planned.canonical_artifact != recurrence.canonical_artifact ||
      planned.group_count != route_group_count ||
      planned.history_group_count != route_history_group_count ||
      planned.binding_alignment != alignment ||
      planned.window_count != recurrence.window_count ||
      !backend_template_plan::same_plan(planned.plan, recurrence.plan) ||
      planned.input_count != recurrence.bindings.resident_inputs.count ||
      planned.output_count != recurrence.bindings.resident_outputs.count) {
    return false;
  }
  const MapRecurrenceSourcePlan &expected_source = recurrence.history == nullptr
                                                       ? planned.terminal_source
                                                       : planned.history_source;
  if (!SameRecurrenceSourcePlan(expected_source, recurrence.source_plan)) {
    return false;
  }
  for (std::size_t index = 0u; index < planned.input_count; ++index) {
    if (!SameIdentityLayout(planned.inputs[index],
                            recurrence.bindings.resident_inputs.ref(index),
                            alignment, 1u)) {
      return false;
    }
  }
  const std::uint64_t output_divisor =
      recurrence.history == nullptr ? 1u : recurrence.iterations;
  for (std::size_t index = 0u; index < planned.output_count; ++index) {
    if (!SameIdentityLayout(planned.outputs[index],
                            recurrence.bindings.resident_outputs.ref(index),
                            alignment, output_divisor)) {
      return false;
    }
  }
  return true;
}

bool RuntimeRecurrenceMatchesPlan(
    const BackendRun &owner, const MapRecurrence &recurrence,
    const std::uint64_t route_group_count,
    const std::uint64_t route_history_group_count,
    const std::uint64_t alignment) noexcept {
  const MapRecurrencePreparationPlan planned = PlanMapRecurrencePreparation(
      owner, route_group_count, route_history_group_count);
  return RuntimeRecurrenceMatchesPreparedPlan(
      planned, recurrence, route_group_count, route_history_group_count,
      alignment);
}

bool SameRuntimeRecurrenceTemplate(
    const BackendRun &left_owner, const MapRecurrence &left,
    const BackendRun &right_owner, const MapRecurrence &right,
    const std::uint64_t alignment) noexcept {
  const bool history = left.history != nullptr;
  if (left_owner.pick == nullptr || left_owner.pick != right_owner.pick ||
      left_owner.ops == nullptr || left_owner.ops != right_owner.ops ||
      history != (right.history != nullptr)) {
    return false;
  }
  const MapRecurrencePreparationPlan left_plan =
      PlanMapRecurrencePreparation(left_owner, 1u, history ? 1u : 0u);
  const MapRecurrencePreparationPlan right_plan =
      PlanMapRecurrencePreparation(right_owner, 1u, history ? 1u : 0u);
  return left_plan.binding_alignment == alignment &&
         right_plan.binding_alignment == alignment &&
         SameMapRecurrenceTemplate(left_plan, right_plan, history);
}

bool ValidRecurrenceReservation(
    const PreparedKernelTemplateRegistry &registry,
    const PreparedPipelineStatusLayout &status,
    const std::uint64_t group_count, const std::uint64_t history_group_count,
    const std::uint64_t terminal_template_group_capacity,
    const std::uint64_t history_template_group_capacity,
    const std::uint64_t template_count) noexcept {
  if (group_count == 0u || history_group_count > group_count ||
      (status.generation_stride != 1u && status.generation_stride != 2u) ||
      !registry.limit.ok) {
    return false;
  }
  std::uint64_t expected_groups = 0u;
  std::uint64_t expected_history = 0u;
  std::uint64_t expected_terminal = 0u;
  if (!rund::kernel::checked::mul(group_count, status.generation_stride,
                                  expected_groups) ||
      !rund::kernel::checked::mul(history_group_count, status.generation_stride,
                                  expected_history) ||
      expected_history > expected_groups) {
    return false;
  }
  expected_terminal = expected_groups - expected_history;
  const PreparedMapRecurrenceReservation &limit = registry.limit.map_recurrence;
  const PreparedMapRecurrenceReservation &consumed =
      registry.reservation.map_recurrence;
  if (limit.group_count != expected_groups ||
      limit.history_group_count != expected_history ||
      limit.terminal_template_group_capacity != expected_terminal ||
      limit.history_template_group_capacity != expected_history ||
      limit.template_count != template_count ||
      terminal_template_group_capacity != expected_terminal ||
      history_template_group_capacity != expected_history ||
      limit.route_step_count != expected_groups ||
      consumed.template_count != limit.template_count ||
      consumed.terminal_template_group_capacity !=
          limit.terminal_template_group_capacity ||
      consumed.history_template_group_capacity !=
          limit.history_template_group_capacity ||
      consumed.group_count < group_count ||
      consumed.group_count > expected_groups ||
      consumed.group_count % group_count != 0u) {
    return false;
  }
  const std::uint64_t consumed_streams = consumed.group_count / group_count;
  std::uint64_t consumed_history = 0u;
  return consumed_streams != 0u &&
         consumed_streams <= status.generation_stride &&
         rund::kernel::checked::mul(history_group_count, consumed_streams,
                                    consumed_history) &&
         consumed.history_group_count == consumed_history &&
         consumed.route_step_count == consumed.group_count;
}

#endif

} // namespace rund::node::accel::detail
