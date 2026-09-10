#include "local.hpp"

namespace node_accel_contract {
namespace {

[[nodiscard]] bool PreparationPlanContract() {
  Fixture terminal{ComputeApi::Metal, ComputeScalar::Lane32};
  const MapRecurrence terminal_recurrence =
      BuildMapRecurrence(terminal.entries, terminal.barriers);
  const MapRecurrencePreparationPlan terminal_plan =
      rund::node::accel::detail::PlanMapRecurrencePreparation(
          terminal.occurrences.front().run, 1u, 0u);
  if (!terminal_recurrence.ready() || !terminal_plan.eligible() ||
      terminal_plan.group_count != 1u ||
      terminal_plan.history_group_count != 0u ||
      terminal_plan.terminal_group_count() != 1u ||
      terminal_plan.input_count != 2u || terminal_plan.output_count != 1u ||
      terminal_plan.window_count != 1u ||
      terminal_plan.terminal_source.exact_source_bytes !=
          terminal_recurrence.source_plan.exact_source_bytes ||
      terminal_plan.terminal_source.source_upper_bytes !=
          terminal_recurrence.source_plan.source_upper_bytes) {
    return false;
  }

  Fixture history{ComputeApi::Vulkan, ComputeScalar::Lane64, false, true};
  const MapRecurrence history_recurrence =
      BuildMapRecurrence(history.entries, history.barriers);
  const MapRecurrencePreparationPlan history_plan =
      rund::node::accel::detail::PlanMapRecurrencePreparation(
          history.occurrences.front().run, 1u, 1u);
  if (!history_recurrence.ready() || !history_plan.eligible() ||
      history_plan.group_count != 1u ||
      history_plan.history_group_count != 1u ||
      history_plan.terminal_group_count() != 0u ||
      history_plan.history_source.exact_source_bytes !=
          history_recurrence.source_plan.exact_source_bytes ||
      history_plan.history_source.source_upper_bytes !=
          history_recurrence.source_plan.source_upper_bytes) {
    return false;
  }

  MapRecurrencePreparationPlan terminal_peer = terminal_plan;
  terminal_peer.group_count = 7u;
  terminal_peer.terminal_template_group_capacity = 14u;
  terminal_peer.outputs[0u].offset_bytes += 4096u;
  if (!SameMapRecurrenceTemplate(terminal_plan, terminal_peer, false)) {
    return false;
  }
  MapRecurrencePreparationPlan bytewise = terminal_plan;
  bytewise.inputs[0u].offset_bytes = 1u;
  MapRecurrencePreparationPlan bytewise_peer = bytewise;
  bytewise_peer.inputs[0u].offset_bytes = 5u;
  const std::array recurrence_routes{terminal_plan, bytewise, bytewise_peer};
  std::uint64_t recurrence_template_count = 0u;
  for (std::size_t index = 0u; index < recurrence_routes.size(); ++index) {
    bool seen = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (SameMapRecurrenceTemplate(recurrence_routes[index],
                                    recurrence_routes[prior], false)) {
        seen = true;
        break;
      }
    }
    recurrence_template_count += seen ? 0u : 1u;
  }
  if (SameMapRecurrenceTemplate(terminal_plan, bytewise, false) ||
      !SameMapRecurrenceTemplate(bytewise, bytewise_peer, false) ||
      recurrence_template_count != 2u) {
    return false;
  }
  terminal_peer.outputs[0u].stride_bytes += 4u;
  if (SameMapRecurrenceTemplate(terminal_plan, terminal_peer, false)) {
    return false;
  }
  MapRecurrencePreparationPlan history_peer = history_plan;
  history_peer.group_count = 7u;
  history_peer.history_group_count = 7u;
  history_peer.history_template_group_capacity = 14u;
  history_peer.outputs[0u].offset_bytes += history_peer.binding_alignment;
  if (!SameMapRecurrenceTemplate(history_plan, history_peer, true)) {
    return false;
  }
  ++history_peer.outputs[0u].count;
  if (SameMapRecurrenceTemplate(history_plan, history_peer, true)) {
    return false;
  }

  const MapRecurrencePreparationPlan invalid =
      rund::node::accel::detail::PlanMapRecurrencePreparation(
          terminal.occurrences.front().run, 0u, 1u);
  if (invalid.ok) {
    return false;
  }
  terminal.occurrences.front().step.map_semantic.recurrence_total = false;
  const MapRecurrencePreparationPlan ineligible =
      rund::node::accel::detail::PlanMapRecurrencePreparation(
          terminal.occurrences.front().run, 1u, 0u);
  return ineligible.ok && !ineligible.eligible() &&
         ineligible.group_count == 0u;
}

[[nodiscard]] bool VulkanPreparationReservationContract() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using rund::node::accel::detail::PlanVulkanPipelineRecurrence;
  using rund::node::accel::detail::PreparedMapRecurrenceReservation;

  Fixture fixture{ComputeApi::Vulkan, ComputeScalar::Lane32};
  MapRecurrencePreparationPlan plan =
      rund::node::accel::detail::PlanMapRecurrencePreparation(
          fixture.occurrences.front().run, 7u, 2u);
  if (!plan.eligible()) {
    return false;
  }
  plan.terminal_template_group_capacity = 10u;
  plan.history_template_group_capacity = 6u;
  PreparedMapRecurrenceReservation reservation{};
  constexpr std::uint64_t descriptor_sets = 16u;
  constexpr std::uint64_t descriptors_per_set = 4u;
  if (!PlanVulkanPipelineRecurrence(plan, reservation).ok ||
      reservation.group_count != 7u || reservation.history_group_count != 2u ||
      reservation.terminal_template_group_capacity != 10u ||
      reservation.history_template_group_capacity != 6u ||
      reservation.descriptor_set_count != descriptor_sets ||
      reservation.descriptor_count != descriptor_sets * descriptors_per_set ||
      reservation.template_native_allocation_count != descriptor_sets + 8u) {
    return false;
  }
  plan.history_template_group_capacity = 1u;
  return !PlanVulkanPipelineRecurrence(plan, reservation).ok;
#else
  return true;
#endif
}

[[nodiscard]] bool MetalPreparationReservationContract() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using rund::node::accel::detail::PlanMetalPipelineRecurrence;
  using rund::node::accel::detail::PreparedMapRecurrenceReservation;

  Fixture fixture{ComputeApi::Metal, ComputeScalar::Lane32};
  const auto plan = [&](const std::uint64_t groups,
                        const std::uint64_t history_groups) {
    return rund::node::accel::detail::PlanMapRecurrencePreparation(
        fixture.occurrences.front().run, groups, history_groups);
  };
  const MapRecurrencePreparationPlan terminal_seven = plan(7u, 0u);
  const MapRecurrencePreparationPlan terminal_seventy = plan(70u, 0u);
  const MapRecurrencePreparationPlan mixed_seven = plan(7u, 2u);
  const MapRecurrencePreparationPlan mixed_seventy = plan(70u, 20u);
  if (!terminal_seven.eligible() || !terminal_seventy.eligible() ||
      !mixed_seven.eligible() || !mixed_seventy.eligible()) {
    return false;
  }

  PreparedMapRecurrenceReservation terminal{};
  PreparedMapRecurrenceReservation terminal_scaled{};
  PreparedMapRecurrenceReservation mixed{};
  PreparedMapRecurrenceReservation mixed_scaled{};
  if (!PlanMetalPipelineRecurrence(terminal_seven, terminal).ok ||
      !PlanMetalPipelineRecurrence(terminal_seventy, terminal_scaled).ok ||
      !PlanMetalPipelineRecurrence(mixed_seven, mixed).ok ||
      !PlanMetalPipelineRecurrence(mixed_seventy, mixed_scaled).ok) {
    return false;
  }

  const auto same_template_budget =
      [](const PreparedMapRecurrenceReservation &left,
         const PreparedMapRecurrenceReservation &right) {
        return left.template_host_bytes == right.template_host_bytes &&
               left.template_native_bytes == right.template_native_bytes &&
               left.template_source_bytes == right.template_source_bytes &&
               left.source_transient_bytes == right.source_transient_bytes &&
               left.template_count == right.template_count &&
               left.template_step_count == right.template_step_count &&
               left.template_native_allocation_count ==
                   right.template_native_allocation_count;
      };
  MapRecurrencePreparationPlan bytewise_terminal = terminal_seven;
  bytewise_terminal.inputs[0u].offset_bytes = 1u;
  PreparedMapRecurrenceReservation bytewise_reservation{};
  if (SameMapRecurrenceTemplate(terminal_seven, bytewise_terminal, false) ||
      !PlanMetalPipelineRecurrence(bytewise_terminal, bytewise_reservation)
           .ok ||
      bytewise_reservation.template_count != 1u ||
      terminal.template_count + bytewise_reservation.template_count != 2u ||
      !same_template_budget(terminal, bytewise_reservation)) {
    return false;
  }
  if (terminal.group_count != 7u || terminal.history_group_count != 0u ||
      terminal.terminal_template_group_capacity != 7u ||
      terminal.history_template_group_capacity != 0u ||
      terminal.route_step_count != 7u || terminal.template_count != 1u ||
      terminal.template_step_count != 1u ||
      terminal.route_native_allocation_count != 7u ||
      terminal.template_native_allocation_count != 2u ||
      terminal_scaled.group_count != 70u ||
      terminal_scaled.terminal_template_group_capacity != 70u ||
      terminal_scaled.history_template_group_capacity != 0u ||
      terminal_scaled.route_step_count != 70u ||
      terminal_scaled.route_native_allocation_count != 70u ||
      !same_template_budget(terminal, terminal_scaled) ||
      terminal_scaled.route_host_bytes != terminal.route_host_bytes * 10u ||
      terminal_scaled.route_native_bytes != terminal.route_native_bytes * 10u) {
    return false;
  }

  if (mixed.group_count != 7u || mixed.history_group_count != 2u ||
      mixed.terminal_template_group_capacity != 5u ||
      mixed.history_template_group_capacity != 2u ||
      mixed.route_step_count != 7u || mixed.template_count != 2u ||
      mixed.template_step_count != 2u ||
      mixed.route_native_allocation_count != 7u ||
      mixed.template_native_allocation_count != 4u ||
      mixed.route_host_bytes !=
          terminal.route_host_bytes +
              2u * static_cast<std::uint64_t>(sizeof(MapRecurrenceHistory)) ||
      mixed.route_native_bytes != terminal.route_native_bytes ||
      mixed_scaled.group_count != 70u ||
      mixed_scaled.history_group_count != 20u ||
      mixed_scaled.terminal_template_group_capacity != 50u ||
      mixed_scaled.history_template_group_capacity != 20u ||
      mixed_scaled.route_step_count != 70u ||
      mixed_scaled.route_native_allocation_count != 70u ||
      !same_template_budget(mixed, mixed_scaled) ||
      mixed_scaled.route_host_bytes != mixed.route_host_bytes * 10u ||
      mixed_scaled.route_native_bytes != mixed.route_native_bytes * 10u) {
    return false;
  }

  MapRecurrencePreparationPlan invalid = mixed_seven;
  invalid.history_group_count = invalid.group_count + 1u;
  PreparedMapRecurrenceReservation rejected{
      .route_host_bytes = 1u,
      .template_host_bytes = 1u,
      .group_count = 1u,
  };
  if (PlanMetalPipelineRecurrence(invalid, rejected).ok ||
      rejected.route_host_bytes != 0u || rejected.template_host_bytes != 0u ||
      rejected.group_count != 0u) {
    return false;
  }
  invalid = mixed_seven;
  invalid.terminal_template_group_capacity =
      invalid.terminal_group_count() - 1u;
  rejected = PreparedMapRecurrenceReservation{
      .route_host_bytes = 1u,
      .template_host_bytes = 1u,
      .group_count = 1u,
  };
  return !PlanMetalPipelineRecurrence(invalid, rejected).ok &&
         rejected.route_host_bytes == 0u &&
         rejected.template_host_bytes == 0u && rejected.group_count == 0u;
#else
  return true;
#endif
}

[[nodiscard]] bool MetalRecurrenceSourceHasOneStorageOwner() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using rund::node::accel::detail::KernelPreparationMode;
  using rund::node::accel::detail::KernelPreparationScope;
  using rund::node::accel::detail::PipelinePrivateMetalSource;
  using rund::node::accel::detail::PipelinePrivateMetalSourceUpperBytes;
  using rund::node::accel::detail::SpecializeMapInPlace;

  Fixture fixture{ComputeApi::Metal, ComputeScalar::Lane32};
  const MapRecurrence recurrence =
      BuildMapRecurrence(fixture.entries, fixture.barriers);
  std::uint64_t specialized_upper = 0u;
  std::uint64_t final_upper = 0u;
  std::uint64_t final_storage_upper = 0u;
  if (!recurrence.ready() || recurrence.canonical_artifact == nullptr ||
      recurrence.source_plan.metadata_storage_upper_bytes == 0u ||
      !rund::node::accel::detail::MapSpecializedSourceUpperBytes(
          recurrence.source_plan.exact_source_bytes,
          recurrence.source_plan.source_upper_bytes, recurrence.plan,
          specialized_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(specialized_upper, 1u, true,
                                            final_upper) ||
      !rund::node::accel::detail::backend_source_recipe::
          string_external_storage_upper_bytes(final_upper,
                                              final_storage_upper)) {
    return false;
  }

  rund::kernel::LoweringArtifact artifact{};
  if (!rund::node::accel::detail::MaterializeMapRecurrenceArtifact(
          *recurrence.canonical_artifact, recurrence.source_plan,
          recurrence.plan.input_buffer_count,
          recurrence.plan.output_buffer_count, {}, artifact, final_upper)) {
    return false;
  }
  const char *const storage = artifact.source_text.data();
  const std::size_t capacity = artifact.source_text.capacity();
  rund::kernel::LoweringArtifact specialized =
      SpecializeMapInPlace(std::move(artifact), recurrence.plan,
                           recurrence.bindings, 1u, final_upper);
  if (!specialized.ok || specialized.source_text.data() != storage ||
      specialized.source_text.capacity() != capacity ||
      specialized.metadata.retained_dynamic_memory_bytes() != 0u ||
      specialized.retained_dynamic_memory_bytes() > final_storage_upper) {
    return false;
  }

  const KernelPreparationScope preparation{
      KernelPreparationMode::PipelinePrivate};
  std::string guarded = PipelinePrivateMetalSource(
      std::move(specialized.source_text), final_upper);
  return !guarded.empty() && guarded.size() <= final_upper &&
         guarded.data() == storage && guarded.capacity() == capacity;
#else
  return true;
#endif
}

} // namespace

bool MapRecurrencePreparationContract() {
  return PreparationPlanContract() &&
         MetalPreparationReservationContract() &&
         VulkanPreparationReservationContract() &&
         MetalRecurrenceSourceHasOneStorageOwner();
}

} // namespace node_accel_contract
