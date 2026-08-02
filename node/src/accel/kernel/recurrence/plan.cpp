#include "plan.hpp"

#include "../backend/template_plan.hpp"
#include "../plan/local.hpp"
#include "source.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <limits>

namespace rund::node::accel::detail {
namespace {

using BindingIdentity = PreparedKernelProgramBindingIdentity;

[[nodiscard]] MapRecurrencePreparationPlan
Invalid(const char *const reason) noexcept {
  return MapRecurrencePreparationPlan{
      .reason =
          reason == nullptr ? "compute_pipeline_recurrence_invalid" : reason,
  };
}

[[nodiscard]] MapRecurrencePreparationPlan Ineligible() noexcept {
  return MapRecurrencePreparationPlan{.ok = true, .reason = "ok"};
}

[[nodiscard]] bool ValidIdentity(const BindingIdentity &identity,
                                 const std::uint32_t usage) noexcept {
  return identity.element_bytes != 0u && identity.count != 0u &&
         identity.stride_bytes >= identity.element_bytes &&
         identity.usage == usage;
}

[[nodiscard]] bool
StaticCandidate(const KernelExecution &execution,
                const KernelExecutionStep &step,
                const rund::kernel::ComputePlan &plan) noexcept {
  const auto api = step.artifact.key.api;
  return execution.steps.size() == 1u && execution.resets.empty() &&
         step.kind() == rund::kernel::NodeKind::Map &&
         step.map_semantic.recurrence_total && !step.control.has_count() &&
         !step.control.has_predicate() && step.artifact.ok &&
         step.artifact.metadata.ok &&
         step.artifact.metadata.read_routes.empty() && plan.ok &&
         plan.dispatch_count != 0u && plan.input_buffer_count != 0u &&
         plan.output_buffer_count != 0u &&
         plan.output_buffer_count <= plan.input_buffer_count &&
         plan.input_buffer_count <= MapRecurrencePreparationPlan::Capacity &&
         plan.output_buffer_count <= MapRecurrencePreparationPlan::Capacity &&
         step.artifact.metadata.read_count == plan.input_buffer_count &&
         step.artifact.metadata.write_count == plan.output_buffer_count &&
         (api == rund::kernel::ComputeApi::Metal ||
          api == rund::kernel::ComputeApi::Vulkan);
}

[[nodiscard]] bool
AppendIdentity(MapRecurrencePreparationPlan &result,
               const rund::kernel::ComputeBindingAccess access,
               const BindingIdentity &identity) noexcept {
  if (access == rund::kernel::ComputeBindingAccess::Read) {
    if (result.input_count >= result.inputs.size() ||
        !ValidIdentity(identity, rund::kernel::kResidentUsageRead)) {
      return false;
    }
    result.inputs[result.input_count++] = identity;
    return true;
  }
  if (access == rund::kernel::ComputeBindingAccess::Write) {
    if (result.output_count >= result.outputs.size() ||
        !ValidIdentity(identity, rund::kernel::kResidentUsageWrite)) {
      return false;
    }
    result.outputs[result.output_count++] = identity;
    return true;
  }
  return false;
}

[[nodiscard]] bool
FinalizeSources(MapRecurrencePreparationPlan &result) noexcept {
  if (result.canonical_artifact == nullptr || result.authority == nullptr ||
      result.group_count == 0u || result.binding_alignment == 0u ||
      result.history_group_count > result.group_count ||
      (result.terminal_group_count() == 0u
           ? result.terminal_template_group_capacity != 0u
           : result.terminal_template_group_capacity <
                 result.terminal_group_count()) ||
      (result.history_group_count == 0u
           ? result.history_template_group_capacity != 0u
           : result.history_template_group_capacity <
                 result.history_group_count) ||
      result.input_count != result.plan.input_buffer_count ||
      result.output_count != result.plan.output_buffer_count ||
      result.window_count != result.plan.dispatch_count) {
    return false;
  }

  if (result.terminal_group_count() != 0u) {
    result.terminal_source = PlanMapRecurrenceSource(
        *result.canonical_artifact, result.input_count, result.output_count);
    if (!result.terminal_source.ok) {
      result.reason = result.terminal_source.reason;
      return false;
    }
  }

  if (result.history_group_count != 0u) {
    std::array<std::uint64_t, MapRecurrencePreparationPlan::Capacity> pitches{};
    for (std::uint32_t index = 0u; index < result.output_count; ++index) {
      const BindingIdentity &output = result.outputs[index];
      if (!rund::kernel::checked::mul(output.count, output.stride_bytes,
                                      pitches[index])) {
        result.reason = "compute_pipeline_capacity";
        return false;
      }
    }
    result.history_source = PlanMapRecurrenceSource(
        *result.canonical_artifact, result.input_count, result.output_count,
        std::span<const std::uint64_t>{pitches.data(), result.output_count});
    if (!result.history_source.ok) {
      result.reason = result.history_source.reason;
      return false;
    }
  }
  result.ok = true;
  result.reason = "ok";
  return true;
}

[[nodiscard]] BindingIdentity
IdentityFor(const rund::kernel::ResidentBufferRef *const ref) noexcept {
  return ref == nullptr ? BindingIdentity{}
                        : BindingIdentity{
                              .offset_bytes = ref->offset_bytes,
                              .element_bytes = ref->element_bytes,
                              .stride_bytes = ref->stride_bytes,
                              .count = ref->count,
                              .usage = ref->usage,
                          };
}

} // namespace

MapRecurrencePreparationPlan
PlanMapRecurrencePreparation(const KernelExecution &execution,
                             const PreparedKernelProgramRoute &route) noexcept {
  if (route.map_recurrence_history_group_count >
      route.map_recurrence_group_count) {
    return Invalid("accel_kernel_run_invalid");
  }
  if (route.map_recurrence_group_count == 0u) {
    return Ineligible();
  }
  if (!execution.admission.check.ok || execution.steps.empty() ||
      route.kernel == nullptr || route.tile_count == 0u ||
      route.program_bindings.size() != execution.graph_roles.size()) {
    return Invalid("accel_kernel_run_invalid");
  }

  const KernelExecutionStep &step = execution.steps.front();
  const rund::AccelRun run{.tile_count = route.tile_count};
  const rund::kernel::ComputePlan plan = PlanStep(execution, step, run, 0u);
  if (!StaticCandidate(execution, step, plan)) {
    return Ineligible();
  }
  const auto &metadata = step.artifact.metadata;
  if (!step.graph_binding_indices_ok || !step.graph_binding_indices.valid() ||
      metadata.binding_accesses.size() != step.graph_binding_indices.size()) {
    return Invalid("compute_binding_mismatch");
  }

  MapRecurrencePreparationPlan result{
      .authority = &step,
      .canonical_artifact = &step.artifact,
      .plan = plan,
      .window_count = plan.dispatch_count,
      .group_count = route.map_recurrence_group_count,
      .history_group_count = route.map_recurrence_history_group_count,
      .terminal_template_group_capacity =
          route.map_recurrence_group_count -
          route.map_recurrence_history_group_count,
      .history_template_group_capacity =
          route.map_recurrence_history_group_count,
      .binding_alignment =
          plan.api == rund::kernel::ComputeApi::Metal
              ? 1u
              : execution.admission.frozen_caps.storage_alignment,
  };
  for (std::size_t local = 0u; local < metadata.binding_accesses.size();
       ++local) {
    const std::uint64_t binding = step.graph_binding_indices[local];
    if (binding >= route.program_bindings.size() ||
        !AppendIdentity(result, metadata.binding_accesses[local],
                        route.program_bindings[binding])) {
      return Invalid("compute_binding_mismatch");
    }
  }
  if (!FinalizeSources(result)) {
    return Invalid(result.reason);
  }
  return result;
}

MapRecurrencePreparationPlan
PlanMapRecurrencePreparation(const BackendRun &run,
                             const std::uint64_t group_count,
                             const std::uint64_t history_group_count) noexcept {
  if (history_group_count > group_count) {
    return Invalid("accel_kernel_run_invalid");
  }
  if (group_count == 0u) {
    return Ineligible();
  }
  if (run.execution == nullptr || run.steps == nullptr ||
      run.step_count == 0u) {
    return Invalid("accel_kernel_run_invalid");
  }
  const KernelExecution &execution = *run.execution;
  const BoundStep &bound = run.steps[0u];
  if (bound.step == nullptr || bound.planned == nullptr ||
      bound.planned->artifact != &bound.step->artifact ||
      !StaticCandidate(execution, *bound.step, bound.planned->plan)) {
    return Ineligible();
  }
  const rund::kernel::BindingSet bindings = MapBindingFor(bound);
  if (!bindings.ok || !bound.map_windows.ok ||
      bound.map_windows.size() != bound.planned->plan.dispatch_count ||
      bindings.resident_inputs.count !=
          bound.planned->plan.input_buffer_count ||
      bindings.resident_outputs.count !=
          bound.planned->plan.output_buffer_count ||
      !bindings.resident_inputs.has_refs() ||
      !bindings.resident_outputs.has_refs()) {
    return Invalid("compute_binding_mismatch");
  }

  MapRecurrencePreparationPlan result{
      .authority = bound.step,
      .canonical_artifact = &bound.step->artifact,
      .plan = bound.planned->plan,
      .window_count = bound.map_windows.size(),
      .group_count = group_count,
      .history_group_count = history_group_count,
      .terminal_template_group_capacity = group_count - history_group_count,
      .history_template_group_capacity = history_group_count,
      .binding_alignment =
          bound.planned->plan.api == rund::kernel::ComputeApi::Metal
              ? 1u
              : execution.admission.frozen_caps.storage_alignment,
  };
  for (std::uint64_t index = 0u; index < bindings.resident_inputs.count;
       ++index) {
    if (!AppendIdentity(result, rund::kernel::ComputeBindingAccess::Read,
                        IdentityFor(bindings.resident_inputs.ref(index)))) {
      return Invalid("compute_binding_mismatch");
    }
  }
  for (std::uint64_t index = 0u; index < bindings.resident_outputs.count;
       ++index) {
    if (!AppendIdentity(result, rund::kernel::ComputeBindingAccess::Write,
                        IdentityFor(bindings.resident_outputs.ref(index)))) {
      return Invalid("compute_binding_mismatch");
    }
  }
  if (!FinalizeSources(result)) {
    return Invalid(result.reason);
  }
  return result;
}

bool SameMapRecurrenceTemplate(const MapRecurrencePreparationPlan &left,
                               const MapRecurrencePreparationPlan &right,
                               const bool history) noexcept {
  const bool left_present = history ? left.history_group_count != 0u
                                    : left.terminal_group_count() != 0u;
  const bool right_present = history ? right.history_group_count != 0u
                                     : right.terminal_group_count() != 0u;
  if (!left.eligible() || !right.eligible() || !left_present ||
      !right_present || left.authority != right.authority ||
      left.canonical_artifact != right.canonical_artifact ||
      left.binding_alignment == 0u ||
      left.binding_alignment != right.binding_alignment ||
      left.window_count != right.window_count ||
      left.input_count != right.input_count ||
      left.output_count != right.output_count ||
      !backend_template_plan::same_plan(left.plan, right.plan)) {
    return false;
  }

  const MapRecurrenceSourcePlan &left_source =
      history ? left.history_source : left.terminal_source;
  const MapRecurrenceSourcePlan &right_source =
      history ? right.history_source : right.terminal_source;
  if (!left_source.ok || !right_source.ok ||
      left_source.exact_source_bytes != right_source.exact_source_bytes ||
      left_source.source_upper_bytes != right_source.source_upper_bytes ||
      left_source.source_storage_upper_bytes !=
          right_source.source_storage_upper_bytes ||
      left_source.metadata_storage_upper_bytes !=
          right_source.metadata_storage_upper_bytes ||
      left_source.history != right_source.history) {
    return false;
  }

  const auto same_layout = [&](const BindingIdentity &a,
                               const BindingIdentity &b,
                               const bool compare_pitch) noexcept {
    if (a.stride_bytes != b.stride_bytes ||
        a.offset_bytes % left.binding_alignment !=
            b.offset_bytes % right.binding_alignment) {
      return false;
    }
    if (!compare_pitch) {
      return true;
    }
    std::uint64_t a_pitch = 0u;
    std::uint64_t b_pitch = 0u;
    return rund::kernel::checked::mul(a.count, a.stride_bytes, a_pitch) &&
           rund::kernel::checked::mul(b.count, b.stride_bytes, b_pitch) &&
           a_pitch == b_pitch;
  };
  for (std::size_t index = 0u; index < left.input_count; ++index) {
    if (!same_layout(left.inputs[index], right.inputs[index], false)) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.output_count; ++index) {
    if (!same_layout(left.outputs[index], right.outputs[index], history)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
