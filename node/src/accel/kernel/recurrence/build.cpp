#include "../recurrence.hpp"
#include "match.hpp"
#include "source.hpp"

#include <limits>
#include <new>

namespace rund::node::accel::detail {
using rund::kernel::BindingSet;
using rund::kernel::ComputeApi;

namespace {

[[nodiscard]] MapRecurrence Invalid(const char *const reason) noexcept {
  return MapRecurrence{
      .state = MapRecurrenceState::Invalid,
      .reason = reason,
  };
}

enum class RecurrenceMarker : std::uint8_t {
  TopLevel,
  NestedAction,
};

[[nodiscard]] MapRecurrence
Build(const std::span<const BackendBatchEntry> entries,
      const std::span<const std::uint8_t> barriers,
      const RecurrenceMarker marker) {
  const bool marked = marker == RecurrenceMarker::TopLevel
                          ? ExactRecurrenceMarker(entries)
                          : ExactNestedMapRecurrenceMarker(entries);
  if (!marked) {
    return {};
  }
  if (barriers.size() != entries.size() ||
      (marker == RecurrenceMarker::TopLevel && barriers.front() != 0u)) {
    return Invalid("compute_pipeline_recurrence_barrier_invalid");
  }
  for (std::size_t index = 1u; index < barriers.size(); ++index) {
    if (barriers[index] == 0u) {
      return Invalid("compute_pipeline_recurrence_barrier_invalid");
    }
  }

  const BackendRun *const first_run = entries.front().run;
  if (first_run == nullptr || first_run->step_count != 1u ||
      first_run->steps == nullptr ||
      (first_run->resets != nullptr && first_run->resets->size() != 0u)) {
    return {};
  }
  const BoundStep &first = first_run->steps[0];
  if (!BoundStepMatches(first, rund::kernel::NodeKind::Map) ||
      first.control.active() || first.planned->artifact == nullptr ||
      first.step->artifact.kind != first.planned->artifact->kind ||
      first.planned->artifact != &first.step->artifact ||
      !first.step->artifact.metadata.read_routes.empty() ||
      (first.step->artifact.key.api != ComputeApi::Metal &&
       first.step->artifact.key.api != ComputeApi::Vulkan)) {
    return {};
  }
  const BindingSet first_binding = MapBindingFor(first);
  const std::uint64_t output_count = first_binding.resident_outputs.count;
  const std::uint64_t input_count = first_binding.resident_inputs.count;
  if (!first_binding.ok || !first_binding.has_resident_output() ||
      !first_binding.resident_inputs.has_refs() ||
      !first_binding.resident_inputs.has_handles() ||
      !first_binding.resident_outputs.has_handles() || output_count == 0u ||
      output_count > input_count ||
      first.planned->plan.input_buffer_count != input_count ||
      first.planned->plan.output_buffer_count != output_count ||
      !first.map_windows.ok || first.map_windows.size() == 0u) {
    return {};
  }

  BindingSet previous = first_binding;
  const BoundStep *last = &first;
  BindingSet last_binding = first_binding;
  for (std::size_t entry_index = 0u; entry_index < entries.size();
       ++entry_index) {
    const BackendRun *const run = entries[entry_index].run;
    if (run == nullptr || run->step_count != 1u || run->steps == nullptr ||
        (run->resets != nullptr && run->resets->size() != 0u)) {
      return {};
    }
    const BoundStep &step = run->steps[0];
    const BindingSet binding = MapBindingFor(step);
    if (!BoundStepMatches(step, rund::kernel::NodeKind::Map) ||
        step.control.active() || step.planned->artifact == nullptr ||
        step.planned->artifact != &step.step->artifact ||
        !SamePlan(first.planned->plan, step.planned->plan) ||
        !SameArtifact(first.step->artifact, step.step->artifact) ||
        !SameWindows(first, step) || !binding.ok ||
        !SameMapBinding(first_binding, binding) ||
        binding.resident_inputs.count != input_count ||
        binding.resident_outputs.count != output_count ||
        !binding.resident_inputs.has_refs() ||
        !binding.resident_inputs.has_handles() ||
        !binding.resident_outputs.has_refs() ||
        !binding.resident_outputs.has_handles()) {
      return {};
    }
    if (entry_index != 0u) {
      for (std::uint64_t index = 0u; index < output_count; ++index) {
        if (!SameView(previous.resident_outputs, index, binding.resident_inputs,
                      index)) {
          return {};
        }
      }
    }
    for (std::uint64_t index = output_count; index < input_count; ++index) {
      if (!SameView(first_binding.resident_inputs, index,
                    binding.resident_inputs, index)) {
        return {};
      }
      for (std::uint64_t output = 0u; output < output_count; ++output) {
        if (Aliases(binding.resident_inputs, index, binding.resident_outputs,
                    output)) {
          return {};
        }
      }
    }
    for (std::uint64_t left = 0u; left < output_count; ++left) {
      for (std::uint64_t right = left + 1u; right < output_count; ++right) {
        if (Aliases(binding.resident_outputs, left, binding.resident_outputs,
                    right)) {
          return {};
        }
      }
    }
    previous = binding;
    last = &step;
    last_binding = binding;
  }

  MapRecurrence result{};
  result.state = MapRecurrenceState::Invalid;
  result.first = &first;
  result.last = last;
  result.bindings = first_binding;
  result.bindings.resident_outputs = last_binding.resident_outputs;
  result.artifact = first.step->artifact;
  result.plan = first.planned->plan;
  result.windows = first.map_windows.data();
  result.window_count = first.map_windows.size();
  result.iterations = entries.size();
  result.reason = "compute_pipeline_recurrence_source_invalid";
  try {
    if (!TransformSource(result.artifact, input_count, output_count)) {
      return result;
    }
  } catch (const std::bad_alloc &) {
    result.reason = "compute_pipeline_capacity";
    return result;
  }
  result.state = MapRecurrenceState::Ready;
  result.reason = "ok";
  return result;
}

} // namespace

MapRecurrence
BuildMapRecurrence(const std::span<const BackendBatchEntry> entries,
                   const std::span<const std::uint8_t> barriers) {
  return Build(entries, barriers, RecurrenceMarker::TopLevel);
}

MapRecurrence
BuildNestedMapRecurrence(const std::span<const BackendBatchEntry> entries,
                         const std::span<const std::uint8_t> barriers) {
  return Build(entries, barriers, RecurrenceMarker::NestedAction);
}
} // namespace rund::node::accel::detail
