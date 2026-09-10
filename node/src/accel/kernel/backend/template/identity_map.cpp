#include "identity_internal.hpp"

#include "../../step/map/stride.hpp"

#include <cstddef>

namespace rund::node::accel::detail::backend_template_plan {
namespace {

void mix_map_specialization(std::uint64_t &hash,
                            const std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
}

void mix_map_specialization(
    MapSpecializationFingerprint &fingerprint,
    const PreparedKernelProgramBindingIdentity &identity,
    const rund::kernel::ComputeApi api) noexcept {
  mix_map_specialization(fingerprint.hi, identity.offset_bytes);
  mix_map_specialization(fingerprint.lo, identity.element_bytes);
  mix_map_specialization(fingerprint.hi, identity.stride_bytes);
  mix_map_specialization(fingerprint.lo, identity.count);
  mix_map_specialization(fingerprint.hi, identity.usage);
  if (api == rund::kernel::ComputeApi::Metal) {
    mix_map_specialization(fingerprint.lo,
                           static_cast<std::uint64_t>(MetalMapBindingWordClass(
                               identity.offset_bytes, identity.stride_bytes)));
  }
}

} // namespace

bool same_map_binding_identity(
    const PreparedKernelProgramBindingIdentity &left,
    const PreparedKernelProgramBindingIdentity &right,
    const std::uint64_t alignment,
    const rund::kernel::ComputeApi api) noexcept {
  return alignment != 0u && left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage &&
         left.offset_bytes % alignment == right.offset_bytes % alignment &&
         (api != rund::kernel::ComputeApi::Metal ||
          MetalMapBindingWordClass(left.offset_bytes, left.stride_bytes) ==
              MetalMapBindingWordClass(right.offset_bytes, right.stride_bytes));
}

bool same_program_map_specialization(const KernelExecution &execution,
                                     const PreparedKernelProgramRoute &left,
                                     const PreparedKernelProgramRoute &right,
                                     const std::uint64_t alignment) noexcept {
  if (alignment == 0u ||
      left.program_bindings.size() != execution.graph_roles.size() ||
      right.program_bindings.size() != execution.graph_roles.size()) {
    return false;
  }
  for (const KernelExecutionStep &step : execution.steps) {
    if (step.kind() != rund::kernel::NodeKind::Map) {
      continue;
    }
    const auto &accesses = step.artifact.metadata.binding_accesses;
    if (!step.graph_binding_indices_ok ||
        accesses.size() > step.graph_binding_indices.size()) {
      return false;
    }
    for (std::size_t local = 0u; local < accesses.size(); ++local) {
      const std::uint64_t binding = step.graph_binding_indices[local];
      if (binding >= left.program_bindings.size() ||
          !same_map_binding_identity(left.program_bindings[binding],
                                     right.program_bindings[binding], alignment,
                                     step.artifact.key.api)) {
        return false;
      }
    }
  }
  return true;
}

MapSpecializationFingerprint program_map_specialization_fingerprint(
    const KernelExecution &execution,
    const PreparedKernelProgramRoute &route) noexcept {
  MapSpecializationFingerprint fingerprint{};
  if (route.program_bindings.size() != execution.graph_roles.size()) {
    return fingerprint;
  }
  for (std::size_t step_index = 0u; step_index < execution.steps.size();
       ++step_index) {
    const KernelExecutionStep &step = execution.steps[step_index];
    if (step.kind() != rund::kernel::NodeKind::Map) {
      continue;
    }
    const auto &metadata = step.artifact.metadata;
    const auto &accesses = metadata.binding_accesses;
    if (!step.graph_binding_indices_ok ||
        accesses.size() > step.graph_binding_indices.size()) {
      return fingerprint;
    }
    mix_map_specialization(fingerprint.hi, step_index);
    mix_map_specialization(fingerprint.lo, metadata.read_count);
    mix_map_specialization(fingerprint.hi, metadata.write_count);
    for (const rund::kernel::ComputeBindingAccess selected :
         {rund::kernel::ComputeBindingAccess::Read,
          rund::kernel::ComputeBindingAccess::Write}) {
      for (std::size_t local = 0u; local < accesses.size(); ++local) {
        if (accesses[local] != selected) {
          continue;
        }
        const std::uint64_t binding = step.graph_binding_indices[local];
        if (binding >= route.program_bindings.size()) {
          return fingerprint;
        }
        mix_map_specialization(fingerprint, route.program_bindings[binding],
                               step.artifact.key.api);
      }
    }
  }
  fingerprint.ok = true;
  return fingerprint;
}

MapSpecializationFingerprint
runtime_map_specialization_fingerprint(const BackendRun &run) noexcept {
  MapSpecializationFingerprint fingerprint{};
  if (run.steps == nullptr || run.step_count == 0u) {
    return fingerprint;
  }
  for (std::size_t step_index = 0u; step_index < run.step_count; ++step_index) {
    const BoundStep &step = run.steps[step_index];
    if (step.step == nullptr || step.planned == nullptr) {
      return fingerprint;
    }
    if (step.step->kind() != rund::kernel::NodeKind::Map) {
      continue;
    }
    const rund::kernel::BindingSet bindings = MapBindingFor(step);
    const auto &metadata = step.step->artifact.metadata;
    if (!bindings.ok || bindings.resident_inputs.count != metadata.read_count ||
        bindings.resident_outputs.count != metadata.write_count) {
      return fingerprint;
    }
    mix_map_specialization(fingerprint.hi, step_index);
    mix_map_specialization(fingerprint.lo, metadata.read_count);
    mix_map_specialization(fingerprint.hi, metadata.write_count);
    for (const rund::kernel::ResidentBindingRange range :
         {bindings.resident_inputs, bindings.resident_outputs}) {
      for (std::uint64_t index = 0u; index < range.count; ++index) {
        const rund::kernel::ResidentBufferRef *const ref = range.ref(index);
        if (ref == nullptr) {
          return fingerprint;
        }
        mix_map_specialization(fingerprint,
                               PreparedKernelProgramBindingIdentity{
                                   .offset_bytes = ref->offset_bytes,
                                   .element_bytes = ref->element_bytes,
                                   .stride_bytes = ref->stride_bytes,
                                   .count = ref->count,
                                   .usage = ref->usage,
                               },
                               step.planned->plan.api);
      }
    }
  }
  fingerprint.ok = true;
  return fingerprint;
}

} // namespace rund::node::accel::detail::backend_template_plan
