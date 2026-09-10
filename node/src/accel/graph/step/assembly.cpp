#include "assembly.hpp"

#include "map_semantic.hpp"

#include <string>
#include <utility>

namespace rund::node::accel::detail::step {
namespace {

[[nodiscard]] bool RetainDeviceVsmMapAdmission(
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const MapSemantic &semantic) noexcept {
  return artifact.ok && input.ok && input.key == artifact.key &&
         ((artifact.key.scalar == rund::kernel::ComputeScalar::Lane32 &&
           artifact.key.domain == rund::kernel::ComputeDomain::U32) ||
          (artifact.key.scalar == rund::kernel::ComputeScalar::Lane64 &&
           artifact.key.domain == rund::kernel::ComputeDomain::U64)) &&
         artifact.metadata.read_count != 0u &&
         artifact.metadata.write_count == 1u && semantic.recurrence_total;
}

} // namespace

KernelExecutionStep AssembleKernelExecutionStep(
    rund::kernel::LoweringArtifact artifact,
    rund::kernel::compute_lowering_detail::ComputeInputAdmission input,
    KernelBindingIndices binding_indices, Operation operation,
    const std::uint64_t primitive_hash_hi,
    const std::uint64_t primitive_hash_lo, const std::uint64_t element_count,
    const rund::kernel::GraphControl control) {
  const rund::kernel::NodeKind kind = operation.kind();
  const MapSemantic semantic = kind == rund::kernel::NodeKind::Map
                                   ? BuildMapSemantic(artifact, input)
                                   : MapSemantic{};
  bool binding_indices_ok = true;
  if (kind == rund::kernel::NodeKind::Map) {
    const std::size_t data_binding_count =
        artifact.metadata.binding_accesses.size();
    if (!binding_indices.valid() ||
        data_binding_count > binding_indices.size() ||
        !control.valid(binding_indices.size())) {
      binding_indices_ok = false;
    }
  } else if (!binding_indices.valid() ||
             !control.valid(binding_indices.size())) {
    binding_indices_ok = false;
  }
  if (!binding_indices_ok) {
    artifact.ok = false;
    artifact.reason = "accel_kernel_graph_invalid";
  }
  if (kind == rund::kernel::NodeKind::Map) {
    if (artifact.key.api == rund::kernel::ComputeApi::Cpu) {
      std::string{}.swap(artifact.source_text);
    } else if (!RetainDeviceVsmMapAdmission(artifact, input, semantic)) {
      input = {};
    }
  } else {
    input = {};
  }
  return KernelExecutionStep{
      .operation = std::move(operation),
      .artifact = std::move(artifact),
      .cpu_input = std::move(input),
      .map_semantic = semantic,
      .graph_binding_indices = std::move(binding_indices),
      .graph_binding_indices_ok = binding_indices_ok,
      .primitive_hash_hi = primitive_hash_hi,
      .primitive_hash_lo = primitive_hash_lo,
      .element_count = element_count,
      .control = control,
  };
}

} // namespace rund::node::accel::detail::step
