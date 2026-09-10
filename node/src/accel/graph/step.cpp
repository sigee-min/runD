#include "step.hpp"

#include "step/assembly.hpp"
#include "step/dispatch.hpp"

#include <kernel/program/compute/lowering/emission.hpp>

#include <utility>

namespace rund::node::accel::detail {

KernelExecutionStep BuildMapKernelExecutionStep(
    rund::kernel::LoweringArtifact artifact,
    rund::kernel::compute_lowering_detail::ComputeInputAdmission input,
    KernelBindingIndices binding_indices, const std::uint64_t element_count) {
  return step::AssembleKernelExecutionStep(
      std::move(artifact), std::move(input), std::move(binding_indices),
      Operation{}, 0u, 0u, element_count, {});
}

KernelExecutionStep BuildKernelExecutionStep(GraphCompileNode &&node) {
  if (node.kind() == rund::kernel::NodeKind::Map) {
    auto retained = rund::kernel::compute_lowering_detail::
        EmitAdmittedRetainedComputeArtifact(std::move(node.map_metadata),
                                            std::move(node.cpu_input));
    return step::AssembleKernelExecutionStep(
        std::move(retained.artifact), std::move(retained.input),
        std::move(node.binding_indices), std::move(node.operation),
        node.primitive_hash_hi, node.primitive_hash_lo, node.element_count,
        node.control);
  }
  return step::AssembleKernelExecutionStep(
      std::move(node.artifact), std::move(node.cpu_input),
      std::move(node.binding_indices), std::move(node.operation),
      node.primitive_hash_hi, node.primitive_hash_lo, node.element_count,
      node.control);
}

FrozenDispatchCount
BuildMapDispatchCount(const rund::kernel::ExecutionMetadata &metadata,
                      const std::uint64_t element_count,
                      const rund::kernel::ComputeCaps &caps,
                      const std::uint64_t phase_id) {
  return step::CountMapDispatch(metadata, element_count, caps, phase_id);
}

FrozenDispatchCount
BuildOriginalDispatchCount(const std::span<const GraphCompileNode> nodes,
                           const rund::kernel::ComputeCaps &caps,
                           const std::uint64_t phase_offset) {
  return step::CountOriginalDispatch(nodes, caps, phase_offset);
}

} // namespace rund::node::accel::detail
