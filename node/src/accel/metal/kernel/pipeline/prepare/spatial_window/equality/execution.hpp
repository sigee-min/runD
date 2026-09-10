#pragma once

#include "range.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool MetalSpatialWindowSameBindingIndices(
    const KernelBindingIndices &left,
    const KernelBindingIndices &right) noexcept {
  if (left.count != right.count || left.ok != right.ok ||
      left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < left.size(); ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool MetalSpatialWindowSameControl(
    const rund::kernel::GraphControl &left,
    const rund::kernel::GraphControl &right) noexcept {
  return left.count_source == right.count_source &&
         left.count_binding == right.count_binding &&
         left.count_byte_offset == right.count_byte_offset &&
         left.capacity == right.capacity &&
         left.predicate_source == right.predicate_source &&
         left.predicate_binding == right.predicate_binding &&
         left.predicate_byte_offset == right.predicate_byte_offset &&
         left.predicate_expected == right.predicate_expected &&
         left.iteration == right.iteration;
}

[[nodiscard]] inline bool
MetalSpatialWindowSameStep(const KernelExecutionStep &left,
                           const KernelExecutionStep &right) noexcept {
  return MetalSpatialWindowSameOperation(left.operation, right.operation) &&
         MetalSpatialWindowSameArtifact(left.artifact, right.artifact) &&
         MetalSpatialWindowSameInputAdmission(left.cpu_input,
                                              right.cpu_input) &&
         left.map_semantic.kind == right.map_semantic.kind &&
         left.map_semantic.immediate == right.map_semantic.immediate &&
         left.map_semantic.maximum == right.map_semantic.maximum &&
         left.map_semantic.tile == right.map_semantic.tile &&
         left.map_semantic.windows == right.map_semantic.windows &&
         left.map_semantic.recurrence_total ==
             right.map_semantic.recurrence_total &&
         MetalSpatialWindowSameBindingIndices(left.graph_binding_indices,
                                              right.graph_binding_indices) &&
         left.graph_binding_indices_ok == right.graph_binding_indices_ok &&
         left.primitive_hash_hi == right.primitive_hash_hi &&
         left.primitive_hash_lo == right.primitive_hash_lo &&
         left.element_count == right.element_count &&
         MetalSpatialWindowSameControl(left.control, right.control) &&
         left.source.begin.index == right.source.begin.index &&
         left.source.end.index == right.source.end.index;
}

[[nodiscard]] inline bool
MetalSpatialWindowSameKernelAdmission(const KernelAdmission &left,
                                      const KernelAdmission &right) noexcept {
  return left.check.ok == right.check.ok &&
         SameReason(left.check.reason, right.check.reason) &&
         left.kernel_id == right.kernel_id &&
         left.context_id == right.context_id &&
         left.graph_id_hi == right.graph_id_hi &&
         left.graph_id_lo == right.graph_id_lo &&
         left.node_count == right.node_count && left.api == right.api &&
         left.scalar == right.scalar && left.domain == right.domain &&
         SameCaps(left.frozen_caps, right.frozen_caps) &&
         SameObject(left.owner, right.owner);
}

[[nodiscard]] inline bool MetalSpatialWindowSameContextAdmission(
    const ContextAdmission &left, const ContextAdmission &right) noexcept {
  return SameCheck(left.check, right.check) &&
         left.context_id == right.context_id && left.api == right.api &&
         SameCaps(left.caps, right.caps) && SameObject(left.owner, right.owner) &&
         SameObject(left.pick, right.pick);
}

[[nodiscard]] inline bool
MetalSpatialWindowSameKernelExecution(const KernelExecution &left,
                                      const KernelExecution &right) noexcept {
  if (!MetalSpatialWindowSameKernelAdmission(left.admission, right.admission) ||
      !MetalSpatialWindowSameContextAdmission(left.context_admission,
                                              right.context_admission) ||
      left.graph_roles.size() != right.graph_roles.size() ||
      left.graph_shapes.size() != right.graph_shapes.size() ||
      left.graph_visibilities.size() != right.graph_visibilities.size() ||
      left.graph_alias_representatives.size() !=
          right.graph_alias_representatives.size() ||
      left.resets.size() != right.resets.size() ||
      left.steps.size() != right.steps.size() ||
      left.required_barriers.size() != right.required_barriers.size() ||
      left.removed_dispatch_count != right.removed_dispatch_count ||
      left.original_operation_count != right.original_operation_count ||
      left.fused_operation_count != right.fused_operation_count ||
      left.fusion_rejection_count != right.fusion_rejection_count ||
      !SameReason(left.fusion_reason, right.fusion_reason)) {
    return false;
  }
  for (std::size_t index = 0u; index < left.graph_roles.size(); ++index) {
    if (left.graph_roles[index] != right.graph_roles[index]) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.graph_shapes.size(); ++index) {
    const rund::AccelBufferDesc &a = left.graph_shapes[index];
    const rund::AccelBufferDesc &b = right.graph_shapes[index];
    if (a.scalar_width_bytes != b.scalar_width_bytes || a.count != b.count ||
        a.usage != b.usage) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.graph_visibilities.size();
       ++index) {
    if (left.graph_visibilities[index] != right.graph_visibilities[index]) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.graph_alias_representatives.size();
       ++index) {
    if (left.graph_alias_representatives[index] !=
        right.graph_alias_representatives[index]) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.resets.size(); ++index) {
    const ResetPlan &a = left.resets[index];
    const ResetPlan &b = right.resets[index];
    if (a.binding != b.binding || a.step.index != b.step.index ||
        a.last.index != b.last.index) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.required_barriers.size(); ++index) {
    if (left.required_barriers[index] != right.required_barriers[index]) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.steps.size(); ++index) {
    if (!MetalSpatialWindowSameStep(left.steps[index], right.steps[index])) {
      return false;
    }
  }
  return true;
}

#endif

} // namespace rund::node::accel::detail
