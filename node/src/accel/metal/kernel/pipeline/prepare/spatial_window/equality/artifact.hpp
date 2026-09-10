#pragma once

#include "base.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool MetalSpatialWindowSameParsedIr(
    const rund::kernel::compute_lowering_detail::ParsedIR &left,
    const rund::kernel::compute_lowering_detail::ParsedIR &right) noexcept {
  if (left.name != right.name || left.scalar_mode != right.scalar_mode ||
      left.fixed_format != right.fixed_format ||
      left.bindings.size() != right.bindings.size() ||
      left.nodes.size() != right.nodes.size() || left.ok != right.ok ||
      !SameReason(left.reason, right.reason)) {
    return false;
  }
  for (std::size_t index = 0u; index < left.bindings.size(); ++index) {
    if (!MetalSpatialWindowSameParsedBinding(left.bindings[index],
                                             right.bindings[index])) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.nodes.size(); ++index) {
    if (!MetalSpatialWindowSameParsedNode(left.nodes[index],
                                          right.nodes[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool MetalSpatialWindowSameInputAdmission(
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &left,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission
        &right) noexcept {
  return left.key == right.key &&
         MetalSpatialWindowSameParsedIr(left.parsed, right.parsed) &&
         left.parse_count == right.parse_count && left.ok == right.ok &&
         SameReason(left.reason, right.reason);
}

[[nodiscard]] inline bool MetalSpatialWindowSameArtifact(
    const rund::kernel::LoweringArtifact &left,
    const rund::kernel::LoweringArtifact &right) noexcept {
  return left.key == right.key && left.kind == right.kind &&
         MetalSpatialWindowSameMetadata(left.metadata, right.metadata) &&
         left.source_text == right.source_text &&
         left.source_text_upper_bytes == right.source_text_upper_bytes &&
         left.canonical_ir_bytes == right.canonical_ir_bytes &&
         left.ok == right.ok && SameReason(left.reason, right.reason);
}

#endif

} // namespace rund::node::accel::detail
