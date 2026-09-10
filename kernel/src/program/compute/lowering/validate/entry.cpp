#include "local.hpp"

#include <kernel/program/compute/lowering/resource.hpp>

#include <vector>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] const char *ValidateLowerableIR(const ParsedIR &parsed,
                                              const ComputeScalar scalar) {
  const bool fixed_mode = parsed.scalar_mode == ScalarModeFor(scalar);
  if ((fixed_mode && !ComputeFixedFormatValid(scalar, parsed.fixed_format)) ||
      (!fixed_mode && !ComputeFixedFormatAbsent(parsed.fixed_format))) {
    return "compute_ir_numeric_policy_mismatch";
  }
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if ((!BindingWidthShapeValid(binding.kind, binding.element_bytes, scalar) &&
         !ReadAtIndexBinding(parsed, static_cast<u32>(index))) ||
        BindingDomainForShape(binding) == static_cast<ComputeDomain>(0u)) {
      return "compute_ir_binding_scalar_mismatch";
    }
  }
  if (!BindingDomainsMatchGraph(parsed, scalar)) {
    return "compute_ir_binding_scalar_mismatch";
  }
  if (!WidthChangingWriteIsCanonicalMask(parsed, scalar)) {
    return "compute_ir_binding_scalar_mismatch";
  }

  const EffectiveNodeDomains effective_domains =
      ResolveEffectiveNodeDomains(parsed, scalar);
  if (!effective_domains.valid) {
    return "compute_ir_node_invalid";
  }
  if (const char *const reason =
          ValidateWriteModes(parsed, scalar, effective_domains);
      reason != nullptr) {
    return reason;
  }

  std::vector<bool> produces_value(parsed.nodes.size() + 1u, false);
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const ParsedNode &node = parsed.nodes[index];
    const u32 current_node = static_cast<u32>(index + 1u);
    const auto op = static_cast<IrOp>(node.op);
    const auto domain = effective_domains.values[index];
    if (CanonicalIrOpForDomain(op, domain) != op ||
        !IrOpDomainValid(op, domain)) {
      return "compute_ir_node_invalid";
    }
    if (const char *const reason =
            ValidateFixedNodeFormat(parsed, node, scalar);
        reason != nullptr) {
      return reason;
    }
    const ParsedNodeResources resources = ParsedNodeResourcesFor(node);
    if (!resources.ok) {
      return "compute_ir_node_invalid";
    }
    for (u32 operand = 0u; operand < resources.ref_count; ++operand) {
      const u32 ref = resources.refs[operand];
      if (ref == 0u || ref >= current_node || ref >= produces_value.size() ||
          !produces_value[ref]) {
        return "compute_ir_node_invalid";
      }
    }
    if (ConstShiftOp(op) && node.aux >= ScalarBitWidth(scalar)) {
      return "compute_shift_count_invalid";
    }
    produces_value[current_node] = resources.produces_value;
  }
  return nullptr;
}
} // namespace rund::kernel::compute_lowering_detail
