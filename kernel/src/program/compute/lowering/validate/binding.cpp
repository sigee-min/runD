#include "local.hpp"

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] ComputeDomain
BindingDomainForShape(const ParsedBinding &binding) noexcept {
  const ComputeScalar binding_scalar =
      binding.element_bytes == sizeof(u32)   ? ComputeScalar::Lane32
      : binding.element_bytes == sizeof(u64) ? ComputeScalar::Lane64
                                             : static_cast<ComputeScalar>(0u);
  return binding_scalar == static_cast<ComputeScalar>(0u)
             ? static_cast<ComputeDomain>(0u)
             : DomainForMode(binding_scalar, binding.numeric_mode);
}

[[nodiscard]] bool ReadAtIndexBinding(const ParsedIR &parsed,
                                      const u32 binding) noexcept {
  bool indexed = false;
  for (const ParsedNode &node : parsed.nodes) {
    const auto op = static_cast<IrOp>(node.op);
    if (op == IrOp::ReadAt && node.lhs == binding) {
      indexed = true;
    }
    if (((op == IrOp::Read || op == IrOp::ReadUniform) &&
         node.aux == binding) ||
        (op == IrOp::ReadAt && node.aux == binding)) {
      return false;
    }
  }
  return indexed;
}

namespace {

[[nodiscard]] bool BoundaryMaskWriteBinding(const ParsedIR &parsed,
                                            const u32 binding,
                                            const ComputeScalar) noexcept {
  bool observed = false;
  for (const ParsedNode &node : parsed.nodes) {
    if (static_cast<IrOp>(node.op) != IrOp::Write || node.aux != binding) {
      continue;
    }
    observed = true;
    if (node.rhs != static_cast<u32>(IrWriteMode::BoundaryMask)) {
      return false;
    }
  }
  return observed;
}

} // namespace

[[nodiscard]] bool
BindingDomainsMatchGraph(const ParsedIR &parsed,
                         const ComputeScalar scalar) noexcept {
  const bool fixed_mode = parsed.scalar_mode == ScalarModeFor(scalar);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    const ComputeDomain binding_domain = BindingDomainForShape(binding);
    if (binding_domain == static_cast<ComputeDomain>(0u)) {
      return false;
    }
    if (ReadAtIndexBinding(parsed, static_cast<u32>(index))) {
      continue;
    }
    if (!fixed_mode) {
      if (binding_domain == ComputeDomain::Fixed) {
        if (binding.kind != kWriteBindingKind ||
            !BoundaryMaskWriteBinding(parsed, static_cast<u32>(index),
                                      scalar)) {
          return false;
        }
      }
      continue;
    }
    if (binding_domain == ComputeDomain::Fixed) {
      continue;
    }
    if (binding.kind != kWriteBindingKind || !UnsignedDomain(binding_domain) ||
        !CanonicalUnsignedMaskWriteBinding(parsed, static_cast<u32>(index))) {
      return false;
    }
  }
  return true;
}

} // namespace rund::kernel::compute_lowering_detail
