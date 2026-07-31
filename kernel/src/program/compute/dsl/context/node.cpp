#include <kernel/program/compute/dsl/expression/context.hpp>

namespace rund::compute_dsl::detail {

rund::kernel::u32
BuildContext::param_node(const std::string_view name) noexcept {
  const rund::kernel::u32 binding = binding_index(name, BindingKind::Param);
  if (!valid_binding(binding, BindingKind::Param)) {
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::Param, 0u, 0u, binding, value_format(),
                     binding_domain(binding));
}

rund::kernel::u32
BuildContext::read_node(const rund::kernel::u32 binding) noexcept {
  if (!valid_binding(binding, BindingKind::Read)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::Read, 0u, 0u, binding, value_format(),
                     binding_domain(binding));
}

rund::kernel::u32 BuildContext::read_node(
    const rund::kernel::u32 binding,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_binding(binding, BindingKind::Read)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  if (fixed_mode() &&
      !rund::kernel::ComputeFixedFormatValid(
          WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                                 : rund::kernel::ComputeScalar::Lane32,
          format)) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::Read, 0u, 0u, binding, format,
                     binding_domain(binding));
}

rund::kernel::u32 BuildContext::read_uniform_node(
    const rund::kernel::u32 binding,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_binding(binding, BindingKind::Read)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (!binding_value_mode_valid(binding)) {
    reject("compute_value_invalid");
    return 0u;
  }
  if (fixed_mode() &&
      !rund::kernel::ComputeFixedFormatValid(
          WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                                 : rund::kernel::ComputeScalar::Lane32,
          format)) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::ReadUniform, 0u, 0u, binding, format,
                     binding_domain(binding));
}

rund::kernel::u32 BuildContext::read_at_node(
    const rund::kernel::u32 binding, const rund::kernel::u32 index,
    const rund::kernel::u32 count,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_binding(binding, BindingKind::Read) ||
      !valid_binding(index, BindingKind::Read) || binding == index ||
      count == 0u || !binding_value_mode_valid(binding) ||
      bindings_ == nullptr ||
      (*bindings_)[index].numeric_mode != ScalarMode::U32 ||
      (*bindings_)[index].element_bytes != sizeof(rund::kernel::u32)) {
    reject("compute_binding_invalid");
    return 0u;
  }
  if (fixed_mode() &&
      !rund::kernel::ComputeFixedFormatValid(
          WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                                 : rund::kernel::ComputeScalar::Lane32,
          format)) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::ReadAt, index, count, binding, format,
                     binding_domain(binding));
}

rund::kernel::u32
BuildContext::binary_node(const rund::kernel::IrOp op,
                          const rund::kernel::u32 lhs,
                          const rund::kernel::u32 rhs) noexcept {
  if (!valid_node(lhs) || !valid_node(rhs)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const auto value_domain = merged_node_domain(lhs, rhs);
  const auto canonical_op =
      rund::kernel::CanonicalIrOpForDomain(op, value_domain);
  if (!rund::kernel::IrOpDomainValid(canonical_op, value_domain)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const rund::kernel::u32 stored_lhs =
      storage_binary(canonical_op) ? storage_node(lhs) : lhs;
  const rund::kernel::u32 stored_rhs =
      storage_binary(canonical_op) ? storage_node(rhs) : rhs;
  return append_node(canonical_op, stored_lhs, stored_rhs, 0u,
                     binary_format(canonical_op, stored_lhs, stored_rhs),
                     value_domain);
}

rund::kernel::u32 BuildContext::binary_node(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_node(lhs) || !valid_node(rhs)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const auto value_domain = merged_node_domain(lhs, rhs);
  const auto canonical_op =
      rund::kernel::CanonicalIrOpForDomain(op, value_domain);
  if (!rund::kernel::IrOpDomainValid(canonical_op, value_domain)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const rund::kernel::u32 stored_lhs =
      storage_binary(canonical_op) ? storage_node(lhs) : lhs;
  const rund::kernel::u32 stored_rhs =
      storage_binary(canonical_op) ? storage_node(rhs) : rhs;
  return append_node(canonical_op, stored_lhs, stored_rhs, 0u, format,
                     value_domain);
}

rund::kernel::u32
BuildContext::unary_node(const rund::kernel::IrOp op,
                         const rund::kernel::u32 lhs) noexcept {
  if (!valid_node(lhs)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const auto value_domain = unary_node_domain(lhs);
  if (!rund::kernel::IrOpDomainValid(op, value_domain)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const rund::kernel::u32 stored = storage_unary(op) ? storage_node(lhs) : lhs;
  return append_node(op, stored, 0u, 0u, unary_format(op, stored),
                     value_domain);
}

rund::kernel::u32
BuildContext::storage_quantize_node(const rund::kernel::u32 lhs) noexcept {
  if (!fixed_mode() || !valid_node(lhs)) {
    reject("compute_value_invalid");
    return 0u;
  }
  auto format = value_format();
  if (source_format(lhs).approximation ==
      rund::kernel::ComputeApproximation::Deterministic) {
    format.approximation = rund::kernel::ComputeApproximation::Deterministic;
  }
  return unary_node(rund::kernel::IrOp::Quantize, lhs, format);
}

rund::kernel::u32 BuildContext::unary_node(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_node(lhs)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const auto value_domain = unary_node_domain(lhs);
  if (!rund::kernel::IrOpDomainValid(op, value_domain)) {
    reject("compute_value_invalid");
    return 0u;
  }
  if (op == rund::kernel::IrOp::Quantize) {
    const rund::kernel::ComputeScalar scalar =
        WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                               : rund::kernel::ComputeScalar::Lane32;
    if (!fixed_mode() ||
        !rund::kernel::ComputeFixedFormatValid(scalar, format)) {
      reject("compute_fixed_format_invalid");
      return 0u;
    }
    if (source_format(lhs).approximation ==
            rund::kernel::ComputeApproximation::Deterministic &&
        format.approximation !=
            rund::kernel::ComputeApproximation::Deterministic) {
      reject("compute_fixed_approximation_downgrade");
      return 0u;
    }
  }
  const rund::kernel::u32 stored = storage_unary(op) ? storage_node(lhs) : lhs;
  return append_node(op, stored, 0u, 0u, format, value_domain);
}

rund::kernel::u32
BuildContext::const_shift_node(const rund::kernel::IrOp op,
                               const rund::kernel::u32 lhs,
                               const rund::kernel::u32 amount) noexcept {
  if (!valid_node(lhs)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const auto value_domain = unary_node_domain(lhs);
  if (!rund::kernel::IrOpDomainValid(op, value_domain)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const rund::kernel::u32 width = WideMode(scalar_mode_) ? 64u : 32u;
  if (amount >= width) {
    reject("compute_shift_count_invalid");
    return 0u;
  }
  const rund::kernel::u32 stored = fixed_mode() ? storage_node(lhs) : lhs;
  return append_node(op, stored, 0u, amount, source_format(stored),
                     value_domain);
}

rund::kernel::u32 BuildContext::ternary_node(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs, const rund::kernel::u32 aux) noexcept {
  if (!valid_node(lhs) || !valid_node(rhs) || !valid_node(aux)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const auto value_domain = ternary_node_domain(op, lhs, rhs, aux);
  const auto canonical_op =
      rund::kernel::CanonicalIrOpForDomain(op, value_domain);
  if (!rund::kernel::IrOpDomainValid(canonical_op, value_domain)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(canonical_op, lhs, rhs, aux,
                     ternary_format(canonical_op, lhs, rhs, aux), value_domain);
}

rund::kernel::u32 BuildContext::ternary_node(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs, const rund::kernel::u32 aux,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_node(lhs) || !valid_node(rhs) || !valid_node(aux)) {
    reject("compute_value_invalid");
    return 0u;
  }
  const auto value_domain = ternary_node_domain(op, lhs, rhs, aux);
  const auto canonical_op =
      rund::kernel::CanonicalIrOpForDomain(op, value_domain);
  if (!rund::kernel::IrOpDomainValid(canonical_op, value_domain)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(canonical_op, lhs, rhs, aux, format, value_domain);
}

rund::kernel::u32
BuildContext::constant_node(const rund::kernel::u64 bits) noexcept {
  return append_node(
      rund::kernel::IrOp::Constant,
      static_cast<rund::kernel::u32>(bits & 0xffffffffull),
      static_cast<rund::kernel::u32>((bits >> 32u) & 0xffffffffull), 0u,
      value_format(), domain());
}

rund::kernel::u32
BuildContext::constant_node(const rund::kernel::u64 bits,
                            const rund::kernel::u32 anchor) noexcept {
  if (!valid_node(anchor)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(
      rund::kernel::IrOp::Constant,
      static_cast<rund::kernel::u32>(bits & 0xffffffffull),
      static_cast<rund::kernel::u32>((bits >> 32u) & 0xffffffffull), 0u,
      source_format(anchor), unary_node_domain(anchor));
}

rund::kernel::u32
BuildContext::typed_constant_node(const rund::kernel::u64 bits,
                                  const ScalarMode mode) noexcept {
  const auto value_domain = ToComputeDomain(mode);
  const bool same_width = WideMode(mode) == WideMode(scalar_mode_);
  if (value_domain == static_cast<rund::kernel::ComputeDomain>(0u) ||
      value_domain == rund::kernel::ComputeDomain::Fixed || !same_width ||
      fixed_mode()) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(
      rund::kernel::IrOp::Constant,
      static_cast<rund::kernel::u32>(bits & 0xffffffffull),
      static_cast<rund::kernel::u32>((bits >> 32u) & 0xffffffffull), 0u, {},
      value_domain);
}

rund::kernel::u32
BuildContext::storage_constant_node(const rund::kernel::u64 bits,
                                    const rund::kernel::u32 anchor) noexcept {
  if (!valid_node(anchor)) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(
      rund::kernel::IrOp::Constant,
      static_cast<rund::kernel::u32>(bits & 0xffffffffull),
      static_cast<rund::kernel::u32>((bits >> 32u) & 0xffffffffull), 0u,
      value_format(), unary_node_domain(anchor));
}

rund::kernel::u32 BuildContext::constant_node(
    const rund::kernel::u64 bits, const rund::kernel::u32 anchor,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  if (!valid_node(anchor)) {
    reject("compute_value_invalid");
    return 0u;
  }
  if (!fixed_mode() || rund::kernel::ComputeFixedFormatAbsent(format) ||
      !rund::kernel::ComputeIntermediateFormatValid(format)) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  return append_node(
      rund::kernel::IrOp::Constant,
      static_cast<rund::kernel::u32>(bits & 0xffffffffull),
      static_cast<rund::kernel::u32>((bits >> 32u) & 0xffffffffull), 0u, format,
      unary_node_domain(anchor));
}

rund::kernel::u32 BuildContext::index_node() noexcept {
  return append_node(rund::kernel::IrOp::Index, 0u, 0u, 0u, value_format(),
                     domain());
}

rund::kernel::u32 BuildContext::index_node(const ScalarMode mode) noexcept {
  const auto value_domain = ToComputeDomain(mode);
  const bool same_width = WideMode(mode) == WideMode(scalar_mode_);
  if (value_domain == static_cast<rund::kernel::ComputeDomain>(0u) ||
      value_domain == rund::kernel::ComputeDomain::Fixed || !same_width ||
      fixed_mode()) {
    reject("compute_value_invalid");
    return 0u;
  }
  return append_node(rund::kernel::IrOp::Index, 0u, 0u, 0u, {}, value_domain);
}

rund::kernel::u32 BuildContext::append_node(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs, const rund::kernel::u32 aux,
    const rund::kernel::ComputeFixedFormat format,
    const rund::kernel::ComputeDomain value_domain) noexcept {
  const bool fixed_boundary_write =
      op == rund::kernel::IrOp::Write &&
      rhs == static_cast<rund::kernel::u32>(
                 rund::kernel::IrWriteMode::BoundaryMask) &&
      value_domain == rund::kernel::ComputeDomain::Fixed &&
      rund::kernel::ComputeFixedFormatValid(
          WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                                 : rund::kernel::ComputeScalar::Lane32,
          format);
  if (!fixed_mode() && !rund::kernel::ComputeFixedFormatAbsent(format) &&
      !fixed_boundary_write) {
    reject("compute_fixed_format_invalid");
    return 0u;
  }
  if (op != rund::kernel::IrOp::Write) {
    const rund::kernel::u32 existing =
        find_node(op, lhs, rhs, aux, format, value_domain);
    if (existing != 0u) {
      return existing;
    }
  }
  if (nodes_.size() >= 0xfffffff0u) {
    reject("compute_ir_too_large");
    return 0u;
  }
  nodes_.push_back(rund::kernel::ComputeIrNode{
      .op = op,
      .lhs = lhs,
      .rhs = rhs,
      .aux = aux,
      .fixed_format = format,
  });
  node_domains_.push_back(value_domain);
  return static_cast<rund::kernel::u32>(nodes_.size());
}

rund::kernel::u32 BuildContext::find_node(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs, const rund::kernel::u32 aux,
    const rund::kernel::ComputeFixedFormat format,
    const rund::kernel::ComputeDomain value_domain) const noexcept {
  for (std::size_t index = 0u; index < nodes_.size(); ++index) {
    const rund::kernel::ComputeIrNode &node = nodes_[index];
    if (node.op == op && node.lhs == lhs && node.rhs == rhs &&
        node.aux == aux && node.fixed_format == format &&
        index < node_domains_.size() && node_domains_[index] == value_domain) {
      return static_cast<rund::kernel::u32>(index + 1u);
    }
  }
  return 0u;
}

} // namespace rund::compute_dsl::detail
