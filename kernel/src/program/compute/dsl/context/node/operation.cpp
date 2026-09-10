#include <kernel/program/compute/dsl/expression/context.hpp>

namespace rund::compute_dsl::detail {

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

} // namespace rund::compute_dsl::detail
