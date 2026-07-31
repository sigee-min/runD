#include <kernel/program/compute/dsl/expression/context.hpp>

#include <algorithm>

namespace rund::compute_dsl::detail {

rund::kernel::u32
BuildContext::storage_node(const rund::kernel::u32 node) noexcept {
  if (!fixed_mode() || !valid_node(node)) {
    return node;
  }
  const rund::kernel::ComputeScalar scalar =
      WideMode(scalar_mode_) ? rund::kernel::ComputeScalar::Lane64
                             : rund::kernel::ComputeScalar::Lane32;
  if (!rund::kernel::ComputeFixedFormatValid(scalar, source_format(node))) {
    reject("compute_fixed_quantize_required");
  }
  return node;
}

rund::kernel::ComputeFixedFormat BuildContext::value_format() const noexcept {
  return fixed_mode() ? fixed_format_ : rund::kernel::ComputeFixedFormat{};
}

rund::kernel::ComputeFixedFormat
BuildContext::source_format(const rund::kernel::u32 node) const noexcept {
  return fixed_mode() && valid_node(node) ? nodes_[node - 1u].fixed_format
                                          : value_format();
}

rund::kernel::ComputeFixedFormat
BuildContext::unary_format(const rund::kernel::IrOp op,
                           const rund::kernel::u32 source) noexcept {
  auto format = source_format(source);
  if (!fixed_mode()) {
    return format;
  }
  if (op == rund::kernel::IrOp::Neg || op == rund::kernel::IrOp::Abs ||
      op == rund::kernel::IrOp::AbsMagnitude) {
    const unsigned width =
        static_cast<unsigned>(format.integer_bits) + format.fraction_bits;
    if (width >= 128u) {
      reject("compute_fixed_precision_capacity");
      return format;
    }
    ++format.integer_bits;
  }
  if (approximate_unary(op)) {
    format.approximation = rund::kernel::ComputeApproximation::Deterministic;
  }
  return format;
}

rund::kernel::ComputeFixedFormat
BuildContext::binary_format(const rund::kernel::IrOp op,
                            const rund::kernel::u32 lhs,
                            const rund::kernel::u32 rhs) noexcept {
  if (!fixed_mode()) {
    return value_format();
  }
  const auto left = source_format(lhs);
  const auto right = source_format(rhs);
  if (left.rounding != right.rounding || left.overflow != right.overflow) {
    reject("compute_fixed_format_mismatch");
    return left;
  }
  rund::kernel::ComputeFixedFormat out = left;
  out.approximation = static_cast<unsigned>(left.approximation) >=
                              static_cast<unsigned>(right.approximation)
                          ? left.approximation
                          : right.approximation;
  unsigned integer = left.integer_bits;
  unsigned fraction = left.fraction_bits;
  if (op == rund::kernel::IrOp::MulWrap || storage_binary(op)) {
    if (left.integer_bits != right.integer_bits ||
        left.fraction_bits != right.fraction_bits) {
      reject("compute_fixed_format_mismatch");
    }
    if (approximate_binary(op)) {
      out.approximation = rund::kernel::ComputeApproximation::Deterministic;
    }
    return out;
  }
  if (op == rund::kernel::IrOp::Mul) {
    integer += right.integer_bits;
    fraction += right.fraction_bits;
  } else if (op == rund::kernel::IrOp::Add || op == rund::kernel::IrOp::Sub) {
    integer = std::max<unsigned>(left.integer_bits, right.integer_bits) + 1u;
    fraction = std::max<unsigned>(left.fraction_bits, right.fraction_bits);
  } else {
    integer = std::max<unsigned>(left.integer_bits, right.integer_bits);
    fraction = std::max<unsigned>(left.fraction_bits, right.fraction_bits);
  }
  if (integer + fraction > 128u) {
    reject("compute_fixed_precision_capacity");
    return left;
  }
  out.integer_bits = static_cast<rund::kernel::u8>(integer);
  out.fraction_bits = static_cast<rund::kernel::u8>(fraction);
  if (approximate_binary(op)) {
    out.approximation = rund::kernel::ComputeApproximation::Deterministic;
  }
  return out;
}

rund::kernel::ComputeFixedFormat BuildContext::ternary_format(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs, const rund::kernel::u32 aux) noexcept {
  if (!fixed_mode()) {
    return value_format();
  }
  if (op == rund::kernel::IrOp::Select) {
    return binary_format(rund::kernel::IrOp::Min, rhs, aux);
  }
  if (op == rund::kernel::IrOp::MulAddFixed) {
    const auto left = source_format(lhs);
    const auto right = source_format(rhs);
    const auto addend = source_format(aux);
    if (left.rounding != right.rounding || left.overflow != right.overflow ||
        left.rounding != addend.rounding || left.overflow != addend.overflow) {
      reject("compute_fixed_format_mismatch");
      return left;
    }
    const unsigned product_integer = left.integer_bits + right.integer_bits;
    const unsigned product_fraction = left.fraction_bits + right.fraction_bits;
    const unsigned integer = product_integer > addend.integer_bits
                                 ? product_integer
                                 : addend.integer_bits + 1u;
    const unsigned fraction =
        std::max<unsigned>(product_fraction, addend.fraction_bits);
    if (integer + fraction > 128u) {
      reject("compute_fixed_precision_capacity");
      return left;
    }
    auto out = left;
    out.integer_bits = static_cast<rund::kernel::u8>(integer);
    out.fraction_bits = static_cast<rund::kernel::u8>(fraction);
    out.approximation = static_cast<rund::kernel::ComputeApproximation>(
        std::max({static_cast<unsigned>(left.approximation),
                  static_cast<unsigned>(right.approximation),
                  static_cast<unsigned>(addend.approximation)}));
    return out;
  }
  auto out = binary_format(rund::kernel::IrOp::Min, lhs, rhs);
  const auto third = source_format(aux);
  if (out.rounding != third.rounding || out.overflow != third.overflow) {
    reject("compute_fixed_format_mismatch");
    return out;
  }
  const unsigned integer =
      std::max<unsigned>(out.integer_bits, third.integer_bits);
  const unsigned fraction =
      std::max<unsigned>(out.fraction_bits, third.fraction_bits);
  if (integer + fraction > 128u) {
    reject("compute_fixed_precision_capacity");
    return out;
  }
  out.integer_bits = static_cast<rund::kernel::u8>(integer);
  out.fraction_bits = static_cast<rund::kernel::u8>(fraction);
  out.approximation = static_cast<unsigned>(out.approximation) >=
                              static_cast<unsigned>(third.approximation)
                          ? out.approximation
                          : third.approximation;
  return out;
}

} // namespace rund::compute_dsl::detail
