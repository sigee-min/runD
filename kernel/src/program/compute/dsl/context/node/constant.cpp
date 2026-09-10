#include <kernel/program/compute/dsl/expression/context.hpp>

namespace rund::compute_dsl::detail {

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

} // namespace rund::compute_dsl::detail
