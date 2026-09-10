#include <kernel/program/compute/dsl/expression/context.hpp>

namespace rund::compute_dsl::detail {

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
