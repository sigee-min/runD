#pragma once

#include <kernel/program/compute/lowering/names.hpp>
#include <kernel/program/compute/metadata.hpp>

#include <array>

namespace rund::kernel::compute_lowering_detail {

struct ParsedNodeResources final {
  std::array<u32, 3u> refs{};
  u32 ref_count = 0u;
  bool produces_value = false;
  bool ok = false;
};

// Sole structural classification of canonical node value edges. Binding
// ordinals, immediates, and write modes never appear in refs.
[[nodiscard]] constexpr ParsedNodeResources
ParsedNodeResourcesFor(const ParsedNode &node) noexcept {
  const auto op = static_cast<IrOp>(node.op);
  if (op == IrOp::Write) {
    return ParsedNodeResources{
        .refs = {node.lhs, 0u, 0u}, .ref_count = 1u, .ok = true};
  }
  if (UnaryValueOp(op) || ConstShiftOp(op)) {
    return ParsedNodeResources{.refs = {node.lhs, 0u, 0u},
                               .ref_count = 1u,
                               .produces_value = true,
                               .ok = true};
  }
  if (BinaryValueOp(op)) {
    return ParsedNodeResources{.refs = {node.lhs, node.rhs, 0u},
                               .ref_count = 2u,
                               .produces_value = true,
                               .ok = true};
  }
  if (TernaryValueOp(op)) {
    return ParsedNodeResources{.refs = {node.lhs, node.rhs, node.aux},
                               .ref_count = 3u,
                               .produces_value = true,
                               .ok = true};
  }
  switch (op) {
  case IrOp::Param:
  case IrOp::Read:
  case IrOp::ReadUniform:
  case IrOp::ReadAt:
  case IrOp::Constant:
  case IrOp::Index:
    return ParsedNodeResources{.produces_value = true, .ok = true};
  default:
    return {};
  }
}

// The caller supplies the ParsedIR produced and validated by
// AdmitComputeInput. The analysis still rejects unclassifiable value edges so
// a partial summary cannot become executable metadata.
[[nodiscard]] ComputeResourceSummary
AnalyzeComputeResources(const ParsedIR &parsed, ComputeScalar scalar,
                        ComputeDomain domain) noexcept;

} // namespace rund::kernel::compute_lowering_detail
