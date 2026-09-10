#include "u64_add.hpp"

#include <kernel/program/compute/ir.hpp>

#include <cstddef>

namespace rund::node::accel::detail {
namespace {

using rund::kernel::IrOp;
using rund::kernel::compute_lowering_detail::ParsedIR;
using rund::kernel::compute_lowering_detail::ParsedNode;

[[nodiscard]] constexpr bool operation(const ParsedNode &node,
                                       const IrOp expected) noexcept {
  return node.op == static_cast<std::uint8_t>(expected);
}

[[nodiscard]] constexpr std::uint64_t
constant_value(const ParsedNode &node) noexcept {
  return static_cast<std::uint64_t>(node.lhs) |
         (static_cast<std::uint64_t>(node.rhs) << 32u);
}

} // namespace

bool ClassifyU64AddImmediateChain(const rund::kernel::ArtifactKey &key,
                                  const ParsedIR &ir,
                                  std::uint64_t &immediate) noexcept {
  immediate = 0u;
  if (key.scalar != rund::kernel::ComputeScalar::Lane64 ||
      key.domain != rund::kernel::ComputeDomain::U64 || !ir.ok ||
      ir.bindings.size() != 2u || ir.nodes.size() < 5u ||
      (ir.nodes.size() - 3u) % 2u != 0u) {
    return false;
  }

  const ParsedNode &read = ir.nodes[0u];
  const ParsedNode &index = ir.nodes[1u];
  const ParsedNode &write = ir.nodes.back();
  if (!operation(read, IrOp::Read) || read.lhs != 0u || read.rhs != 0u ||
      read.aux != 0u || !operation(index, IrOp::Index) || index.lhs != 0u ||
      index.rhs != 0u || index.aux != 0u || !operation(write, IrOp::Write) ||
      write.rhs !=
          static_cast<std::uint32_t>(rund::kernel::IrWriteMode::Value) ||
      write.aux != 1u || write.lhs != ir.nodes.size() - 1u) {
    return false;
  }

  std::uint32_t value = write.lhs;
  while (value != 1u) {
    if (value < 4u || value > ir.nodes.size() - 1u) {
      return false;
    }
    const ParsedNode &add = ir.nodes[value - 1u];
    const ParsedNode &constant = ir.nodes[value - 2u];
    const std::uint32_t predecessor = value == 4u ? 1u : value - 2u;
    if (!operation(add, IrOp::Add) || add.lhs != predecessor ||
        add.rhs != value - 1u || add.aux != 0u ||
        !operation(constant, IrOp::Constant) || constant.aux != 0u) {
      return false;
    }
    immediate += constant_value(constant);
    value = predecessor;
  }
  return true;
}

} // namespace rund::node::accel::detail
