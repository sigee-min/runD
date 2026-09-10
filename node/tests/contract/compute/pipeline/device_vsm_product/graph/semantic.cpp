#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "src/accel/graph/map_semantic/u64_add.hpp"

#include <kernel/program/compute/ir.hpp>

#include <cstdint>
#include <limits>

namespace rund_node_test_device_vsm_product::graph_test {
namespace {

using rund::kernel::IrOp;
using rund::kernel::compute_lowering_detail::ParsedIR;
using rund::kernel::compute_lowering_detail::ParsedNode;

[[nodiscard]] constexpr ParsedNode node(const IrOp op, const std::uint32_t lhs,
                                        const std::uint32_t rhs,
                                        const std::uint32_t aux) noexcept {
  return ParsedNode{
      .op = static_cast<std::uint8_t>(op), .lhs = lhs, .rhs = rhs, .aux = aux};
}

[[nodiscard]] ParsedIR additive_chain() {
  ParsedIR parsed{};
  parsed.ok = true;
  parsed.bindings.resize(2u);
  parsed.nodes = {
      node(IrOp::Read, 0u, 0u, 0u),
      node(IrOp::Index, 0u, 0u, 0u),
      node(IrOp::Constant, std::numeric_limits<std::uint32_t>::max(),
           std::numeric_limits<std::uint32_t>::max(), 0u),
      node(IrOp::Add, 1u, 3u, 0u),
      node(IrOp::Constant, 2u, 0u, 0u),
      node(IrOp::Add, 4u, 5u, 0u),
      node(IrOp::Write, 6u,
           static_cast<std::uint32_t>(rund::kernel::IrWriteMode::Value), 1u),
  };
  return parsed;
}

} // namespace

bool CheckFusedGraphSemantic() noexcept {
  rund::kernel::ArtifactKey key{};
  key.scalar = rund::kernel::ComputeScalar::Lane64;
  key.domain = rund::kernel::ComputeDomain::U64;
  ParsedIR accepted = additive_chain();
  std::uint64_t immediate = 0u;
  if (!rund::node::accel::detail::ClassifyU64AddImmediateChain(key, accepted,
                                                               immediate) ||
      immediate != 1u) {
    return false;
  }

  ParsedIR hidden = accepted;
  hidden.nodes.insert(hidden.nodes.end() - 1u,
                      node(IrOp::Constant, 7u, 0u, 0u));
  ParsedIR reordered = accepted;
  reordered.nodes[5u].lhs = 3u;
  ParsedIR wrong_write = accepted;
  wrong_write.nodes.back().rhs =
      static_cast<std::uint32_t>(rund::kernel::IrWriteMode::CheckedOrdinal);
  return !rund::node::accel::detail::ClassifyU64AddImmediateChain(key, hidden,
                                                                  immediate) &&
         !rund::node::accel::detail::ClassifyU64AddImmediateChain(
             key, reordered, immediate) &&
         !rund::node::accel::detail::ClassifyU64AddImmediateChain(
             key, wrong_write, immediate);
}

} // namespace rund_node_test_device_vsm_product::graph_test

#endif
