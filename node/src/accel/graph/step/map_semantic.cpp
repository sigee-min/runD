#include "map_semantic.hpp"

#include "../map_semantic/u64_add.hpp"

#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/lowering/model.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::step {
namespace {

using rund::kernel::IrOp;
using rund::kernel::compute_lowering_detail::ParsedIR;
using rund::kernel::compute_lowering_detail::ParsedNode;

[[nodiscard]] bool NodeIs(const ParsedIR &ir, const std::size_t index,
                          const IrOp op, const std::uint32_t lhs,
                          const std::uint32_t rhs,
                          const std::uint32_t aux) noexcept {
  if (index >= ir.nodes.size()) {
    return false;
  }
  const ParsedNode &node = ir.nodes[index];
  return node.op == static_cast<std::uint8_t>(op) && node.lhs == lhs &&
         node.rhs == rhs && node.aux == aux;
}

[[nodiscard]] bool U32Map(const rund::kernel::ArtifactKey &key,
                          const ParsedIR &ir, const std::size_t bindings,
                          const std::size_t nodes) noexcept {
  return key.scalar == rund::kernel::ComputeScalar::Lane32 &&
         key.domain == rund::kernel::ComputeDomain::U32 && ir.ok &&
         ir.bindings.size() == bindings && ir.nodes.size() == nodes;
}

[[nodiscard]] bool RecurrenceNodeTotal(const ParsedNode &node) noexcept {
  const IrOp op = static_cast<IrOp>(node.op);
  switch (op) {
  case IrOp::Write:
    return node.rhs ==
           static_cast<rund::kernel::u32>(rund::kernel::IrWriteMode::Value);
  case IrOp::DivSigned:
  case IrOp::DivUnsigned:
  case IrOp::ReadAt:
    return false;
  case IrOp::Param:
  case IrOp::Read:
  case IrOp::Add:
  case IrOp::Sub:
  case IrOp::Mul:
  case IrOp::MulWrap:
  case IrOp::Min:
  case IrOp::Max:
  case IrOp::Clamp:
  case IrOp::Select:
  case IrOp::Eq:
  case IrOp::Lt:
  case IrOp::Le:
  case IrOp::Constant:
  case IrOp::Neg:
  case IrOp::Abs:
  case IrOp::AbsMagnitude:
  case IrOp::Sign:
  case IrOp::Ne:
  case IrOp::Gt:
  case IrOp::Ge:
  case IrOp::PredicateNot:
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr:
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor:
  case IrOp::BitNot:
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst:
  case IrOp::AddSat:
  case IrOp::AddSatUnsigned:
  case IrOp::SubSat:
  case IrOp::NegPositiveFixed:
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed:
  case IrOp::MulAddFixed:
  case IrOp::DivFixed:
  case IrOp::Recip:
  case IrOp::Sqrt:
  case IrOp::Rsqrt:
  case IrOp::Sin:
  case IrOp::Cos:
  case IrOp::Tan:
  case IrOp::Exp:
  case IrOp::Log:
  case IrOp::Atan2:
  case IrOp::MinUnsigned:
  case IrOp::MaxUnsigned:
  case IrOp::ClampUnsigned:
  case IrOp::LtUnsigned:
  case IrOp::LeUnsigned:
  case IrOp::GtUnsigned:
  case IrOp::GeUnsigned:
  case IrOp::Index:
  case IrOp::Quantize:
  case IrOp::ReadUniform:
    return true;
  }
  return false;
}

[[nodiscard]] bool RecurrenceTotal(const ParsedIR &ir) noexcept {
  if (!ir.ok || ir.nodes.empty()) {
    return false;
  }
  for (const ParsedNode &node : ir.nodes) {
    if (!RecurrenceNodeTotal(node)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] MapSemantic BuildMapSemanticShape(
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input) {
  const ParsedIR &ir = input.parsed;
  if (!artifact.ok || !input.ok || input.key != artifact.key) {
    return {};
  }

  if (U32Map(artifact.key, ir, 3u, 5u) &&
      NodeIs(ir, 0u, IrOp::Read, 0u, 0u, 0u) &&
      NodeIs(ir, 1u, IrOp::Read, 0u, 0u, 1u) &&
      NodeIs(ir, 2u, IrOp::Index, 0u, 0u, 0u) &&
      NodeIs(ir, 3u, IrOp::Add, 1u, 2u, 0u) &&
      NodeIs(ir, 4u, IrOp::Write, 4u, 0u, 2u)) {
    return MapSemantic{.kind = MapSemanticKind::AddWrapU32Pair};
  }

  if (U32Map(artifact.key, ir, 2u, 5u) &&
      NodeIs(ir, 0u, IrOp::Read, 0u, 0u, 0u) &&
      NodeIs(ir, 1u, IrOp::Index, 0u, 0u, 0u) &&
      ir.nodes[2u].op == static_cast<std::uint8_t>(IrOp::Constant) &&
      ir.nodes[2u].rhs == 0u && ir.nodes[2u].aux == 0u &&
      NodeIs(ir, 3u, IrOp::Add, 1u, 3u, 0u) &&
      NodeIs(ir, 4u, IrOp::Write, 4u, 0u, 1u)) {
    return MapSemantic{.kind = MapSemanticKind::AddWrapU32Immediate,
                       .immediate = ir.nodes[2u].lhs};
  }

  if (U32Map(artifact.key, ir, 2u, 5u) &&
      NodeIs(ir, 0u, IrOp::Read, 0u, 0u, 0u) &&
      NodeIs(ir, 1u, IrOp::Index, 0u, 0u, 0u) &&
      ir.nodes[2u].op == static_cast<std::uint8_t>(IrOp::Constant) &&
      ir.nodes[2u].rhs == 0u && ir.nodes[2u].aux == 0u &&
      NodeIs(ir, 3u, IrOp::Mul, 1u, 3u, 0u) &&
      NodeIs(ir, 4u, IrOp::Write, 4u, 0u, 1u)) {
    return MapSemantic{.kind = MapSemanticKind::MulWrapU32Immediate,
                       .immediate = ir.nodes[2u].lhs};
  }

  std::uint64_t u64_immediate = 0u;
  if (ClassifyU64AddImmediateChain(artifact.key, ir, u64_immediate)) {
    return MapSemantic{
        .kind = MapSemanticKind::AddWrapU64Immediate,
        .immediate = static_cast<std::uint32_t>(u64_immediate),
        .maximum = static_cast<std::uint32_t>(u64_immediate >> 32u),
    };
  }

  // Fold's checked tail `tile + count * 0` is an exact U32 identity. Record
  // only the normalized outer+tile expression; the backend never sees or
  // independently rediscovers the eliminated zero term.
  if (U32Map(artifact.key, ir, 4u, 9u) &&
      NodeIs(ir, 0u, IrOp::Read, 0u, 0u, 0u) &&
      NodeIs(ir, 1u, IrOp::Read, 0u, 0u, 1u) &&
      NodeIs(ir, 2u, IrOp::Read, 0u, 0u, 2u) &&
      NodeIs(ir, 3u, IrOp::Index, 0u, 0u, 0u) &&
      NodeIs(ir, 4u, IrOp::Constant, 0u, 0u, 0u) &&
      NodeIs(ir, 5u, IrOp::Mul, 3u, 5u, 0u) &&
      NodeIs(ir, 6u, IrOp::Add, 2u, 6u, 0u) &&
      NodeIs(ir, 7u, IrOp::Add, 1u, 7u, 0u) &&
      NodeIs(ir, 8u, IrOp::Write, 8u, 0u, 3u)) {
    return MapSemantic{.kind = MapSemanticKind::AddWrapU32Pair};
  }

  if (U32Map(artifact.key, ir, 2u, 4u) &&
      NodeIs(ir, 0u, IrOp::ReadUniform, 0u, 0u, 0u) &&
      NodeIs(ir, 1u, IrOp::Index, 0u, 0u, 0u) &&
      NodeIs(ir, 2u, IrOp::Add, 1u, 2u, 0u) &&
      NodeIs(ir, 3u, IrOp::Write, 3u, 0u, 1u)) {
    return MapSemantic{.kind = MapSemanticKind::ResidentWindowItemsU32};
  }

  if (U32Map(artifact.key, ir, 3u, 20u) &&
      NodeIs(ir, 0u, IrOp::Read, 0u, 0u, 0u) &&
      NodeIs(ir, 1u, IrOp::Read, 0u, 0u, 1u) &&
      NodeIs(ir, 2u, IrOp::Index, 0u, 0u, 0u) &&
      ir.nodes[3u].op == static_cast<std::uint8_t>(IrOp::Constant) &&
      ir.nodes[3u].rhs == 0u && ir.nodes[3u].aux == 0u &&
      NodeIs(ir, 4u, IrOp::GtUnsigned, 1u, 4u, 0u) &&
      ir.nodes[5u].op == static_cast<std::uint8_t>(IrOp::Constant) &&
      ir.nodes[5u].rhs == 0u && ir.nodes[5u].aux == 0u &&
      NodeIs(ir, 6u, IrOp::GeUnsigned, 2u, 6u, 0u) &&
      NodeIs(ir, 7u, IrOp::PredicateOr, 5u, 7u, 0u) &&
      ir.nodes[8u].op == static_cast<std::uint8_t>(IrOp::Constant) &&
      ir.nodes[8u].rhs == 0u && ir.nodes[8u].aux == 0u &&
      NodeIs(ir, 9u, IrOp::Constant, 1u, 0u, 0u) &&
      NodeIs(ir, 10u, IrOp::Add, 9u, 10u, 0u) &&
      NodeIs(ir, 11u, IrOp::Mul, 2u, 9u, 0u) &&
      NodeIs(ir, 12u, IrOp::GtUnsigned, 1u, 12u, 0u) &&
      NodeIs(ir, 13u, IrOp::Sub, 1u, 12u, 0u) &&
      NodeIs(ir, 14u, IrOp::Constant, 0u, 0u, 0u) &&
      NodeIs(ir, 15u, IrOp::Select, 13u, 14u, 15u) &&
      NodeIs(ir, 16u, IrOp::GtUnsigned, 16u, 9u, 0u) &&
      NodeIs(ir, 17u, IrOp::Select, 17u, 9u, 16u) &&
      NodeIs(ir, 18u, IrOp::Select, 8u, 11u, 18u) &&
      NodeIs(ir, 19u, IrOp::Write, 19u, 0u, 2u)) {
    return MapSemantic{.kind = MapSemanticKind::ResidentWindowCountU32,
                       .maximum = ir.nodes[3u].lhs,
                       .tile = ir.nodes[8u].lhs,
                       .windows = ir.nodes[5u].lhs};
  }
  return {};
}

} // namespace

MapSemantic BuildMapSemantic(
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input) {
  MapSemantic semantic = BuildMapSemanticShape(artifact, input);
  semantic.recurrence_total = artifact.ok && input.ok &&
                              input.key == artifact.key &&
                              RecurrenceTotal(input.parsed);
  return semantic;
}

} // namespace rund::node::accel::detail::step
