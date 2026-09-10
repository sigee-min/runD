#pragma once

#include "wide.hpp"

inline void AppendVulkanNode(std::string &out, const ParsedIR &parsed,
                             const ArtifactKey &key,
                             const std::vector<BindingLayout> &layouts,
                             const ParsedNode &node, const u32 current_node,
                             std::vector<std::string> &node_names) {
  if (key.domain == ComputeDomain::Fixed &&
      AppendVulkanWideCoreNode(out, parsed, key, layouts, node, current_node,
                               node_names)) {
    return;
  }
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param: {
    const BindingLayout &layout = layouts[node.aux];
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanParamNodeExpr(key, layout));
    break;
  }
  case IrOp::Read: {
    const BindingLayout &layout = layouts[node.aux];
    const ParsedBinding &binding = parsed.bindings[node.aux];
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanReadNodeExpr(key, layout, binding));
    break;
  }
  case IrOp::ReadUniform: {
    AppendVulkanAssignedValue(
        out, key.scalar, SetVulkanNodeName(node_names, current_node),
        VulkanReadUniformNodeExpr(key, layouts[node.aux]));
    break;
  }
  case IrOp::ReadAt: {
    AppendVulkanAssignedValue(
        out, key.scalar, SetVulkanNodeName(node_names, current_node),
        VulkanReadAtNodeExpr(key, layouts[node.aux], layouts[node.lhs]));
    break;
  }
  case IrOp::Constant: {
    const u64 bits =
        static_cast<u64>(node.lhs) | (static_cast<u64>(node.rhs) << 32u);
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanConstantExpr(key.scalar, bits));
    break;
  }
  case IrOp::Index: {
    std::string expr = VulkanType(key.scalar);
    expr += "(gid)";
    AppendVulkanAssignedValue(
        out, key.scalar, SetVulkanNodeName(node_names, current_node), expr);
    break;
  }
  case IrOp::Add:
  case IrOp::Sub:
  case IrOp::Mul:
  case IrOp::MulWrap: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanBasicBinaryNodeExpr(node, node_names));
    break;
  }
  case IrOp::Neg:
  case IrOp::Abs:
  case IrOp::AbsMagnitude:
  case IrOp::Sign:
  case IrOp::NegPositiveFixed:
  case IrOp::Recip:
  case IrOp::Sqrt:
  case IrOp::Rsqrt:
  case IrOp::Sin:
  case IrOp::Cos:
  case IrOp::Tan:
  case IrOp::Exp:
  case IrOp::Log:
  case IrOp::PredicateNot:
  case IrOp::BitNot: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanUnaryNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Quantize: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              node_names[node.lhs]);
    break;
  }
  case IrOp::AddSat:
  case IrOp::AddSatUnsigned:
  case IrOp::SubSat:
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed:
  case IrOp::DivFixed:
  case IrOp::Atan2: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanFixedBinaryNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::DivSigned:
  case IrOp::DivUnsigned: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanIntegerDivNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Min:
  case IrOp::Max:
  case IrOp::MinUnsigned:
  case IrOp::MaxUnsigned:
  case IrOp::Eq:
  case IrOp::Lt:
  case IrOp::Le:
  case IrOp::Ne:
  case IrOp::Gt:
  case IrOp::Ge:
  case IrOp::LtUnsigned:
  case IrOp::LeUnsigned:
  case IrOp::GtUnsigned:
  case IrOp::GeUnsigned:
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr:
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor: {
    AppendVulkanAssignedValue(
        out, key.scalar, SetVulkanNodeName(node_names, current_node),
        VulkanCompareLogicNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanShiftNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Clamp:
  case IrOp::ClampUnsigned: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanClampNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Select: {
    AppendVulkanAssignedValue(
        out, key.scalar, SetVulkanNodeName(node_names, current_node),
        VulkanSelectExpr(key.scalar, node_names[node.lhs], node_names[node.rhs],
                         node_names[node.aux]));
    break;
  }
  case IrOp::MulAddFixed: {
    AppendVulkanAssignedValue(out, key.scalar,
                              SetVulkanNodeName(node_names, current_node),
                              VulkanMulAddNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Write: {
    AppendVulkanWriteNode(out, parsed, key, layouts, node, node_names);
    break;
  }
  }
}
