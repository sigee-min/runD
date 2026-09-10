#pragma once

#include "wide.hpp"

inline void AppendMetalNode(std::string &out, const ParsedIR &parsed,
                            const ArtifactKey &key,
                            const std::vector<BindingLayout> &layouts,
                            const ParsedNode &node, const u32 current_node,
                            std::vector<std::string> &node_names) {
  if (key.domain == ComputeDomain::Fixed) {
    AppendMetalWideFixedNode(out, parsed, key, layouts, node, current_node,
                             node_names);
    return;
  }
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param: {
    const BindingLayout &layout = layouts[node.aux];
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalParamNodeExpr(key, layout));
    break;
  }
  case IrOp::Read: {
    const BindingLayout &layout = layouts[node.aux];
    const ParsedBinding &binding = parsed.bindings[node.aux];
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalReadNodeExpr(key, layout, binding));
    break;
  }
  case IrOp::ReadUniform: {
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalReadUniformNodeExpr(key, layouts[node.aux]));
    break;
  }
  case IrOp::ReadAt: {
    AppendMetalAssignedValue(
        out, key.scalar, SetMetalNodeName(node_names, current_node),
        MetalReadAtNodeExpr(key, layouts[node.aux], layouts[node.lhs]));
    break;
  }
  case IrOp::Constant: {
    const u64 bits =
        static_cast<u64>(node.lhs) | (static_cast<u64>(node.rhs) << 32u);
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalConstantExpr(key.scalar, bits));
    break;
  }
  case IrOp::Index: {
    std::string expr = MetalType(key.scalar);
    expr += "(gid)";
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node), expr);
    break;
  }
  case IrOp::Add:
  case IrOp::Sub:
  case IrOp::Mul:
  case IrOp::MulWrap: {
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalBasicBinaryNodeExpr(key, node, node_names));
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
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalUnaryNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Quantize: {
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
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
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalFixedBinaryNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::DivSigned:
  case IrOp::DivUnsigned: {
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalIntegerDivNodeExpr(key, node, node_names));
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
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalCompareLogicNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst: {
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalShiftNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Clamp:
  case IrOp::ClampUnsigned: {
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalClampNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Select: {
    AppendMetalAssignedValue(
        out, key.scalar, SetMetalNodeName(node_names, current_node),
        MetalSelectExpr(key.scalar, node_names[node.lhs], node_names[node.rhs],
                        node_names[node.aux]));
    break;
  }
  case IrOp::MulAddFixed: {
    AppendMetalAssignedValue(out, key.scalar,
                             SetMetalNodeName(node_names, current_node),
                             MetalMulAddNodeExpr(key, node, node_names));
    break;
  }
  case IrOp::Write: {
    AppendMetalWriteNode(out, parsed, key, layouts, node, node_names);
    break;
  }
  }
}
