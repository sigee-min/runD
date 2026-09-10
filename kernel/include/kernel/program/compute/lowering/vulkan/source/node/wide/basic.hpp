#pragma once

#include "context.hpp"

[[nodiscard]] inline bool
AppendVulkanWideBasicNode(VulkanWideNodeContext &context) {
  std::string &out = context.out;
  const ParsedIR &parsed = context.parsed;
  const ArtifactKey &key = context.key;
  const ParsedNode &node = context.node;
  std::vector<std::string> &node_names = context.node_names;
  const std::string &name = context.name;
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param:
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(
            key, VulkanParamNodeExpr(key, context.layouts[node.aux])));
    return true;
  case IrOp::Read:
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(
            key, VulkanReadNodeExpr(key, context.layouts[node.aux],
                                    parsed.bindings[node.aux])));
    return true;
  case IrOp::ReadUniform:
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(
            key, VulkanReadUniformNodeExpr(key, context.layouts[node.aux])));
    return true;
  case IrOp::ReadAt:
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(
            key, VulkanReadAtNodeExpr(key, context.layouts[node.aux],
                                      context.layouts[node.lhs])));
    return true;
  case IrOp::Constant: {
    const u64 bits =
        static_cast<u64>(node.lhs) | (static_cast<u64>(node.rhs) << 32u);
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(key, VulkanConstantExpr(key.scalar, bits)));
    return true;
  }
  case IrOp::Index:
    AppendVulkanWideValue(out, name,
                          key.scalar == ComputeScalar::Lane64
                              ? "RundWideFrom64(uint64_t(gid))"
                              : "RundWideFrom32(gid)");
    return true;
  case IrOp::Add:
  case IrOp::Sub: {
    const char *fn =
        static_cast<IrOp>(node.op) == IrOp::Add ? "RundWideAdd" : "RundWideSub";
    AppendVulkanWideValue(
        out, name,
        std::string{fn} + "(" + context.Align(node.lhs, node.fixed_format) +
            ", " + context.Align(node.rhs, node.fixed_format) + ")");
    return true;
  }
  case IrOp::Mul:
    AppendVulkanWideValue(out, name,
                          "RundWideMul(" + node_names[node.lhs] + ", " +
                              node_names[node.rhs] + ")");
    return true;
  case IrOp::MulWrap: {
    const std::string lhs = context.Lane(node.lhs);
    const std::string rhs = context.Lane(node.rhs);
    AppendVulkanWideValue(out, name,
                          VulkanWideFromLaneExpr(key, lhs + " * " + rhs));
    return true;
  }
  case IrOp::Neg:
    AppendVulkanWideValue(out, name,
                          "RundWideNeg(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::Abs:
    AppendVulkanWideValue(out, name,
                          "RundWideAbs(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::AbsMagnitude:
    AppendVulkanWideValue(out, name,
                          "RundWideAbs(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::Sign:
    AppendVulkanWideValue(out, name,
                          "RundWideSign(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::PredicateNot:
    AppendVulkanWideValue(out, name,
                          "RundWideBool(!RundWideTruthy(" +
                              node_names[node.lhs] + "))");
    return true;
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr: {
    const char *op =
        static_cast<IrOp>(node.op) == IrOp::PredicateAnd ? " && " : " || ";
    AppendVulkanWideValue(out, name,
                          "RundWideBool(RundWideTruthy(" +
                              node_names[node.lhs] + ")" + op +
                              "RundWideTruthy(" + node_names[node.rhs] + "))");
    return true;
  }
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor: {
    const char *op = static_cast<IrOp>(node.op) == IrOp::BitAnd  ? " & "
                     : static_cast<IrOp>(node.op) == IrOp::BitOr ? " | "
                                                                 : " ^ ";
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(key, context.Lane(node.lhs) + op +
                                        context.Lane(node.rhs)));
    return true;
  }
  case IrOp::BitNot:
    AppendVulkanWideValue(
        out, name, VulkanWideFromLaneExpr(key, "~" + context.Lane(node.lhs)));
    return true;
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string shifted =
        op == IrOp::ShrArithmeticConst
            ? VulkanArithmeticShiftExpr(key.scalar, context.Lane(node.lhs),
                                        node.aux)
            : context.Lane(node.lhs) +
                  (op == IrOp::ShlConst ? " << " : " >> ") +
                  std::to_string(node.aux) + "u";
    AppendVulkanWideValue(out, name, VulkanWideFromLaneExpr(key, shifted));
    return true;
  }
  default:
    return false;
  }
}
