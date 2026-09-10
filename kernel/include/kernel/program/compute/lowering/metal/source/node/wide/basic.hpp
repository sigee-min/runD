#pragma once

#include "context.hpp"

[[nodiscard]] inline bool
AppendMetalWideBasicNode(MetalWideNodeContext &context) {
  std::string &out = context.out;
  const ParsedIR &parsed = context.parsed;
  const ArtifactKey &key = context.key;
  const ParsedNode &node = context.node;
  std::vector<std::string> &node_names = context.node_names;
  const std::string &name = context.name;
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param:
    AppendMetalWideValue(
        out, name,
        context.Wrap(MetalParamNodeExpr(key, context.layouts[node.aux])));
    return true;
  case IrOp::Read:
    AppendMetalWideValue(
        out, name,
        context.Wrap(MetalReadNodeExpr(key, context.layouts[node.aux],
                                       parsed.bindings[node.aux])));
    return true;
  case IrOp::ReadUniform:
    AppendMetalWideValue(
        out, name,
        context.Wrap(MetalReadUniformNodeExpr(key, context.layouts[node.aux])));
    return true;
  case IrOp::ReadAt:
    AppendMetalWideValue(
        out, name,
        context.Wrap(MetalReadAtNodeExpr(key, context.layouts[node.aux],
                                         context.layouts[node.lhs])));
    return true;
  case IrOp::Constant: {
    const u64 bits =
        static_cast<u64>(node.lhs) | (static_cast<u64>(node.rhs) << 32u);
    AppendMetalWideValue(out, name,
                         context.Wrap(MetalConstantExpr(key.scalar, bits)));
    return true;
  }
  case IrOp::Index:
    AppendMetalWideValue(out, name,
                         key.scalar == ComputeScalar::Lane64
                             ? "RundWideFrom64(long(gid))"
                             : "RundWideFrom32(int(gid))");
    return true;
  case IrOp::Add:
  case IrOp::Sub: {
    const char *fn =
        static_cast<IrOp>(node.op) == IrOp::Add ? "RundWideAdd" : "RundWideSub";
    AppendMetalWideValue(out, name,
                         std::string{fn} + "(" +
                             context.Align(node.lhs, node.fixed_format) + ", " +
                             context.Align(node.rhs, node.fixed_format) + ")");
    return true;
  }
  case IrOp::Mul:
    AppendMetalWideValue(out, name,
                         "RundWideMul(" + node_names[node.lhs] + ", " +
                             node_names[node.rhs] + ")");
    return true;
  case IrOp::MulWrap:
    AppendMetalWideValue(
        out, name,
        context.Wrap(MetalWrapBinaryExpr(key.scalar, context.Lane(node.lhs),
                                         "*", context.Lane(node.rhs))));
    return true;
  case IrOp::Neg:
    AppendMetalWideValue(out, name,
                         "RundWideNeg(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::Abs:
    AppendMetalWideValue(out, name,
                         "RundWideAbs(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::AbsMagnitude:
    AppendMetalWideValue(out, name,
                         "RundWideAbs(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::Sign:
    AppendMetalWideValue(out, name,
                         "RundWideSign(" + node_names[node.lhs] + ")");
    return true;
  case IrOp::PredicateNot:
    AppendMetalWideValue(out, name,
                         "RundWideBool(!RundWideTruthy(" +
                             node_names[node.lhs] + "))");
    return true;
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr: {
    const char *op =
        static_cast<IrOp>(node.op) == IrOp::PredicateAnd ? " && " : " || ";
    AppendMetalWideValue(out, name,
                         "RundWideBool(RundWideTruthy(" + node_names[node.lhs] +
                             ")" + op + "RundWideTruthy(" +
                             node_names[node.rhs] + "))");
    return true;
  }
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor: {
    const char *op = static_cast<IrOp>(node.op) == IrOp::BitAnd  ? "&"
                     : static_cast<IrOp>(node.op) == IrOp::BitOr ? "|"
                                                                 : "^";
    AppendMetalWideValue(
        out, name,
        context.Wrap(MetalBitBinaryExpr(key.scalar, context.Lane(node.lhs), op,
                                        context.Lane(node.rhs))));
    return true;
  }
  case IrOp::BitNot:
    AppendMetalWideValue(
        out, name,
        context.Wrap(MetalBitNotExpr(key.scalar, context.Lane(node.lhs))));
    return true;
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string shifted =
        op == IrOp::ShrArithmeticConst
            ? MetalArithmeticShiftExpr(key.scalar, context.Lane(node.lhs),
                                       node.aux)
            : MetalShiftExpr(key.scalar, op, context.Lane(node.lhs), node.aux);
    AppendMetalWideValue(out, name, context.Wrap(shifted));
    return true;
  }
  default:
    return false;
  }
}
