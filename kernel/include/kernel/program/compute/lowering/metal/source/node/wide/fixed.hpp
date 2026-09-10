#pragma once

#include "context.hpp"

inline void AppendMetalWideFixedNodeParts(MetalWideNodeContext &context) {
  std::string &out = context.out;
  const ParsedIR &parsed = context.parsed;
  const ArtifactKey &key = context.key;
  const std::vector<BindingLayout> &layouts = context.layouts;
  const ParsedNode &node = context.node;
  std::vector<std::string> &node_names = context.node_names;
  const std::string &name = context.name;
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Quantize:
    AppendMetalWideValue(
        out, name,
        "RundWideQuantize(" + node_names[node.lhs] + ", " +
            std::to_string(context.SourceFormat(node.lhs).fraction_bits) +
            "u, " + std::to_string(node.fixed_format.fraction_bits) + "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
            "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
            "u, " +
            std::to_string(static_cast<u32>(node.fixed_format.integer_bits) +
                           node.fixed_format.fraction_bits) +
            "u)");
    return;
  case IrOp::Min:
  case IrOp::Max: {
    const std::string lhs = context.Align(node.lhs, node.fixed_format);
    const std::string rhs = context.Align(node.rhs, node.fixed_format);
    const bool minimum = static_cast<IrOp>(node.op) == IrOp::Min;
    AppendMetalWideValue(out, name,
                         "(RundWideSignedLess(" + lhs + ", " + rhs + ") ? " +
                             (minimum ? lhs : rhs) + " : " +
                             (minimum ? rhs : lhs) + ")");
    return;
  }
  case IrOp::Eq:
  case IrOp::Ne:
  case IrOp::Lt:
  case IrOp::Le:
  case IrOp::Gt:
  case IrOp::Ge: {
    const auto common = ComputeFixedFormat{
        .integer_bits = static_cast<u8>(
            std::max<u32>(context.SourceFormat(node.lhs).integer_bits,
                          context.SourceFormat(node.rhs).integer_bits)),
        .fraction_bits = static_cast<u8>(
            std::max<u32>(context.SourceFormat(node.lhs).fraction_bits,
                          context.SourceFormat(node.rhs).fraction_bits)),
        .rounding = node.fixed_format.rounding,
        .overflow = node.fixed_format.overflow,
        .approximation = node.fixed_format.approximation};
    const std::string lhs = context.Align(node.lhs, common);
    const std::string rhs = context.Align(node.rhs, common);
    const IrOp op = static_cast<IrOp>(node.op);
    std::string predicate;
    if (op == IrOp::Eq || op == IrOp::Ne) {
      predicate = "RundWideEqual(" + lhs + ", " + rhs + ")";
      if (op == IrOp::Ne)
        predicate = "!(" + predicate + ")";
    } else if (op == IrOp::Lt) {
      predicate = "RundWideSignedLess(" + lhs + ", " + rhs + ")";
    } else if (op == IrOp::Le) {
      predicate = "!RundWideSignedLess(" + rhs + ", " + lhs + ")";
    } else if (op == IrOp::Gt) {
      predicate = "RundWideSignedLess(" + rhs + ", " + lhs + ")";
    } else {
      predicate = "!RundWideSignedLess(" + lhs + ", " + rhs + ")";
    }
    AppendMetalWideValue(out, name, "RundWideBool(" + predicate + ")");
    return;
  }
  case IrOp::Clamp: {
    const std::string value = context.Align(node.lhs, node.fixed_format);
    const std::string low = context.Align(node.rhs, node.fixed_format);
    const std::string high = context.Align(node.aux, node.fixed_format);
    AppendMetalWideValue(out, name,
                         "(RundWideSignedLess(" + value + ", " + low + ") ? " +
                             low + " : (RundWideSignedLess(" + high + ", " +
                             value + ") ? " + high + " : " + value + "))");
    return;
  }
  case IrOp::Select:
    AppendMetalWideValue(out, name,
                         "(RundWideTruthy(" + node_names[node.lhs] + ") ? " +
                             context.Align(node.rhs, node.fixed_format) +
                             " : " +
                             context.Align(node.aux, node.fixed_format) + ")");
    return;
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string lhs = op == IrOp::MulUnsignedFixed
                                ? "RundWideUnsignedLane(" +
                                      node_names[node.lhs] + ", " +
                                      context.width_text + ")"
                                : node_names[node.lhs];
    const std::string rhs =
        op == IrOp::MulFixed ? node_names[node.rhs]
                             : "RundWideUnsignedLane(" + node_names[node.rhs] +
                                   ", " + context.width_text + ")";
    const u32 source_fraction =
        static_cast<u32>(context.SourceFormat(node.lhs).fraction_bits) +
        context.SourceFormat(node.rhs).fraction_bits;
    const std::string product = "RundWideMul(" + lhs + ", " + rhs + ")";
    if (op == IrOp::MulUnsignedFixed) {
      AppendMetalWideValue(out, name,
                           "RundWideQuantizeUnsignedFixedProduct(" + product +
                               ", " + std::to_string(source_fraction) + "u, " +
                               context.fraction + ", " + context.rounding +
                               ", " + context.overflow + ", " +
                               context.width_text + ")");
    } else {
      AppendMetalWideValue(out, name,
                           context.Quantized(product, source_fraction));
    }
    return;
  }
  case IrOp::MulAddFixed: {
    const auto product_format = ComputeFixedFormat{
        .integer_bits =
            static_cast<u8>(context.SourceFormat(node.lhs).integer_bits +
                            context.SourceFormat(node.rhs).integer_bits),
        .fraction_bits =
            static_cast<u8>(context.SourceFormat(node.lhs).fraction_bits +
                            context.SourceFormat(node.rhs).fraction_bits),
        .rounding = node.fixed_format.rounding,
        .overflow = node.fixed_format.overflow,
        .approximation = node.fixed_format.approximation};
    const std::string product =
        MetalWideAlign("RundWideMul(" + node_names[node.lhs] + ", " +
                           node_names[node.rhs] + ")",
                       product_format, node.fixed_format);
    AppendMetalWideValue(out, name,
                         "RundWideAdd(" + product + ", " +
                             context.Align(node.aux, node.fixed_format) + ")");
    return;
  }
  case IrOp::Sin:
  case IrOp::Cos: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string canonical =
        FixedOpExpr(key.scalar, op, context.PhaseLane(node.lhs), {}, {});
    AppendMetalWideValue(
        out, name,
        context.Quantized(context.Wrap(canonical), context.width - 1u));
    return;
  }
  case IrOp::Tan: {
    const std::string phase = context.PhaseLane(node.lhs);
    const std::string sin_value =
        context.Wrap(FixedOpExpr(key.scalar, IrOp::Sin, phase, {}, {}));
    const std::string cos_value =
        context.Wrap(FixedOpExpr(key.scalar, IrOp::Cos, phase, {}, {}));
    AppendMetalWideValue(out, name,
                         "RundWideDivFixed(" + sin_value + ", " + cos_value +
                             ", " + context.fraction + ", " + context.rounding +
                             ", " + context.overflow + ", " +
                             context.width_text + ")");
    return;
  }
  case IrOp::Exp:
  case IrOp::Log: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string canonical =
        FixedOpExpr(key.scalar, op, context.CanonicalLane(node.lhs), {}, {});
    AppendMetalWideValue(
        out, name,
        context.Quantized(context.Wrap(canonical), context.width - 1u));
    return;
  }
  case IrOp::Atan2: {
    const std::string canonical =
        FixedOpExpr(key.scalar, IrOp::Atan2, context.Lane(node.lhs),
                    context.Lane(node.rhs), {});
    AppendMetalWideValue(
        out, name, context.Quantized(context.Wrap(canonical), context.width));
    return;
  }
  case IrOp::DivFixed:
    AppendMetalWideValue(
        out, name,
        "RundWideDivFixed(" + node_names[node.lhs] + ", " +
            node_names[node.rhs] + ", " +
            std::to_string(node.fixed_format.fraction_bits) + "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
            "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
            "u, " + std::to_string(ComputeScalarBits(key.scalar)) + "u)");
    return;
  case IrOp::Recip:
    AppendMetalWideValue(
        out, name,
        "RundWideDivFixed(RundWideShl(RundWideOne(), " +
            std::to_string(node.fixed_format.fraction_bits) + "u), " +
            node_names[node.lhs] + ", " +
            std::to_string(node.fixed_format.fraction_bits) + "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
            "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
            "u, " + std::to_string(ComputeScalarBits(key.scalar)) + "u)");
    return;
  case IrOp::Sqrt:
    AppendMetalWideValue(
        out, name,
        "RundWideSqrtFixed(" + node_names[node.lhs] + ", " +
            std::to_string(node.fixed_format.fraction_bits) + "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
            "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
            "u, " + std::to_string(ComputeScalarBits(key.scalar)) + "u)");
    return;
  case IrOp::Rsqrt:
    AppendMetalWideValue(
        out, name,
        "RundWideDivFixed(RundWideShl(RundWideOne(), " + context.fraction +
            "), RundWideSqrtFixed(" + node_names[node.lhs] + ", " +
            context.fraction + ", " + context.rounding + ", " +
            context.overflow + ", " + context.width_text + "), " +
            context.fraction + ", " + context.rounding + ", " +
            context.overflow + ", " + context.width_text + ")");
    return;
  case IrOp::Write: {
    const BindingLayout &layout = layouts[node.aux];
    const ParsedBinding &binding = parsed.bindings[node.aux];
    const ComputeScalar store_scalar = MetalStoreScalar(binding.element_bytes);
    out += "  ";
    out += MetalStoreFunction(store_scalar);
    out += "(" + layout.symbol + ", " + BindingBaseSymbol(layout) +
           " + gid * " + BindingStrideSymbol(layout) + ", " +
           (store_scalar == ComputeScalar::Lane64
                ? "as_type<long>(" + node_names[node.lhs] + ".lo)"
                : "int(uint(" + node_names[node.lhs] + ".lo))") +
           ");\n";
    return;
  }
  default: {
    const IrOp op = static_cast<IrOp>(node.op);
    std::string expr;
    if (op == IrOp::NegPositiveFixed || op == IrOp::Recip || op == IrOp::Sqrt ||
        op == IrOp::Rsqrt || op == IrOp::Sin || op == IrOp::Cos ||
        op == IrOp::Tan || op == IrOp::Exp || op == IrOp::Log) {
      expr = FixedOpExpr(key.scalar, op, context.Lane(node.lhs), {}, {});
    } else {
      expr = FixedOpExpr(
          key.scalar, op, context.Lane(node.lhs), context.Lane(node.rhs),
          node.aux == 0u ? std::string{} : context.Lane(node.aux));
    }
    AppendMetalWideValue(out, name, context.Wrap(expr));
    return;
  }
  }
}
