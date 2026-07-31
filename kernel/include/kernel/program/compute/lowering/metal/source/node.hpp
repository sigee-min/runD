#pragma once

[[nodiscard]] inline std::string
MetalWideFromLaneExpr(const ArtifactKey &key, const std::string &value) {
  return key.scalar == ComputeScalar::Lane64 ? "RundWideFrom64(" + value + ")"
                                             : "RundWideFrom32(" + value + ")";
}

[[nodiscard]] inline std::string MetalWideLaneExpr(const ArtifactKey &key,
                                                   const std::string &value) {
  return key.scalar == ComputeScalar::Lane64 ? "as_type<long>(" + value + ".lo)"
                                             : "int(uint(" + value + ".lo))";
}

inline void AppendMetalWideValue(std::string &out, const std::string &name,
                                 const std::string &expr) {
  out += "  const RundWide " + name + " = " + expr + ";\n";
}

[[nodiscard]] inline std::string
MetalWideAlign(const std::string &value, const ComputeFixedFormat source,
               const ComputeFixedFormat target) {
  if (source.fraction_bits == target.fraction_bits) {
    return value;
  }
  return "RundWideShl(" + value + ", " +
         std::to_string(static_cast<u32>(target.fraction_bits) -
                        source.fraction_bits) +
         "u)";
}

inline void AppendMetalWideFixedNode(std::string &out, const ParsedIR &parsed,
                                     const ArtifactKey &key,
                                     const std::vector<BindingLayout> &layouts,
                                     const ParsedNode &node,
                                     const u32 current_node,
                                     std::vector<std::string> &node_names) {
  const std::string &name = SetMetalNodeName(node_names, current_node);
  const auto source_format = [&](const u32 ref) {
    return parsed.nodes[ref - 1u].fixed_format;
  };
  const auto lane = [&](const u32 ref) {
    return MetalWideLaneExpr(key, node_names[ref]);
  };
  const auto wrap_lane = [&](const std::string &expr) {
    return MetalWideFromLaneExpr(key, expr);
  };
  const auto aligned = [&](const u32 ref, const ComputeFixedFormat target) {
    return MetalWideAlign(node_names[ref], source_format(ref), target);
  };
  const u32 width = ComputeScalarBits(key.scalar);
  const std::string width_text = std::to_string(width) + "u";
  const std::string fraction =
      std::to_string(node.fixed_format.fraction_bits) + "u";
  const std::string rounding =
      std::to_string(static_cast<u8>(node.fixed_format.rounding)) + "u";
  const std::string overflow =
      std::to_string(static_cast<u8>(node.fixed_format.overflow)) + "u";
  const auto quantized = [&](const std::string &value,
                             const u32 source_fraction) {
    return "RundWideQuantize(" + value + ", " +
           std::to_string(source_fraction) + "u, " + fraction + ", " +
           rounding + ", " + overflow + ", " + width_text + ")";
  };
  const auto phase_lane = [&](const u32 ref) {
    const std::string value = lane(ref);
    const std::string shift =
        std::to_string(width - source_format(ref).fraction_bits) + "u";
    return key.scalar == ComputeScalar::Lane64
               ? "RundAsSigned64(RundAsUnsigned64(" + value + ") << " + shift +
                     ")"
               : "int(uint(" + value + ") << " + shift + ")";
  };
  const auto canonical_lane = [&](const u32 ref) {
    const std::string canonical =
        "RundWideQuantize(" + node_names[ref] + ", " +
        std::to_string(source_format(ref).fraction_bits) + "u, " +
        std::to_string(width - 1u) + "u, " + rounding + ", 1u, " + width_text +
        ")";
    return MetalWideLaneExpr(key, canonical);
  };
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param:
    AppendMetalWideValue(out, name,
                         wrap_lane(MetalParamNodeExpr(key, layouts[node.aux])));
    return;
  case IrOp::Read:
    AppendMetalWideValue(
        out, name,
        wrap_lane(MetalReadNodeExpr(key, layouts[node.aux],
                                    parsed.bindings[node.aux])));
    return;
  case IrOp::ReadUniform:
    AppendMetalWideValue(
        out, name,
        wrap_lane(MetalReadUniformNodeExpr(key, layouts[node.aux])));
    return;
  case IrOp::ReadAt:
    AppendMetalWideValue(out, name,
                         wrap_lane(MetalReadAtNodeExpr(key, layouts[node.aux],
                                                       layouts[node.lhs])));
    return;
  case IrOp::Constant: {
    const u64 bits =
        static_cast<u64>(node.lhs) | (static_cast<u64>(node.rhs) << 32u);
    AppendMetalWideValue(out, name,
                         wrap_lane(MetalConstantExpr(key.scalar, bits)));
    return;
  }
  case IrOp::Index:
    AppendMetalWideValue(out, name,
                         key.scalar == ComputeScalar::Lane64
                             ? "RundWideFrom64(long(gid))"
                             : "RundWideFrom32(int(gid))");
    return;
  case IrOp::Add:
  case IrOp::Sub: {
    const char *fn =
        static_cast<IrOp>(node.op) == IrOp::Add ? "RundWideAdd" : "RundWideSub";
    AppendMetalWideValue(out, name,
                         std::string{fn} + "(" +
                             aligned(node.lhs, node.fixed_format) + ", " +
                             aligned(node.rhs, node.fixed_format) + ")");
    return;
  }
  case IrOp::Mul:
    AppendMetalWideValue(out, name,
                         "RundWideMul(" + node_names[node.lhs] + ", " +
                             node_names[node.rhs] + ")");
    return;
  case IrOp::MulWrap:
    AppendMetalWideValue(out, name,
                         wrap_lane(MetalWrapBinaryExpr(
                             key.scalar, lane(node.lhs), "*", lane(node.rhs))));
    return;
  case IrOp::Neg:
    AppendMetalWideValue(out, name,
                         "RundWideNeg(" + node_names[node.lhs] + ")");
    return;
  case IrOp::Abs:
    AppendMetalWideValue(out, name,
                         "RundWideAbs(" + node_names[node.lhs] + ")");
    return;
  case IrOp::AbsMagnitude:
    AppendMetalWideValue(out, name,
                         "RundWideAbs(" + node_names[node.lhs] + ")");
    return;
  case IrOp::Sign:
    AppendMetalWideValue(out, name,
                         "RundWideSign(" + node_names[node.lhs] + ")");
    return;
  case IrOp::Quantize:
    AppendMetalWideValue(
        out, name,
        "RundWideQuantize(" + node_names[node.lhs] + ", " +
            std::to_string(source_format(node.lhs).fraction_bits) + "u, " +
            std::to_string(node.fixed_format.fraction_bits) + "u, " +
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
    const std::string lhs = aligned(node.lhs, node.fixed_format);
    const std::string rhs = aligned(node.rhs, node.fixed_format);
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
            std::max<u32>(source_format(node.lhs).integer_bits,
                          source_format(node.rhs).integer_bits)),
        .fraction_bits = static_cast<u8>(
            std::max<u32>(source_format(node.lhs).fraction_bits,
                          source_format(node.rhs).fraction_bits)),
        .rounding = node.fixed_format.rounding,
        .overflow = node.fixed_format.overflow,
        .approximation = node.fixed_format.approximation};
    const std::string lhs = aligned(node.lhs, common);
    const std::string rhs = aligned(node.rhs, common);
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
  case IrOp::PredicateNot:
    AppendMetalWideValue(out, name,
                         "RundWideBool(!RundWideTruthy(" +
                             node_names[node.lhs] + "))");
    return;
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr: {
    const char *op =
        static_cast<IrOp>(node.op) == IrOp::PredicateAnd ? " && " : " || ";
    AppendMetalWideValue(out, name,
                         "RundWideBool(RundWideTruthy(" + node_names[node.lhs] +
                             ")" + op + "RundWideTruthy(" +
                             node_names[node.rhs] + "))");
    return;
  }
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor: {
    const char *op = static_cast<IrOp>(node.op) == IrOp::BitAnd  ? "&"
                     : static_cast<IrOp>(node.op) == IrOp::BitOr ? "|"
                                                                 : "^";
    AppendMetalWideValue(out, name,
                         wrap_lane(MetalBitBinaryExpr(
                             key.scalar, lane(node.lhs), op, lane(node.rhs))));
    return;
  }
  case IrOp::BitNot:
    AppendMetalWideValue(
        out, name, wrap_lane(MetalBitNotExpr(key.scalar, lane(node.lhs))));
    return;
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string shifted =
        op == IrOp::ShrArithmeticConst
            ? MetalArithmeticShiftExpr(key.scalar, lane(node.lhs), node.aux)
            : MetalShiftExpr(key.scalar, op, lane(node.lhs), node.aux);
    AppendMetalWideValue(out, name, wrap_lane(shifted));
    return;
  }
  case IrOp::Clamp: {
    const std::string value = aligned(node.lhs, node.fixed_format);
    const std::string low = aligned(node.rhs, node.fixed_format);
    const std::string high = aligned(node.aux, node.fixed_format);
    AppendMetalWideValue(out, name,
                         "(RundWideSignedLess(" + value + ", " + low + ") ? " +
                             low + " : (RundWideSignedLess(" + high + ", " +
                             value + ") ? " + high + " : " + value + "))");
    return;
  }
  case IrOp::Select:
    AppendMetalWideValue(out, name,
                         "(RundWideTruthy(" + node_names[node.lhs] + ") ? " +
                             aligned(node.rhs, node.fixed_format) + " : " +
                             aligned(node.aux, node.fixed_format) + ")");
    return;
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string lhs = op == IrOp::MulUnsignedFixed
                                ? "RundWideUnsignedLane(" +
                                      node_names[node.lhs] + ", " + width_text +
                                      ")"
                                : node_names[node.lhs];
    const std::string rhs = op == IrOp::MulFixed ? node_names[node.rhs]
                                                 : "RundWideUnsignedLane(" +
                                                       node_names[node.rhs] +
                                                       ", " + width_text + ")";
    const u32 source_fraction =
        static_cast<u32>(source_format(node.lhs).fraction_bits) +
        source_format(node.rhs).fraction_bits;
    const std::string product = "RundWideMul(" + lhs + ", " + rhs + ")";
    if (op == IrOp::MulUnsignedFixed) {
      AppendMetalWideValue(out, name,
                           "RundWideQuantizeUnsignedFixedProduct(" + product +
                               ", " + std::to_string(source_fraction) + "u, " +
                               fraction + ", " + rounding + ", " + overflow +
                               ", " + width_text + ")");
    } else {
      AppendMetalWideValue(out, name, quantized(product, source_fraction));
    }
    return;
  }
  case IrOp::MulAddFixed: {
    const auto product_format = ComputeFixedFormat{
        .integer_bits = static_cast<u8>(source_format(node.lhs).integer_bits +
                                        source_format(node.rhs).integer_bits),
        .fraction_bits = static_cast<u8>(source_format(node.lhs).fraction_bits +
                                         source_format(node.rhs).fraction_bits),
        .rounding = node.fixed_format.rounding,
        .overflow = node.fixed_format.overflow,
        .approximation = node.fixed_format.approximation};
    const std::string product =
        MetalWideAlign("RundWideMul(" + node_names[node.lhs] + ", " +
                           node_names[node.rhs] + ")",
                       product_format, node.fixed_format);
    AppendMetalWideValue(out, name,
                         "RundWideAdd(" + product + ", " +
                             aligned(node.aux, node.fixed_format) + ")");
    return;
  }
  case IrOp::Sin:
  case IrOp::Cos: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string canonical =
        FixedOpExpr(key.scalar, op, phase_lane(node.lhs), {}, {});
    AppendMetalWideValue(out, name,
                         quantized(wrap_lane(canonical), width - 1u));
    return;
  }
  case IrOp::Tan: {
    const std::string phase = phase_lane(node.lhs);
    const std::string sin_value =
        wrap_lane(FixedOpExpr(key.scalar, IrOp::Sin, phase, {}, {}));
    const std::string cos_value =
        wrap_lane(FixedOpExpr(key.scalar, IrOp::Cos, phase, {}, {}));
    AppendMetalWideValue(out, name,
                         "RundWideDivFixed(" + sin_value + ", " + cos_value +
                             ", " + fraction + ", " + rounding + ", " +
                             overflow + ", " + width_text + ")");
    return;
  }
  case IrOp::Exp:
  case IrOp::Log: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string canonical =
        FixedOpExpr(key.scalar, op, canonical_lane(node.lhs), {}, {});
    AppendMetalWideValue(out, name,
                         quantized(wrap_lane(canonical), width - 1u));
    return;
  }
  case IrOp::Atan2: {
    const std::string canonical = FixedOpExpr(
        key.scalar, IrOp::Atan2, lane(node.lhs), lane(node.rhs), {});
    AppendMetalWideValue(out, name, quantized(wrap_lane(canonical), width));
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
  case IrOp::Rsqrt: {
    AppendMetalWideValue(out, name,
                         "RundWideDivFixed(RundWideShl(RundWideOne(), " +
                             fraction + "), RundWideSqrtFixed(" +
                             node_names[node.lhs] + ", " + fraction + ", " +
                             rounding + ", " + overflow + ", " + width_text +
                             "), " + fraction + ", " + rounding + ", " +
                             overflow + ", " + width_text + ")");
    return;
  }
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
      expr = FixedOpExpr(key.scalar, op, lane(node.lhs), {}, {});
    } else {
      expr = FixedOpExpr(key.scalar, op, lane(node.lhs), lane(node.rhs),
                         node.aux == 0u ? std::string{} : lane(node.aux));
    }
    AppendMetalWideValue(out, name, wrap_lane(expr));
    return;
  }
  }
}

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
    AppendMetalAssignedValue(
        out, key.scalar, SetMetalNodeName(node_names, current_node),
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
inline void AppendMetalNodeBody(std::string &out, const ParsedIR &parsed,
                                const ArtifactKey &key,
                                const std::vector<BindingLayout> &layouts) {
  std::vector<std::string> node_names(parsed.nodes.size() + 1u);
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const ParsedNode &node = parsed.nodes[index];
    const u32 current_node = static_cast<u32>(index + 1u);
    AppendMetalNode(out, parsed, key, layouts, node, current_node, node_names);
  }
}
