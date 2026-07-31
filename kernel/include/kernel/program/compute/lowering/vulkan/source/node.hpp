#pragma once

[[nodiscard]] inline std::string
VulkanWideFromLaneExpr(const ArtifactKey &key, const std::string &value) {
  return key.scalar == ComputeScalar::Lane64 ? "RundWideFrom64(" + value + ")"
                                             : "RundWideFrom32(" + value + ")";
}

inline void AppendVulkanWideValue(std::string &out, const std::string &name,
                                  const std::string &expr) {
  out += "  const RundWide " + name + " = " + expr + ";\n";
}

[[nodiscard]] inline std::string
VulkanWideAlign(const std::string &value, const ComputeFixedFormat source,
                const ComputeFixedFormat target) {
  if (source.fraction_bits == target.fraction_bits)
    return value;
  return "RundWideShl(" + value + ", " +
         std::to_string(static_cast<u32>(target.fraction_bits) -
                        source.fraction_bits) +
         "u)";
}

[[nodiscard]] inline bool AppendVulkanWideCoreNode(
    std::string &out, const ParsedIR &parsed, const ArtifactKey &key,
    const std::vector<BindingLayout> &layouts, const ParsedNode &node,
    const u32 current_node, std::vector<std::string> &node_names) {
  const std::string &name = SetVulkanNodeName(node_names, current_node);
  const auto source_format = [&](const u32 ref) {
    return parsed.nodes[ref - 1u].fixed_format;
  };
  const auto lane = [&](const u32 ref) {
    return key.scalar == ComputeScalar::Lane64
               ? node_names[ref] + ".lo"
               : "uint(" + node_names[ref] + ".lo)";
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
    return lane(ref) + " << " +
           std::to_string(width - source_format(ref).fraction_bits) + "u";
  };
  const auto canonical_lane = [&](const u32 ref) {
    const std::string canonical =
        "RundWideQuantize(" + node_names[ref] + ", " +
        std::to_string(source_format(ref).fraction_bits) + "u, " +
        std::to_string(width - 1u) + "u, " + rounding + ", 1u, " + width_text +
        ")";
    return key.scalar == ComputeScalar::Lane64 ? canonical + ".lo"
                                               : "uint(" + canonical + ".lo)";
  };
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param:
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(key,
                               VulkanParamNodeExpr(key, layouts[node.aux])));
    return true;
  case IrOp::Read:
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(key,
                               VulkanReadNodeExpr(key, layouts[node.aux],
                                                  parsed.bindings[node.aux])));
    return true;
  case IrOp::ReadAt:
    AppendVulkanWideValue(
        out, name,
        VulkanWideFromLaneExpr(key, VulkanReadAtNodeExpr(key, layouts[node.aux],
                                                         layouts[node.lhs])));
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
        std::string{fn} + "(" +
            VulkanWideAlign(node_names[node.lhs], source_format(node.lhs),
                            node.fixed_format) +
            ", " +
            VulkanWideAlign(node_names[node.rhs], source_format(node.rhs),
                            node.fixed_format) +
            ")");
    return true;
  }
  case IrOp::Mul:
    AppendVulkanWideValue(out, name,
                          "RundWideMul(" + node_names[node.lhs] + ", " +
                              node_names[node.rhs] + ")");
    return true;
  case IrOp::MulWrap: {
    const std::string lhs = key.scalar == ComputeScalar::Lane64
                                ? node_names[node.lhs] + ".lo"
                                : "uint(" + node_names[node.lhs] + ".lo)";
    const std::string rhs = key.scalar == ComputeScalar::Lane64
                                ? node_names[node.rhs] + ".lo"
                                : "uint(" + node_names[node.rhs] + ".lo)";
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
  case IrOp::Quantize:
    AppendVulkanWideValue(
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
    return true;
  case IrOp::Min:
  case IrOp::Max: {
    const std::string lhs = VulkanWideAlign(
        node_names[node.lhs], source_format(node.lhs), node.fixed_format);
    const std::string rhs = VulkanWideAlign(
        node_names[node.rhs], source_format(node.rhs), node.fixed_format);
    const bool minimum = static_cast<IrOp>(node.op) == IrOp::Min;
    AppendVulkanWideValue(out, name,
                          "RundWideSelect(RundWideSignedLess(" + lhs + ", " +
                              rhs + "), " + (minimum ? lhs : rhs) + ", " +
                              (minimum ? rhs : lhs) + ")");
    return true;
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
    const std::string lhs =
        VulkanWideAlign(node_names[node.lhs], source_format(node.lhs), common);
    const std::string rhs =
        VulkanWideAlign(node_names[node.rhs], source_format(node.rhs), common);
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
    AppendVulkanWideValue(out, name, "RundWideBool(" + predicate + ")");
    return true;
  }
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
        VulkanWideFromLaneExpr(key, lane(node.lhs) + op + lane(node.rhs)));
    return true;
  }
  case IrOp::BitNot:
    AppendVulkanWideValue(out, name,
                          VulkanWideFromLaneExpr(key, "~" + lane(node.lhs)));
    return true;
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string shifted =
        op == IrOp::ShrArithmeticConst
            ? VulkanArithmeticShiftExpr(key.scalar, lane(node.lhs), node.aux)
            : lane(node.lhs) + (op == IrOp::ShlConst ? " << " : " >> ") +
                  std::to_string(node.aux) + "u";
    AppendVulkanWideValue(out, name, VulkanWideFromLaneExpr(key, shifted));
    return true;
  }
  case IrOp::Clamp: {
    const std::string value = VulkanWideAlign(
        node_names[node.lhs], source_format(node.lhs), node.fixed_format);
    const std::string low = VulkanWideAlign(
        node_names[node.rhs], source_format(node.rhs), node.fixed_format);
    const std::string high = VulkanWideAlign(
        node_names[node.aux], source_format(node.aux), node.fixed_format);
    const std::string lower = "RundWideSelect(RundWideSignedLess(" + value +
                              ", " + low + "), " + low + ", " + value + ")";
    AppendVulkanWideValue(out, name,
                          "RundWideSelect(RundWideSignedLess(" + high + ", " +
                              lower + "), " + high + ", " + lower + ")");
    return true;
  }
  case IrOp::Select:
    AppendVulkanWideValue(
        out, name,
        "RundWideSelect(RundWideTruthy(" + node_names[node.lhs] + "), " +
            VulkanWideAlign(node_names[node.rhs], source_format(node.rhs),
                            node.fixed_format) +
            ", " +
            VulkanWideAlign(node_names[node.aux], source_format(node.aux),
                            node.fixed_format) +
            ")");
    return true;
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
      AppendVulkanWideValue(out, name,
                            "RundWideQuantizeUnsignedFixedProduct(" + product +
                                ", " + std::to_string(source_fraction) + "u, " +
                                fraction + ", " + rounding + ", " + overflow +
                                ", " + width_text + ")");
    } else {
      AppendVulkanWideValue(out, name, quantized(product, source_fraction));
    }
    return true;
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
        VulkanWideAlign("RundWideMul(" + node_names[node.lhs] + ", " +
                            node_names[node.rhs] + ")",
                        product_format, node.fixed_format);
    AppendVulkanWideValue(out, name,
                          "RundWideAdd(" + product + ", " +
                              VulkanWideAlign(node_names[node.aux],
                                              source_format(node.aux),
                                              node.fixed_format) +
                              ")");
    return true;
  }
  case IrOp::Sin:
  case IrOp::Cos: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string canonical =
        FixedOpExpr(key.scalar, op, phase_lane(node.lhs), {}, {});
    AppendVulkanWideValue(
        out, name,
        quantized(VulkanWideFromLaneExpr(key, canonical), width - 1u));
    return true;
  }
  case IrOp::Tan: {
    const std::string phase = phase_lane(node.lhs);
    const std::string sin_value = VulkanWideFromLaneExpr(
        key, FixedOpExpr(key.scalar, IrOp::Sin, phase, {}, {}));
    const std::string cos_value = VulkanWideFromLaneExpr(
        key, FixedOpExpr(key.scalar, IrOp::Cos, phase, {}, {}));
    AppendVulkanWideValue(out, name,
                          "RundWideDivFixed(" + sin_value + ", " + cos_value +
                              ", " + fraction + ", " + rounding + ", " +
                              overflow + ", " + width_text + ")");
    return true;
  }
  case IrOp::Exp:
  case IrOp::Log: {
    const IrOp op = static_cast<IrOp>(node.op);
    const std::string canonical =
        FixedOpExpr(key.scalar, op, canonical_lane(node.lhs), {}, {});
    AppendVulkanWideValue(
        out, name,
        quantized(VulkanWideFromLaneExpr(key, canonical), width - 1u));
    return true;
  }
  case IrOp::Atan2: {
    const std::string canonical = FixedOpExpr(
        key.scalar, IrOp::Atan2, lane(node.lhs), lane(node.rhs), {});
    AppendVulkanWideValue(
        out, name, quantized(VulkanWideFromLaneExpr(key, canonical), width));
    return true;
  }
  case IrOp::DivFixed:
    AppendVulkanWideValue(
        out, name,
        "RundWideDivFixed(" + node_names[node.lhs] + ", " +
            node_names[node.rhs] + ", " +
            std::to_string(node.fixed_format.fraction_bits) + "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
            "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
            "u, " + std::to_string(ComputeScalarBits(key.scalar)) + "u)");
    return true;
  case IrOp::Recip:
    AppendVulkanWideValue(
        out, name,
        "RundWideDivFixed(RundWideShl(RundWideOne(), " +
            std::to_string(node.fixed_format.fraction_bits) + "u), " +
            node_names[node.lhs] + ", " +
            std::to_string(node.fixed_format.fraction_bits) + "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
            "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
            "u, " + std::to_string(ComputeScalarBits(key.scalar)) + "u)");
    return true;
  case IrOp::Sqrt:
    AppendVulkanWideValue(
        out, name,
        "RundWideSqrtFixed(" + node_names[node.lhs] + ", " +
            std::to_string(node.fixed_format.fraction_bits) + "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
            "u, " +
            std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
            "u, " + std::to_string(ComputeScalarBits(key.scalar)) + "u)");
    return true;
  case IrOp::Rsqrt: {
    AppendVulkanWideValue(out, name,
                          "RundWideDivFixed(RundWideShl(RundWideOne(), " +
                              fraction + "), RundWideSqrtFixed(" +
                              node_names[node.lhs] + ", " + fraction + ", " +
                              rounding + ", " + overflow + ", " + width_text +
                              "), " + fraction + ", " + rounding + ", " +
                              overflow + ", " + width_text + ")");
    return true;
  }
  case IrOp::Write: {
    const BindingLayout &layout = layouts[node.aux];
    const ParsedBinding &binding = parsed.bindings[node.aux];
    const ComputeScalar store_scalar = VulkanStoreScalar(binding.element_bytes);
    out += "  " + VulkanStoreFunctionName(store_scalar, layout) + "(" +
           BindingBaseSymbol(layout) + " + gid * " +
           BindingStrideSymbol(layout) + ", " +
           (store_scalar == ComputeScalar::Lane64
                ? node_names[node.lhs] + ".lo"
                : "uint(" + node_names[node.lhs] + ".lo)") +
           ");\n";
    return true;
  }
  default:
    break;
  }
  const IrOp op = static_cast<IrOp>(node.op);
  std::string expr;
  if (op == IrOp::Sign || op == IrOp::NegPositiveFixed || op == IrOp::Recip ||
      op == IrOp::Sqrt || op == IrOp::Rsqrt || op == IrOp::Sin ||
      op == IrOp::Cos || op == IrOp::Tan || op == IrOp::Exp ||
      op == IrOp::Log) {
    expr = FixedOpExpr(key.scalar, op, lane(node.lhs), {}, {});
  } else {
    expr = FixedOpExpr(key.scalar, op, lane(node.lhs), lane(node.rhs),
                       node.aux == 0u ? std::string{} : lane(node.aux));
  }
  AppendVulkanWideValue(out, name, VulkanWideFromLaneExpr(key, expr));
  return true;
}

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
inline void AppendVulkanNodeBody(std::string &out, const ParsedIR &parsed,
                                 const ArtifactKey &key,
                                 const std::vector<BindingLayout> &layouts) {
  std::vector<std::string> node_names(parsed.nodes.size() + 1u);
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const ParsedNode &node = parsed.nodes[index];
    const u32 current_node = static_cast<u32>(index + 1u);
    AppendVulkanNode(out, parsed, key, layouts, node, current_node, node_names);
  }
}
