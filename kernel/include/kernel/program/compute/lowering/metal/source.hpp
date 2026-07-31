#pragma once

#include <kernel/program/compute/lowering/fixed/ops.hpp>
#include <kernel/program/compute/lowering/format.hpp>
#include <kernel/program/compute/lowering/metal/expr.hpp>
#include <kernel/program/compute/lowering/metal/io.hpp>
#include <kernel/program/compute/lowering/metal/wide.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

inline void AppendMetalIntegerDivHelpers(std::string &out,
                                         const ParsedIR &parsed,
                                         const ArtifactKey &key) {
  const bool signed_divide = ParsedIrHasOp(parsed, IrOp::DivSigned);
  const bool unsigned_divide = ParsedIrHasOp(parsed, IrOp::DivUnsigned);
  if (key.scalar == ComputeScalar::Lane32) {
    if (signed_divide) {
      out += "inline int RundDivSigned32(int lhs, int rhs) { return lhs / rhs; "
             "}\n";
    }
    if (unsigned_divide) {
      out += "inline int RundDivUnsigned32(int lhs, int rhs) { return "
             "int(uint(lhs) / uint(rhs)); }\n";
    }
    return;
  }
  if (signed_divide) {
    out += "inline long RundDivSigned64(long lhs, long rhs) { return lhs / "
           "rhs; }\n";
  }
  if (unsigned_divide) {
    out += "inline long RundDivUnsigned64(long lhs, long rhs) { return "
           "long(ulong(lhs) / ulong(rhs)); }\n";
  }
}

inline void
AppendMetalSourcePreamble(std::string &out, const ParsedIR &parsed,
                          const ArtifactKey &key,
                          const std::vector<BindingLayout> &layouts) {
  out += "#include <metal_stdlib>\n";
  out += "using namespace metal;\n";
  out += "// rund.compute.metal.source\n";
  out += "// operation_hex=";
  out += HexText(parsed.name);
  out += "\n";
  AppendKey(out, key, "// ");
  AppendBindingLayout(out, parsed, layouts, "// ");
  AppendNodeLayout(out, parsed, "// ");
  AppendMetalHelpers(out, parsed, key);
  if (key.domain == ComputeDomain::Fixed) {
    AppendMetalWideFixedHelpers(out);
  }
  AppendMetalIntegerDivHelpers(out, parsed, key);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind != 2u && binding.kind != 3u) {
      continue;
    }
    out += "constant uint ";
    out += BindingBaseSymbol(layouts[index]);
    out += " = 0u;\n";
    out += "constant uint ";
    out += BindingStrideSymbol(layouts[index]);
    out += " = ";
    out += std::to_string(binding.element_bytes);
    out += "u;\n";
  }
  out += "kernel void rund_compute_map_";
  AppendHex64Digits(out, key.op_hash_hi);
  out += "_";
  AppendHex64Digits(out, key.op_hash_lo);
  out += "(\n";
  out += "    constant uchar* rund_params [[buffer(0)]],\n";
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    const BindingLayout &layout = layouts[index];
    if (binding.kind == 2u) {
      out += "    const device uchar* ";
      out += layout.symbol;
      out += " [[buffer(";
      out += std::to_string(layout.buffer);
      out += ")]],\n";
    }
  }
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    const BindingLayout &layout = layouts[index];
    if (binding.kind == 3u) {
      out += "    device uchar* ";
      out += layout.symbol;
      out += " [[buffer(";
      out += std::to_string(layout.buffer);
      out += ")]],\n";
    }
  }
  out += "    uint gid [[thread_position_in_grid]]) {\n";
}

inline std::string &SetMetalNodeName(std::vector<std::string> &node_names,
                                     const u32 current_node) {
  node_names[current_node] = "node_" + std::to_string(current_node);
  return node_names[current_node];
}

inline void AppendMetalAssignedValue(std::string &out,
                                     const ComputeScalar scalar,
                                     const std::string &name,
                                     const std::string &expr) {
  out += "  const ";
  out += MetalType(scalar);
  out += " ";
  out += name;
  out += " = ";
  out += expr;
  out += ";\n";
}

[[nodiscard]] inline std::string
MetalParamNodeExpr(const ArtifactKey &key, const BindingLayout &layout) {
  std::string expr = MetalParamLoadFunction(key.scalar);
  expr += "(rund_params, ";
  expr += std::to_string(layout.param_offset);
  expr += "u)";
  return expr;
}

[[nodiscard]] inline std::string MetalReadNodeExpr(const ArtifactKey &key,
                                                   const BindingLayout &layout,
                                                   const ParsedBinding &) {
  std::string expr = MetalLoadFunction(key.scalar);
  expr += "(";
  expr += layout.symbol;
  expr += ", ";
  expr += BindingBaseSymbol(layout);
  expr += " + gid * ";
  expr += BindingStrideSymbol(layout);
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
MetalReadAtNodeExpr(const ArtifactKey &key, const BindingLayout &source,
                    const BindingLayout &index) {
  std::string offset = MetalLoadFunction(ComputeScalar::Lane32);
  offset += "(";
  offset += index.symbol;
  offset += ", ";
  offset += BindingBaseSymbol(index);
  offset += " + gid * ";
  offset += BindingStrideSymbol(index);
  offset += ")";
  std::string expr = MetalLoadFunction(key.scalar);
  expr += "(";
  expr += source.symbol;
  expr += ", ";
  expr += BindingBaseSymbol(source);
  expr += " + ulong(";
  expr += offset;
  expr += ") * ";
  expr += BindingStrideSymbol(source);
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
MetalBasicBinaryNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                         const std::vector<std::string> &node_names) {
  const char *op_text = node.op == static_cast<u8>(IrOp::Add)   ? "+"
                        : node.op == static_cast<u8>(IrOp::Sub) ? "-"
                                                                : "*";
  return MetalWrapBinaryExpr(key.scalar, node_names[node.lhs], op_text,
                             node_names[node.rhs]);
}

[[nodiscard]] inline std::string
MetalUnaryNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                   const std::vector<std::string> &node_names) {
  if (node.op == static_cast<u8>(IrOp::Neg)) {
    return MetalWrapNegExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::Abs)) {
    return MetalAbsExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::AbsMagnitude)) {
    return MetalAbsMagnitudeExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::Sign)) {
    return MetalSignExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::BitNot)) {
    return MetalBitNotExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::PredicateNot)) {
    return MetalPredicateNotExpr(key.scalar, node_names[node.lhs]);
  }
  return FixedOpExpr(key.scalar, static_cast<IrOp>(node.op),
                     node_names[node.lhs], {}, {});
}

[[nodiscard]] inline std::string
MetalFixedBinaryNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                         const std::vector<std::string> &node_names) {
  return FixedOpExpr(key.scalar, static_cast<IrOp>(node.op),
                     node_names[node.lhs], node_names[node.rhs], {});
}

[[nodiscard]] inline std::string
MetalIntegerDivNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                        const std::vector<std::string> &node_names) {
  const bool wide = key.scalar == ComputeScalar::Lane64;
  const bool signed_divide = node.op == static_cast<u8>(IrOp::DivSigned);
  const char *const name =
      signed_divide ? (wide ? "RundDivSigned64" : "RundDivSigned32")
                    : (wide ? "RundDivUnsigned64" : "RundDivUnsigned32");
  return std::string{name} + "(" + node_names[node.lhs] + ", " +
         node_names[node.rhs] + ")";
}

[[nodiscard]] inline std::string
MetalCompareLogicNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                          const std::vector<std::string> &node_names) {
  const IrOp op = static_cast<IrOp>(node.op);
  const bool force_unsigned =
      op == IrOp::MinUnsigned || op == IrOp::MaxUnsigned ||
      op == IrOp::LtUnsigned || op == IrOp::LeUnsigned ||
      op == IrOp::GtUnsigned || op == IrOp::GeUnsigned;
  const ComputeDomain domain =
      force_unsigned
          ? (key.scalar == ComputeScalar::Lane64 ? ComputeDomain::U64
                                                 : ComputeDomain::U32)
      : key.domain == ComputeDomain::Fixed  ? ComputeDomain::Fixed
      : key.scalar == ComputeScalar::Lane64 ? ComputeDomain::I64
                                            : ComputeDomain::I32;
  if (op == IrOp::Min || op == IrOp::MinUnsigned) {
    return MetalMinExpr(domain, key.scalar, node_names[node.lhs],
                        node_names[node.rhs]);
  }
  if (op == IrOp::Max || op == IrOp::MaxUnsigned) {
    return MetalMaxExpr(domain, key.scalar, node_names[node.lhs],
                        node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::PredicateAnd)) {
    return MetalPredicateLogicExpr(key.scalar, node_names[node.lhs], "&&",
                                   node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::PredicateOr)) {
    return MetalPredicateLogicExpr(key.scalar, node_names[node.lhs], "||",
                                   node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::BitAnd)) {
    return MetalBitBinaryExpr(key.scalar, node_names[node.lhs], "&",
                              node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::BitOr)) {
    return MetalBitBinaryExpr(key.scalar, node_names[node.lhs], "|",
                              node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::BitXor)) {
    return MetalBitBinaryExpr(key.scalar, node_names[node.lhs], "^",
                              node_names[node.rhs]);
  }
  const char *const op_text = op == IrOp::Eq                             ? "=="
                              : op == IrOp::Lt || op == IrOp::LtUnsigned ? "<"
                              : op == IrOp::Le || op == IrOp::LeUnsigned ? "<="
                              : op == IrOp::Ne                           ? "!="
                              : op == IrOp::Gt || op == IrOp::GtUnsigned ? ">"
                                                                         : ">=";
  return MetalPredicateExpr(domain, key.scalar, node_names[node.lhs], op_text,
                            node_names[node.rhs]);
}

[[nodiscard]] inline std::string
MetalShiftNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                   const std::vector<std::string> &node_names) {
  if (node.op == static_cast<u8>(IrOp::ShrArithmeticConst)) {
    return MetalArithmeticShiftExpr(key.scalar, node_names[node.lhs], node.aux);
  }
  return MetalShiftExpr(key.scalar, static_cast<IrOp>(node.op),
                        node_names[node.lhs], node.aux);
}

[[nodiscard]] inline std::string
MetalClampNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                   const std::vector<std::string> &node_names) {
  const ComputeDomain domain =
      static_cast<IrOp>(node.op) == IrOp::ClampUnsigned
          ? (key.scalar == ComputeScalar::Lane64 ? ComputeDomain::U64
                                                 : ComputeDomain::U32)
      : key.domain == ComputeDomain::Fixed  ? ComputeDomain::Fixed
      : key.scalar == ComputeScalar::Lane64 ? ComputeDomain::I64
                                            : ComputeDomain::I32;
  const std::string maxed =
      "(" +
      MetalMaxExpr(domain, key.scalar, node_names[node.lhs],
                   node_names[node.rhs]) +
      ")";
  return MetalMinExpr(domain, key.scalar, maxed, node_names[node.aux]);
}

[[nodiscard]] inline std::string
MetalMulAddNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                    const std::vector<std::string> &node_names) {
  return FixedOpExpr(key.scalar, static_cast<IrOp>(node.op),
                     node_names[node.lhs], node_names[node.rhs],
                     node_names[node.aux]);
}

inline void AppendMetalWriteNode(std::string &out, const ParsedIR &parsed,
                                 const ArtifactKey &key,
                                 const std::vector<BindingLayout> &layouts,
                                 const ParsedNode &node,
                                 const std::vector<std::string> &node_names) {
  const BindingLayout &layout = layouts[node.aux];
  const ParsedBinding &binding = parsed.bindings[node.aux];
  const ComputeScalar store_scalar = MetalStoreScalar(binding.element_bytes);
  out += "  ";
  out += MetalStoreFunction(store_scalar);
  out += "(";
  out += layout.symbol;
  out += ", ";
  out += BindingBaseSymbol(layout);
  out += " + gid * ";
  out += BindingStrideSymbol(layout);
  out += ", ";
  if (store_scalar != key.scalar) {
    out += store_scalar == ComputeScalar::Lane64 ? "long(uint(" : "int(";
  }
  out += node_names[node.lhs];
  if (store_scalar != key.scalar) {
    out += store_scalar == ComputeScalar::Lane64 ? "))" : ")";
  }
  out += ");\n";
}

#include <kernel/program/compute/lowering/metal/source/node.hpp>

[[nodiscard]] inline std::string
MetalSource(const ParsedIR &parsed, const ArtifactKey &key,
            const std::vector<BindingLayout> &layouts) {
  std::string out;
  AppendMetalSourcePreamble(out, parsed, key, layouts);
  AppendMetalNodeBody(out, parsed, key, layouts);
  out += "}\n";
  return out;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
