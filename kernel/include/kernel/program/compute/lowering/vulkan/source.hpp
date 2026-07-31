#pragma once

#include <kernel/program/compute/lowering/fixed/ops.hpp>
#include <kernel/program/compute/lowering/format.hpp>
#include <kernel/program/compute/lowering/vulkan/expr.hpp>
#include <kernel/program/compute/lowering/vulkan/io.hpp>
#include <kernel/program/compute/lowering/vulkan/shape.hpp>
#include <kernel/program/compute/lowering/vulkan/wide.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

inline void AppendVulkanIntegerDivHelpers(std::string &out,
                                          const ParsedIR &parsed,
                                          const ArtifactKey &key) {
  const bool signed_divide = ParsedIrHasOp(parsed, IrOp::DivSigned);
  const bool unsigned_divide = ParsedIrHasOp(parsed, IrOp::DivUnsigned);
  if (key.scalar == ComputeScalar::Lane32) {
    if (signed_divide) {
      out +=
          "uint RundDivSigned32(uint lhs, uint rhs) { return uint(int(lhs) / "
          "int(rhs)); }\n";
    }
    if (unsigned_divide) {
      out +=
          "uint RundDivUnsigned32(uint lhs, uint rhs) { return lhs / rhs; }\n";
    }
    return;
  }
  if (signed_divide) {
    out += "uint64_t RundDivSigned64(uint64_t lhs, uint64_t rhs) { return "
           "uint64_t(int64_t(lhs) / int64_t(rhs)); }\n";
  }
  if (unsigned_divide) {
    out += "uint64_t RundDivUnsigned64(uint64_t lhs, uint64_t rhs) { return "
           "lhs / rhs; }\n";
  }
}

inline void
AppendVulkanSourcePreamble(std::string &out, const ParsedIR &parsed,
                           const ArtifactKey &key,
                           const std::vector<BindingLayout> &layouts) {
  out += "#version 450\n";
  out += "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  out += "// rund.compute.vulkan.source\n";
  out += "// operation_hex=";
  out += HexText(parsed.name);
  out += "\n";
  AppendKey(out, key, "// ");
  AppendBindingLayout(out, parsed, layouts, "// ");
  AppendNodeLayout(out, parsed, "// ");
  out += "layout(local_size_x = ";
  out += std::to_string(kVulkanMapWidth);
  out += ", local_size_y = 1, local_size_z = 1) in;\n";
  out += "layout(push_constant) uniform RundDispatch {\n";
  out += "  uint tile_count;\n";
  out += "  uint iterations;\n";
  out += "} rund_dispatch;\n";
  AppendVulkanBuffers(out, parsed, layouts);
  AppendVulkanHelpers(out, parsed, key, layouts);
  if (key.domain == ComputeDomain::Fixed) {
    AppendVulkanWideFixedHelpers(out);
  }
  AppendVulkanIntegerDivHelpers(out, parsed, key);
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    if (binding.kind != 2u && binding.kind != 3u) {
      continue;
    }
    out += "const uint ";
    out += BindingBaseSymbol(layouts[index]);
    out += " = 0u;\n";
    out += "const uint ";
    out += BindingStrideSymbol(layouts[index]);
    out += " = ";
    out += std::to_string(binding.element_bytes);
    out += "u;\n";
  }
  out += "void main() {\n";
  out += "  const uint gid = gl_GlobalInvocationID.x;\n";
  out += "  if (gid >= rund_dispatch.tile_count) { return; }\n";
}
inline std::string &SetVulkanNodeName(std::vector<std::string> &node_names,
                                      const u32 current_node) {
  node_names[current_node] = "node_" + std::to_string(current_node);
  return node_names[current_node];
}

inline void AppendVulkanAssignedValue(std::string &out,
                                      const ComputeScalar scalar,
                                      const std::string &name,
                                      const std::string &expr) {
  out += "  const ";
  out += VulkanType(scalar);
  out += " ";
  out += name;
  out += " = ";
  out += expr;
  out += ";\n";
}

[[nodiscard]] inline std::string
VulkanParamNodeExpr(const ArtifactKey &key, const BindingLayout &layout) {
  std::string expr = VulkanParamLoadFunction(key.scalar);
  expr += "(";
  expr += std::to_string(layout.param_offset);
  expr += "u)";
  return expr;
}

[[nodiscard]] inline std::string VulkanReadNodeExpr(const ArtifactKey &key,
                                                    const BindingLayout &layout,
                                                    const ParsedBinding &) {
  std::string expr = VulkanLoadFunctionName(key.scalar, layout);
  expr += "(";
  expr += BindingBaseSymbol(layout);
  expr += " + gid * ";
  expr += BindingStrideSymbol(layout);
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
VulkanReadUniformNodeExpr(const ArtifactKey &key,
                          const BindingLayout &layout) {
  std::string expr = VulkanLoadFunctionName(key.scalar, layout);
  expr += "(";
  expr += BindingBaseSymbol(layout);
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
VulkanReadAtNodeExpr(const ArtifactKey &key, const BindingLayout &source,
                     const BindingLayout &index) {
  std::string offset = VulkanLoadFunctionName(ComputeScalar::Lane32, index);
  offset += "(";
  offset += BindingBaseSymbol(index);
  offset += " + gid * ";
  offset += BindingStrideSymbol(index);
  offset += ")";
  std::string expr = VulkanLoadFunctionName(key.scalar, source);
  expr += "(";
  expr += BindingBaseSymbol(source);
  expr += " + ";
  expr += offset;
  expr += " * ";
  expr += BindingStrideSymbol(source);
  expr += ")";
  return expr;
}

[[nodiscard]] inline std::string
VulkanBasicBinaryNodeExpr(const ParsedNode &node,
                          const std::vector<std::string> &node_names) {
  const char *op_text = node.op == static_cast<u8>(IrOp::Add)   ? "+"
                        : node.op == static_cast<u8>(IrOp::Sub) ? "-"
                                                                : "*";
  return node_names[node.lhs] + " " + op_text + " " + node_names[node.rhs];
}

[[nodiscard]] inline std::string
VulkanUnaryNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                    const std::vector<std::string> &node_names) {
  if (node.op == static_cast<u8>(IrOp::Neg)) {
    return VulkanWrapNegExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::Abs)) {
    return VulkanAbsExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::AbsMagnitude)) {
    return VulkanAbsMagnitudeExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::Sign)) {
    return VulkanSignExpr(key.scalar, node_names[node.lhs]);
  }
  if (node.op == static_cast<u8>(IrOp::BitNot)) {
    return "~" + node_names[node.lhs];
  }
  if (node.op == static_cast<u8>(IrOp::PredicateNot)) {
    return VulkanPredicateNotExpr(key.scalar, node_names[node.lhs]);
  }
  return FixedOpExpr(key.scalar, static_cast<IrOp>(node.op),
                     node_names[node.lhs], {}, {});
}

[[nodiscard]] inline std::string
VulkanFixedBinaryNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                          const std::vector<std::string> &node_names) {
  return FixedOpExpr(key.scalar, static_cast<IrOp>(node.op),
                     node_names[node.lhs], node_names[node.rhs], {});
}

[[nodiscard]] inline std::string
VulkanIntegerDivNodeExpr(const ArtifactKey &key, const ParsedNode &node,
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
VulkanCompareLogicNodeExpr(const ArtifactKey &key, const ParsedNode &node,
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
    return VulkanMinExpr(domain, key.scalar, node_names[node.lhs],
                         node_names[node.rhs]);
  }
  if (op == IrOp::Max || op == IrOp::MaxUnsigned) {
    return VulkanMaxExpr(domain, key.scalar, node_names[node.lhs],
                         node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::PredicateAnd)) {
    return VulkanPredicateLogicExpr(key.scalar, node_names[node.lhs], "&&",
                                    node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::PredicateOr)) {
    return VulkanPredicateLogicExpr(key.scalar, node_names[node.lhs], "||",
                                    node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::BitAnd)) {
    return node_names[node.lhs] + " & " + node_names[node.rhs];
  }
  if (node.op == static_cast<u8>(IrOp::BitOr)) {
    return node_names[node.lhs] + " | " + node_names[node.rhs];
  }
  if (node.op == static_cast<u8>(IrOp::BitXor)) {
    return node_names[node.lhs] + " ^ " + node_names[node.rhs];
  }
  if (node.op == static_cast<u8>(IrOp::Eq)) {
    return VulkanPredicateExpr(key.scalar, node_names[node.lhs] +
                                               " == " + node_names[node.rhs]);
  }
  if (node.op == static_cast<u8>(IrOp::Ne)) {
    return VulkanPredicateExpr(key.scalar, node_names[node.lhs] +
                                               " != " + node_names[node.rhs]);
  }
  const char *const op_text = op == IrOp::Lt || op == IrOp::LtUnsigned   ? "<"
                              : op == IrOp::Le || op == IrOp::LeUnsigned ? "<="
                              : op == IrOp::Gt || op == IrOp::GtUnsigned ? ">"
                                                                         : ">=";
  return VulkanPredicateExpr(
      key.scalar, VulkanCompareExpr(domain, key.scalar, node_names[node.lhs],
                                    op_text, node_names[node.rhs]));
}

[[nodiscard]] inline std::string
VulkanShiftNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                    const std::vector<std::string> &node_names) {
  if (node.op == static_cast<u8>(IrOp::ShrArithmeticConst)) {
    return VulkanArithmeticShiftExpr(key.scalar, node_names[node.lhs],
                                     node.aux);
  }
  std::string expr = node_names[node.lhs];
  expr += node.op == static_cast<u8>(IrOp::ShlConst) ? " << " : " >> ";
  expr += std::to_string(node.aux);
  expr += "u";
  return expr;
}

[[nodiscard]] inline std::string
VulkanClampNodeExpr(const ArtifactKey &key, const ParsedNode &node,
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
      VulkanMaxExpr(domain, key.scalar, node_names[node.lhs],
                    node_names[node.rhs]) +
      ")";
  return VulkanMinExpr(domain, key.scalar, maxed, node_names[node.aux]);
}

[[nodiscard]] inline std::string
VulkanMulAddNodeExpr(const ArtifactKey &key, const ParsedNode &node,
                     const std::vector<std::string> &node_names) {
  return FixedOpExpr(key.scalar, static_cast<IrOp>(node.op),
                     node_names[node.lhs], node_names[node.rhs],
                     node_names[node.aux]);
}

inline void AppendVulkanWriteNode(std::string &out, const ParsedIR &parsed,
                                  const ArtifactKey &key,
                                  const std::vector<BindingLayout> &layouts,
                                  const ParsedNode &node,
                                  const std::vector<std::string> &node_names) {
  const BindingLayout &layout = layouts[node.aux];
  const ParsedBinding &binding = parsed.bindings[node.aux];
  const ComputeScalar store_scalar = VulkanStoreScalar(binding.element_bytes);
  out += "  ";
  out += VulkanStoreFunctionName(store_scalar, layout);
  out += "(";
  out += BindingBaseSymbol(layout);
  out += " + gid * ";
  out += BindingStrideSymbol(layout);
  out += ", ";
  if (store_scalar != key.scalar) {
    out += store_scalar == ComputeScalar::Lane64 ? "uint64_t(" : "uint(";
  }
  out += node_names[node.lhs];
  if (store_scalar != key.scalar) {
    out += ")";
  }
  out += ");\n";
}

#include <kernel/program/compute/lowering/vulkan/source/node.hpp>

[[nodiscard]] inline std::string
VulkanSource(const ParsedIR &parsed, const ArtifactKey &key,
             const std::vector<BindingLayout> &layouts) {
  std::string out;
  AppendVulkanSourcePreamble(out, parsed, key, layouts);
  AppendVulkanNodeBody(out, parsed, key, layouts);
  out += "}\n";
  return out;
}

} // namespace compute_lowering_detail
} // namespace rund::kernel
