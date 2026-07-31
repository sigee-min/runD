#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/layout.hpp>
#include <kernel/program/compute/lowering/text.hpp>

namespace rund::kernel {
namespace compute_lowering_detail {

[[nodiscard]] constexpr const char *
ArtifactVariantName(const LoweringArtifactVariant variant) noexcept {
  switch (variant) {
  case LoweringArtifactVariant::Canonical:
    return "canonical";
  case LoweringArtifactVariant::Controlled:
    return "controlled";
  case LoweringArtifactVariant::Recurrence:
    return "recurrence";
  }
  return "invalid";
}

inline void AppendKey(std::string &out, const ArtifactKey &key,
                      const char *const prefix) {
  out += prefix;
  out += "api=";
  out += ApiName(key.api);
  out += "\n";
  out += prefix;
  out += "scalar=";
  out += ScalarName(key.scalar);
  out += "\n";
  out += prefix;
  out += "artifact_variant=";
  out += ArtifactVariantName(key.variant);
  out += "\n";
  out += prefix;
  out += "op_hash_hi=";
  AppendHex64(out, key.op_hash_hi);
  out += "\n";
  out += prefix;
  out += "op_hash_lo=";
  AppendHex64(out, key.op_hash_lo);
  out += "\n";
  out += prefix;
  out += "canonical_ir_hash_hi=";
  AppendHex64(out, key.canonical_ir_hash_hi);
  out += "\n";
  out += prefix;
  out += "canonical_ir_hash_lo=";
  AppendHex64(out, key.canonical_ir_hash_lo);
  out += "\n";
}

inline void AppendBindingLayout(std::string &out, const ParsedIR &parsed,
                                const std::vector<BindingLayout> &layouts,
                                const char *const prefix) {
  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    const ParsedBinding &binding = parsed.bindings[index];
    const BindingLayout &layout = layouts[index];
    out += prefix;
    out += "binding[";
    out += std::to_string(index);
    out += "].kind=";
    out += BindingKindName(binding.kind);
    out += " name_hex=";
    out += layout.name_hex;
    out += " symbol=";
    out += layout.symbol;
    out += " element_bytes=";
    out += std::to_string(binding.element_bytes);
    if (binding.kind == 1u) {
      out += " value_bytes=";
      out += std::to_string(binding.value_bytes.size());
      out += " buffer=0";
      out += " param_offset=";
      out += std::to_string(layout.param_offset);
    } else if (binding.kind == 2u) {
      out += " stride_bytes=";
      out += std::to_string(binding.element_bytes);
      out += " buffer=";
      out += std::to_string(layout.buffer);
    } else {
      out += " stride_bytes=";
      out += std::to_string(binding.element_bytes);
      out += " buffer=";
      out += std::to_string(layout.buffer);
    }
    out += "\n";
  }
}

inline void AppendNodeLayout(std::string &out, const ParsedIR &parsed,
                             const char *const prefix) {
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const ParsedNode &node = parsed.nodes[index];
    out += prefix;
    out += "node[";
    out += std::to_string(index + 1u);
    out += "].op=";
    out += OpName(node.op);
    out += " lhs=";
    out += std::to_string(node.lhs);
    out += " rhs=";
    out += std::to_string(node.rhs);
    out += " aux=";
    out += std::to_string(node.aux);
    out += "\n";
  }
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
