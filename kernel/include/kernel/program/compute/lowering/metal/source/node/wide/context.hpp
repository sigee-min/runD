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

struct MetalWideNodeContext final {
  std::string &out;
  const ParsedIR &parsed;
  const ArtifactKey &key;
  const std::vector<BindingLayout> &layouts;
  const ParsedNode &node;
  std::vector<std::string> &node_names;
  const u32 current_node;
  const std::string &name;
  const u32 width;
  const std::string width_text;
  const std::string fraction;
  const std::string rounding;
  const std::string overflow;

  MetalWideNodeContext(std::string &out_, const ParsedIR &parsed_,
                       const ArtifactKey &key_,
                       const std::vector<BindingLayout> &layouts_,
                       const ParsedNode &node_, const u32 current_node_,
                       std::vector<std::string> &node_names_)
      : out{out_}, parsed{parsed_}, key{key_}, layouts{layouts_}, node{node_},
        node_names{node_names_}, current_node{current_node_},
        name{SetMetalNodeName(node_names, current_node)},
        width{ComputeScalarBits(key.scalar)},
        width_text{std::to_string(width) + "u"},
        fraction{std::to_string(node.fixed_format.fraction_bits) + "u"},
        rounding{std::to_string(static_cast<u8>(node.fixed_format.rounding)) +
                 "u"},
        overflow{std::to_string(static_cast<u8>(node.fixed_format.overflow)) +
                 "u"} {}

  [[nodiscard]] ComputeFixedFormat SourceFormat(const u32 ref) const noexcept {
    return parsed.nodes[ref - 1u].fixed_format;
  }

  [[nodiscard]] std::string Lane(const u32 ref) const {
    return MetalWideLaneExpr(key, node_names[ref]);
  }

  [[nodiscard]] std::string Wrap(const std::string &expr) const {
    return MetalWideFromLaneExpr(key, expr);
  }

  [[nodiscard]] std::string Align(const u32 ref,
                                  const ComputeFixedFormat target) const {
    return MetalWideAlign(node_names[ref], SourceFormat(ref), target);
  }

  [[nodiscard]] std::string Quantized(const std::string &value,
                                      const u32 source_fraction) const {
    return "RundWideQuantize(" + value + ", " +
           std::to_string(source_fraction) + "u, " + fraction + ", " +
           rounding + ", " + overflow + ", " + width_text + ")";
  }

  [[nodiscard]] std::string PhaseLane(const u32 ref) const {
    const std::string value = Lane(ref);
    const std::string shift =
        std::to_string(width - SourceFormat(ref).fraction_bits) + "u";
    return key.scalar == ComputeScalar::Lane64
               ? "RundAsSigned64(RundAsUnsigned64(" + value + ") << " + shift +
                     ")"
               : "int(uint(" + value + ") << " + shift + ")";
  }

  [[nodiscard]] std::string CanonicalLane(const u32 ref) const {
    const std::string canonical =
        "RundWideQuantize(" + node_names[ref] + ", " +
        std::to_string(SourceFormat(ref).fraction_bits) + "u, " +
        std::to_string(width - 1u) + "u, " + rounding + ", 1u, " + width_text +
        ")";
    return MetalWideLaneExpr(key, canonical);
  }
};
