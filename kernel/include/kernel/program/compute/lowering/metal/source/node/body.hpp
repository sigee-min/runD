#pragma once

#include "ordinary.hpp"

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
