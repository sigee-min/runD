#pragma once

#include "wide/basic.hpp"
#include "wide/context.hpp"
#include "wide/fixed.hpp"

[[nodiscard]] inline bool AppendVulkanWideCoreNode(
    std::string &out, const ParsedIR &parsed, const ArtifactKey &key,
    const std::vector<BindingLayout> &layouts, const ParsedNode &node,
    const u32 current_node, std::vector<std::string> &node_names) {
  VulkanWideNodeContext context{out,  parsed,       key,       layouts,
                                node, current_node, node_names};
  if (AppendVulkanWideBasicNode(context)) {
    return true;
  }
  return AppendVulkanWideFixedNode(context);
}
