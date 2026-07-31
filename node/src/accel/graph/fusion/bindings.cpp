#include "local.hpp"

#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/lowering/fusion/graph.hpp>

#include <string>
#include <string_view>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool
PrefixedBindingNameMatches(const char *const name_prefix,
                           const std::string &local_name,
                           const std::string &metadata_name) noexcept {
  const std::string_view prefix{name_prefix == nullptr ? "" : name_prefix};
  const std::string_view local{local_name};
  const std::string_view metadata{metadata_name};
  if (metadata.size() != prefix.size() + local.size()) {
    return false;
  }
  return metadata.substr(0u, prefix.size()) == prefix &&
         metadata.substr(prefix.size()) == local;
}

[[nodiscard]] bool AppendFusedBinding(
    FusedStepBindings &bindings,
    const rund::kernel::ExecutionMetadata &metadata,
    const GraphCompileNode &node, const std::uint64_t graph_buffer_index,
    const char *const name_prefix, const rund::kernel::BufferRole role) {
  const std::size_t local = static_cast<std::size_t>(graph_buffer_index);
  const auto &names = node.map_metadata.binding_names;
  if (local >= names.size() || local >= node.binding_indices.size() ||
      bindings.metadata_index >= metadata.binding_accesses.size() ||
      bindings.metadata_index >= metadata.binding_names.size()) {
    return false;
  }
  if (metadata.binding_accesses[bindings.metadata_index] !=
          rund::kernel::ComputeAccessFor(role) ||
      !PrefixedBindingNameMatches(
          name_prefix, names[local],
          metadata.binding_names[bindings.metadata_index])) {
    return false;
  }
  if (!bindings.indices.push_back(node.binding_indices[local])) {
    return false;
  }
  ++bindings.metadata_index;
  return true;
}

} // namespace

bool FusedStepBindingsMatchMetadata(
    const FusedStepBindings &bindings,
    const rund::kernel::ExecutionMetadata &metadata) noexcept {
  return metadata.binding_accesses.size() == metadata.binding_names.size() &&
         bindings.indices.valid() &&
         bindings.metadata_index == bindings.indices.size() &&
         bindings.metadata_index == metadata.binding_accesses.size();
}

FusedStepBindings
FusedStepBindingsFor(const rund::kernel::Graph &graph,
                     const std::span<const GraphCompileNode> nodes,
                     const rund::kernel::ExecutionMetadata &metadata) {
  FusedStepBindings bindings{};
  if (graph.node_count < 2u || graph.nodes == nullptr ||
      nodes.size() != graph.node_count) {
    return bindings;
  }
  bindings.indices.reserve(
      static_cast<std::uint64_t>(metadata.binding_accesses.size()));
  if (!bindings.indices.ok) {
    return bindings;
  }
  for (std::uint64_t node_index = 0u; node_index < graph.node_count;
       ++node_index) {
    const rund::kernel::GraphNode &node = graph.nodes[node_index];
    const std::uint64_t intermediate =
        node_index == 0u
            ? 0u
            : rund::kernel::compute_lowering_detail::IntermediateLogicalId(
                  graph, node_index - 1u);
    if (node_index != 0u && intermediate == 0u) {
      return bindings;
    }
    const std::string prefix = "f" + std::to_string(node_index) + "_";
    for (std::uint64_t index = 0u; index < node.buffer_count; ++index) {
      const rund::kernel::GraphBufferRef &buffer = node.buffers[index];
      if (buffer.role == rund::kernel::BufferRole::Read &&
          buffer.logical_id != intermediate &&
          !AppendFusedBinding(bindings, metadata, nodes[node_index], index,
                              prefix.c_str(), rund::kernel::BufferRole::Read)) {
        return bindings;
      }
    }
  }

  const std::uint64_t final_index = graph.node_count - 1u;
  const rund::kernel::GraphNode &final = graph.nodes[final_index];
  const std::string prefix = "f" + std::to_string(final_index) + "_";
  std::uint64_t write_count = 0u;
  for (std::uint64_t index = 0u; index < final.buffer_count; ++index) {
    if (final.buffers[index].role != rund::kernel::BufferRole::Write) {
      continue;
    }
    if (!AppendFusedBinding(bindings, metadata, nodes[final_index], index,
                            prefix.c_str(), rund::kernel::BufferRole::Write)) {
      return bindings;
    }
    ++write_count;
  }
  bindings.ok =
      write_count != 0u && FusedStepBindingsMatchMetadata(bindings, metadata);
  return bindings;
}

} // namespace rund::node::accel::detail
