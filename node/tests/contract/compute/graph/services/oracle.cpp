#include "local.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace rund_node_graph_services {

bool ValidResourceGraph(const Info &graph) {
  if (!graph.fingerprint || graph.resources.empty() || graph.nodes.empty() ||
      graph.inputs.empty() || graph.outputs.empty() ||
      graph.authored_nodes < graph.lowered_nodes ||
      graph.lowered_nodes != graph.nodes.size()) {
    return false;
  }
  std::uint64_t logical_bytes = 0u;
  std::vector<std::uint64_t> group_extents(graph.resources.size());
  std::vector<bool> group_present(graph.resources.size());
  for (const auto &resource : graph.resources) {
    if (resource.id == 0u || resource.id > graph.resources.size() ||
        resource.element_bytes == 0u ||
        resource.elements > std::numeric_limits<std::uint64_t>::max() /
                                resource.element_bytes ||
        resource.bytes != resource.elements * resource.element_bytes ||
        resource.alias_group == 0u) {
      return false;
    }
    const auto valid_count = [&](const std::uint32_t id) {
      if (id == 0u || id > graph.resources.size()) {
        return false;
      }
      const auto &count = graph.resources[id - 1u];
      return count.elements == 1u &&
             (count.type == rund::compute::graph::Value::U32 ||
              count.type == rund::compute::graph::Value::U64);
    };
    if ((resource.active != 0u && !valid_count(resource.active)) ||
        (resource.parent != 0u &&
         (!valid_count(resource.id) || !valid_count(resource.parent)))) {
      return false;
    }
    if (resource.source != 0u && (resource.source > graph.resources.size() ||
                                  resource.source == resource.id)) {
      return false;
    }
    std::uint32_t ancestor = resource.parent;
    for (std::size_t depth = 0u; ancestor != 0u; ++depth) {
      if (depth >= graph.resources.size() || ancestor == resource.id) {
        return false;
      }
      ancestor = graph.resources[ancestor - 1u].parent;
    }
    if (resource.alias_group > graph.resources.size() ||
        (resource.first_use == rund::compute::resource::NoNode) !=
            (resource.last_use == rund::compute::resource::NoNode) ||
        (resource.first_use != rund::compute::resource::NoNode &&
         resource.first_use > resource.last_use)) {
      return false;
    }
    if (resource.visibility != Visibility::Internal) {
      if (resource.alias_group != resource.id ||
          resource.alias_offset_bytes != 0u || resource.requires_reset()) {
        return false;
      }
      continue;
    }
    if (resource.bytes >
            std::numeric_limits<std::uint64_t>::max() - logical_bytes ||
        resource.alias_offset_bytes % resource.element_bytes != 0u ||
        resource.alias_offset_bytes >
            std::numeric_limits<std::uint64_t>::max() - resource.bytes) {
      return false;
    }
    logical_bytes += resource.bytes;
    if (resource.first_use == rund::compute::resource::NoNode) {
      if (resource.alias_group != resource.id ||
          resource.alias_offset_bytes != 0u || resource.requires_reset()) {
        return false;
      }
      continue;
    }
    const auto &representative = graph.resources[resource.alias_group - 1u];
    if (representative.visibility != Visibility::Internal ||
        representative.alias_group != resource.alias_group ||
        representative.first_use == rund::compute::resource::NoNode ||
        representative.reset_node != resource.reset_node) {
      return false;
    }
    const std::size_t group = resource.alias_group - 1u;
    group_present[group] = true;
    group_extents[group] = std::max(
        group_extents[group], resource.alias_offset_bytes + resource.bytes);
  }
  for (std::size_t left = 0u; left < graph.resources.size(); ++left) {
    for (std::size_t right = left + 1u; right < graph.resources.size();
         ++right) {
      const auto &first = graph.resources[left];
      const auto &second = graph.resources[right];
      if (first.alias_group != second.alias_group) {
        continue;
      }
      if (first.visibility != Visibility::Internal ||
          second.visibility != Visibility::Internal ||
          first.first_use == rund::compute::resource::NoNode ||
          second.first_use == rund::compute::resource::NoNode) {
        return false;
      }
      const bool lifetime_overlap = !(first.last_use < second.first_use ||
                                      second.last_use < first.first_use);
      const bool range_overlap =
          first.alias_offset_bytes < second.alias_offset_bytes + second.bytes &&
          second.alias_offset_bytes < first.alias_offset_bytes + first.bytes;
      const bool destructive =
          (second.source == first.id && first.last_use == second.first_use) ||
          (first.source == second.id && second.last_use == first.first_use);
      if ((lifetime_overlap || first.requires_reset() ||
           second.requires_reset()) &&
          range_overlap && !destructive) {
        return false;
      }
    }
  }
  std::uint64_t physical_bytes = 0u;
  std::uint64_t allocation_count = 0u;
  for (std::size_t group = 0u; group < group_extents.size(); ++group) {
    if (!group_present[group]) {
      continue;
    }
    if (group_extents[group] >
        std::numeric_limits<std::uint64_t>::max() - physical_bytes) {
      return false;
    }
    physical_bytes += group_extents[group];
    ++allocation_count;
  }
  std::uint64_t live_bytes = 0u;
  for (std::uint32_t node = 0u; node < graph.nodes.size(); ++node) {
    std::uint64_t current = 0u;
    for (const auto &resource : graph.resources) {
      if (resource.visibility != Visibility::Internal ||
          resource.first_use == rund::compute::resource::NoNode ||
          node < resource.first_use || node > resource.last_use) {
        continue;
      }
      if (resource.bytes >
          std::numeric_limits<std::uint64_t>::max() - current) {
        return false;
      }
      current += resource.bytes;
    }
    live_bytes = std::max(live_bytes, current);
  }
  if (graph.memory.logical_bytes != logical_bytes ||
      graph.memory.live_bytes != live_bytes ||
      graph.memory.physical_bytes != physical_bytes ||
      graph.memory.allocation_count != allocation_count) {
    return false;
  }
  for (const auto input : graph.inputs) {
    if (input == 0u || input > graph.resources.size() ||
        graph.resources[input - 1u].visibility != Visibility::Input) {
      return false;
    }
  }
  for (const auto output : graph.outputs) {
    if (output == 0u || output > graph.resources.size() ||
        graph.resources[output - 1u].visibility != Visibility::Output) {
      return false;
    }
  }
  std::uint64_t read_bytes = 0u;
  for (const auto &node : graph.nodes) {
    if (node.index >= graph.nodes.size() || node.accesses.empty()) {
      return false;
    }
    for (const auto dependency : node.dependencies) {
      if (dependency >= node.index) {
        return false;
      }
    }
    for (const auto &access : node.accesses) {
      if (access.resource == 0u || access.resource > graph.resources.size()) {
        return false;
      }
      const auto &resource = graph.resources[access.resource - 1u];
      if (access.offset_bytes != 0u || access.size_bytes != resource.bytes ||
          access.element_bytes != resource.element_bytes ||
          access.element_count != resource.elements ||
          access.stride_bytes != resource.element_bytes ||
          resource.first_use > node.index || resource.last_use < node.index) {
        return false;
      }
      if (access.mode == AccessMode::Read) {
        read_bytes += access.element_bytes * access.element_count;
      }
    }
  }
  if (graph.read_bytes != read_bytes) {
    return false;
  }
  std::vector<bool> boundary(graph.nodes.size());
  for (const auto &barrier : graph.barriers) {
    if (barrier.alias_group == 0u || barrier.before_resource == 0u ||
        barrier.after_resource == 0u || barrier.size_bytes == 0u ||
        barrier.before_node >= barrier.after_node ||
        barrier.after_node >= graph.nodes.size() ||
        boundary[barrier.after_node] ||
        (barrier.before == AccessMode::Read &&
         barrier.after == AccessMode::Read)) {
      return false;
    }
    boundary[barrier.after_node] = true;
  }
  return true;
}

[[nodiscard]] bool
SameGraphResource(const rund::compute::graph::Resource &left,
                  const rund::compute::graph::Resource &right) {
  return left.id == right.id && left.type == right.type &&
         left.integer_bits == right.integer_bits &&
         left.fraction_bits == right.fraction_bits &&
         left.rounding == right.rounding && left.overflow == right.overflow &&
         left.approximation == right.approximation &&
         left.visibility == right.visibility &&
         left.reset_node == right.reset_node &&
         left.elements == right.elements &&
         left.element_bytes == right.element_bytes &&
         left.bytes == right.bytes && left.active == right.active &&
         left.parent == right.parent && left.source == right.source &&
         left.alias_group == right.alias_group &&
         left.alias_offset_bytes == right.alias_offset_bytes &&
         left.first_use == right.first_use && left.last_use == right.last_use;
}

[[nodiscard]] bool SameGraphAccess(const rund::compute::graph::Access &left,
                                   const rund::compute::graph::Access &right) {
  return left.resource == right.resource && left.mode == right.mode &&
         left.offset_bytes == right.offset_bytes &&
         left.size_bytes == right.size_bytes &&
         left.element_bytes == right.element_bytes &&
         left.element_count == right.element_count &&
         left.stride_bytes == right.stride_bytes;
}

[[nodiscard]] bool SameGraphNode(const rund::compute::graph::Node &left,
                                 const rund::compute::graph::Node &right) {
  if (left.index != right.index || left.operation != right.operation ||
      left.elements != right.elements ||
      left.dependencies != right.dependencies ||
      left.accesses.size() != right.accesses.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < left.accesses.size(); ++index) {
    if (!SameGraphAccess(left.accesses[index], right.accesses[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
SameGraphBarrier(const rund::compute::graph::Barrier &left,
                 const rund::compute::graph::Barrier &right) {
  return left.alias_group == right.alias_group &&
         left.before_resource == right.before_resource &&
         left.after_resource == right.after_resource &&
         left.offset_bytes == right.offset_bytes &&
         left.size_bytes == right.size_bytes &&
         left.before_offset_bytes == right.before_offset_bytes &&
         left.before_element_bytes == right.before_element_bytes &&
         left.before_element_count == right.before_element_count &&
         left.before_stride_bytes == right.before_stride_bytes &&
         left.after_offset_bytes == right.after_offset_bytes &&
         left.after_element_bytes == right.after_element_bytes &&
         left.after_element_count == right.after_element_count &&
         left.after_stride_bytes == right.after_stride_bytes &&
         left.before_node == right.before_node &&
         left.after_node == right.after_node && left.before == right.before &&
         left.after == right.after;
}

[[nodiscard]] bool SameGraph(const Info &left, const Info &right) {
  if (left.fingerprint != right.fingerprint || left.inputs != right.inputs ||
      left.outputs != right.outputs || left.memory != right.memory ||
      left.authored_nodes != right.authored_nodes ||
      left.lowered_nodes != right.lowered_nodes ||
      left.resources.size() != right.resources.size() ||
      left.nodes.size() != right.nodes.size() ||
      left.barriers.size() != right.barriers.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < left.resources.size(); ++index) {
    if (!SameGraphResource(left.resources[index], right.resources[index])) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.nodes.size(); ++index) {
    if (!SameGraphNode(left.nodes[index], right.nodes[index])) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.barriers.size(); ++index) {
    if (!SameGraphBarrier(left.barriers[index], right.barriers[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool MatchReference(CanonicalReference &reference,
                                  const Info &graph,
                                  const ExecutionEvidence execution) {
  if (!reference.initialized) {
    reference.initialized = true;
    reference.graph = graph;
    reference.execution = execution;
    return true;
  }
  return SameGraph(reference.graph, graph) &&
         reference.execution.graph_hash == execution.graph_hash &&
         reference.execution.output_hash == execution.output_hash;
}

} // namespace rund_node_graph_services
