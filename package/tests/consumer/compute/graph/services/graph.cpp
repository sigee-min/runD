#include "local.hpp"

#include <rund/compute.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace package_compute::graph_services {

namespace {

bool Count(const rund::compute::graph::Info &graph, const std::uint32_t id) {
  if (id == 0u || id > graph.resources.size()) {
    return false;
  }
  const auto &value = graph.resources[id - 1u];
  return value.elements == 1u &&
         (value.type == rund::compute::graph::Value::U32 ||
          value.type == rund::compute::graph::Value::U64);
}

bool SameStorage(const rund::compute::graph::Resource &left,
                 const rund::compute::graph::Resource &right) {
  return left.type == right.type && left.integer_bits == right.integer_bits &&
         left.fraction_bits == right.fraction_bits &&
         left.rounding == right.rounding && left.overflow == right.overflow &&
         left.approximation == right.approximation &&
         left.elements == right.elements &&
         left.element_bytes == right.element_bytes && left.bytes == right.bytes;
}

bool Destructive(const rund::compute::graph::Resource &source,
                 const rund::compute::graph::Resource &target) {
  return target.source == source.id && source.last_use == target.first_use &&
         source.alias_offset_bytes == target.alias_offset_bytes &&
         SameStorage(source, target);
}

} // namespace

bool ValidateGraph(const rund::compute::graph::Info &graph) {
  using rund::compute::graph::Visibility;
  using rund::compute::resource::AccessMode;
  if (!graph.fingerprint || graph.resources.empty() || graph.nodes.empty() ||
      graph.inputs.empty() || graph.outputs.empty()) {
    return false;
  }
  std::vector<std::uint64_t> extents(graph.resources.size());
  std::vector<bool> groups(graph.resources.size());
  std::uint64_t logical = 0u;
  for (std::size_t index = 0u; index < graph.resources.size(); ++index) {
    const auto &resource = graph.resources[index];
    if (resource.id != index + 1u || resource.element_bytes == 0u ||
        resource.elements > std::numeric_limits<std::uint64_t>::max() /
                                resource.element_bytes ||
        resource.bytes != resource.elements * resource.element_bytes ||
        resource.alias_group == 0u ||
        resource.alias_group > graph.resources.size() ||
        (resource.active != 0u && !Count(graph, resource.active)) ||
        (resource.parent != 0u &&
         (!Count(graph, resource.id) || !Count(graph, resource.parent))) ||
        (resource.source != 0u && (resource.source > graph.resources.size() ||
                                   resource.source == resource.id)) ||
        (resource.first_use == rund::compute::resource::NoNode) !=
            (resource.last_use == rund::compute::resource::NoNode) ||
        (resource.first_use != rund::compute::resource::NoNode &&
         resource.first_use > resource.last_use)) {
      return false;
    }
    std::uint32_t ancestor = resource.parent;
    for (std::size_t depth = 0u; ancestor != 0u; ++depth) {
      if (depth >= graph.resources.size() || ancestor == resource.id) {
        return false;
      }
      ancestor = graph.resources[ancestor - 1u].parent;
    }
    if (resource.visibility != Visibility::Internal) {
      if (resource.alias_group != resource.id ||
          resource.alias_offset_bytes != 0u || resource.requires_reset()) {
        return false;
      }
      continue;
    }
    if (resource.bytes > std::numeric_limits<std::uint64_t>::max() - logical ||
        resource.alias_offset_bytes % 256u != 0u ||
        resource.alias_offset_bytes >
            std::numeric_limits<std::uint64_t>::max() - resource.bytes) {
      return false;
    }
    logical += resource.bytes;
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
    groups[group] = true;
    extents[group] =
        std::max(extents[group], resource.alias_offset_bytes + resource.bytes);
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
      const bool live_overlap = !(first.last_use < second.first_use ||
                                  second.last_use < first.first_use);
      const bool byte_overlap =
          first.alias_offset_bytes < second.alias_offset_bytes + second.bytes &&
          second.alias_offset_bytes < first.alias_offset_bytes + first.bytes;
      const bool proved_alias =
          Destructive(first, second) || Destructive(second, first);
      if ((live_overlap || first.requires_reset() || second.requires_reset()) &&
          byte_overlap && !proved_alias) {
        return false;
      }
    }
  }
  std::uint64_t physical = 0u;
  std::uint64_t allocations = 0u;
  for (std::size_t group = 0u; group < groups.size(); ++group) {
    if (!groups[group]) {
      continue;
    }
    if (extents[group] > std::numeric_limits<std::uint64_t>::max() - physical) {
      return false;
    }
    physical += extents[group];
    ++allocations;
  }
  std::uint64_t live = 0u;
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
    live = std::max(live, current);
  }
  if (graph.memory.logical_bytes != logical ||
      graph.memory.live_bytes != live ||
      graph.memory.physical_bytes != physical ||
      graph.memory.allocation_count != allocations) {
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
  std::uint64_t reads = 0u;
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
        const std::uint64_t bytes = access.element_bytes * access.element_count;
        reads = bytes > std::numeric_limits<std::uint64_t>::max() - reads
                    ? std::numeric_limits<std::uint64_t>::max()
                    : reads + bytes;
      }
    }
  }
  if (graph.read_bytes != reads) {
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

} // namespace package_compute::graph_services
