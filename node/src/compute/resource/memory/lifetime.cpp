#include "local.hpp"

#include "../../size.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::resource_detail::memory_detail {

Status collect(const graph::Info &info, const std::span<const MemoryNode> nodes,
               Work &work) {
  for (std::size_t node_index = 0u; node_index < info.nodes.size();
       ++node_index) {
    const graph::Node &node = info.nodes[node_index];
    if (node.index != node_index) {
      return Status::fail(Reason::GraphInvalid);
    }
    for (const graph::Access &access : node.accesses) {
      if (access.resource == 0u || access.resource > work.lifetimes.size() ||
          (access.mode != resource::AccessMode::Read &&
           access.mode != resource::AccessMode::Write)) {
        return Status::fail(Reason::GraphInvalid);
      }
      std::size_t &count = work.counts[access.resource - 1u];
      if (count == std::numeric_limits<std::size_t>::max()) {
        return Status::fail(Reason::GraphCapacity);
      }
      ++count;
      Lifetime &lifetime = work.lifetimes[access.resource - 1u];
      if (lifetime.first == resource::NoNode) {
        lifetime.first = node.index;
      }
      lifetime.last = node.index;
      if (access.mode == resource::AccessMode::Read) {
        if (lifetime.first_read == resource::NoNode) {
          lifetime.first_read = node.index;
        }
      } else if (lifetime.first_write == resource::NoNode) {
        lifetime.first_write = node.index;
        lifetime.first_write_dense =
            whole_value(info.resources[access.resource - 1u], access);
        lifetime.first_write_complete = lifetime.first_write_dense &&
                                        nodes[node.index].write == Write::Full;
        lifetime.domain = nodes[node.index].domain;
        if (lifetime.domain.empty() &&
            info.resources[access.resource - 1u].active != 0u) {
          lifetime.domain = Domain{
              .count = info.resources[access.resource - 1u].active,
          };
        }
        lifetime.first_write_domain = lifetime.first_write_dense &&
                                      !lifetime.domain.empty() &&
                                      nodes[node.index].write == Write::Domain;
      } else if (lifetime.first_write == node.index) {
        const bool dense =
            whole_value(info.resources[access.resource - 1u], access);
        lifetime.first_write_dense = lifetime.first_write_dense && dense;
        lifetime.first_write_complete = lifetime.first_write_complete &&
                                        dense &&
                                        nodes[node.index].write == Write::Full;
        lifetime.first_write_domain =
            lifetime.first_write_domain && dense &&
            nodes[node.index].write == Write::Domain &&
            (lifetime.domain == nodes[node.index].domain ||
             (nodes[node.index].domain.empty() &&
              info.resources[access.resource - 1u].active ==
                  lifetime.domain.count));
      }
    }
  }

  work.offsets.resize(info.resources.size() + 1u);
  for (std::size_t index = 0u; index < work.counts.size(); ++index) {
    if (!size::add(work.offsets[index], work.counts[index],
                   work.offsets[index + 1u])) {
      return Status::fail(Reason::GraphCapacity);
    }
  }
  work.uses.resize(work.offsets.back());
  std::copy(work.offsets.begin(), work.offsets.end() - 1u, work.counts.begin());
  for (const graph::Node &node : info.nodes) {
    for (const graph::Access &access : node.accesses) {
      work.uses[work.counts[access.resource - 1u]++] = node.index;
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail::resource_detail::memory_detail
