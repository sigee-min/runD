#include "local.hpp"

#include <algorithm>
#include <cstdint>

namespace rund::compute::detail::resource_detail::memory_detail {
namespace {

[[nodiscard]] bool
count_subset(const std::span<const graph::Resource> resources,
             std::uint32_t child, const std::uint32_t parent) noexcept {
  if (parent == 0u) {
    return true;
  }
  for (std::size_t depth = 0u; child != 0u && depth < resources.size();
       ++depth) {
    const graph::Resource &count = resources[child - 1u];
    if (child == parent) {
      return true;
    }
    child = count.parent;
  }
  return false;
}

[[nodiscard]] bool
domain_subset(const Domain &read, const Domain &write,
              const std::span<const graph::Resource> resources) noexcept {
  if (write.empty()) {
    return read.empty();
  }
  if (read.empty()) {
    return false;
  }
  if (write.predicate != 0u &&
      (read.predicate != write.predicate || read.expected != write.expected)) {
    return false;
  }
  return count_subset(resources, read.count, write.count);
}

} // namespace

Status classify(graph::Info &info, const std::span<const MemoryNode> nodes,
                const std::uint64_t page_bytes, Work &work) {
  const auto add = [](std::uint64_t &total,
                      const std::uint64_t value) noexcept {
    return kernel::checked::add(total, value, total);
  };

  work.live.resize(info.nodes.size());
  work.arena.reserve(info.resources.size());
  for (std::size_t index = 0u; index < info.resources.size(); ++index) {
    graph::Resource &value = info.resources[index];
    const Lifetime lifetime = work.lifetimes[index];
    if (value.visibility != graph::Visibility::Internal) {
      if (value.visibility == graph::Visibility::Output &&
          lifetime.first_write != resource::NoNode &&
          !lifetime.first_write_complete) {
        if (lifetime.first_write != lifetime.first ||
            lifetime.first_read == lifetime.first) {
          return Status::fail(Reason::GraphInvalid);
        }
        value.reset_node = lifetime.first_write;
        if (!add(info.memory.reset_bytes, value.bytes) ||
            !add(info.memory.reset_count, 1u)) {
          return Status::fail(Reason::GraphCapacity);
        }
      }
      continue;
    }
    if (!add(info.memory.logical_bytes, value.bytes)) {
      return Status::fail(Reason::GraphCapacity);
    }
    if (value.bytes > page_bytes) {
      return Status::fail(Reason::GraphCapacity);
    }
    if (lifetime.first == resource::NoNode) {
      continue;
    }
    if (lifetime.first >= info.nodes.size() ||
        lifetime.last >= info.nodes.size()) {
      return Status::fail(Reason::GraphInvalid);
    }
    if (!add(work.live[lifetime.first].start, value.bytes) ||
        !add(work.live[lifetime.last].stop, value.bytes)) {
      return Status::fail(Reason::GraphCapacity);
    }
    if (lifetime.first_write != lifetime.first || !lifetime.first_write_dense ||
        lifetime.first_read == lifetime.first) {
      return Status::fail(Reason::GraphInvalid);
    }
    bool reusable = lifetime.first_write_complete;
    if (!reusable && lifetime.first_write_domain) {
      reusable = true;
      for (std::size_t cursor = work.offsets[index];
           cursor < work.offsets[index + 1u]; ++cursor) {
        const std::uint32_t node = work.uses[cursor];
        if (!domain_subset(nodes[node].domain, lifetime.domain,
                           info.resources)) {
          reusable = false;
          break;
        }
      }
    }
    if (!reusable) {
      value.reset_node = lifetime.first_write;
      if (!add(info.memory.reset_bytes, value.bytes) ||
          !add(info.memory.reset_count, 1u)) {
        return Status::fail(Reason::GraphCapacity);
      }
    }
    work.arena.push_back(value.id);
  }
  return Status::success();
}

} // namespace rund::compute::detail::resource_detail::memory_detail
