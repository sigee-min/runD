#include "memory.hpp"
#include "memory/model.hpp"

#include "../size.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <tuple>
#include <vector>

namespace rund::compute::detail::resource_detail {
namespace {

using memory_detail::Alignment;
using memory_detail::Layout;
using memory_detail::Lifetime;
using memory_detail::pack;

struct Live final {
  std::uint64_t start{};
  std::uint64_t stop{};
};

[[nodiscard]] auto shape(const graph::Resource &value) noexcept {
  return std::tuple{value.type,     value.integer_bits,  value.fraction_bits,
                    value.rounding, value.overflow,      value.approximation,
                    value.elements, value.element_bytes, value.bytes};
}

[[nodiscard]] bool covers_value(const graph::Resource &value,
                                const graph::Access &access) noexcept {
  return access.offset_bytes == 0u && access.size_bytes == value.bytes &&
         access.element_bytes == value.element_bytes &&
         access.element_count == value.elements &&
         access.stride_bytes == value.element_bytes;
}

[[nodiscard]] bool count_value(const graph::Resource &value) noexcept {
  return value.elements == 1u &&
         (value.type == graph::Value::U32 || value.type == graph::Value::U64);
}

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

Status plan_memory(graph::Info &info, const std::span<const MemoryNode> nodes,
                   const std::uint64_t page_bytes) noexcept {
  try {
    if (nodes.size() != info.nodes.size() || page_bytes < Alignment ||
        page_bytes % Alignment != 0u) {
      return Status::fail(Reason::GraphInvalid);
    }
    const auto add = [](std::uint64_t &total,
                        const std::uint64_t value) noexcept {
      return kernel::checked::add(total, value, total);
    };
    info.memory = {};
    std::vector<Lifetime> lifetimes(info.resources.size());
    std::vector<std::size_t> counts(info.resources.size());
    for (std::size_t index = 0u; index < info.resources.size(); ++index) {
      graph::Resource &value = info.resources[index];
      std::uint64_t bytes = 0u;
      if (value.id != index + 1u || value.element_bytes == 0u ||
          !kernel::checked::mul(value.elements, value.element_bytes, bytes) ||
          value.bytes != bytes) {
        return Status::fail(Reason::GraphInvalid);
      }
      if (value.active != 0u &&
          (value.active > info.resources.size() ||
           !count_value(info.resources[value.active - 1u]))) {
        return Status::fail(Reason::GraphInvalid);
      }
      if (value.parent != 0u &&
          (!count_value(value) || value.parent > info.resources.size() ||
           !count_value(info.resources[value.parent - 1u]))) {
        return Status::fail(Reason::GraphInvalid);
      }
      value.alias_group = value.id;
      value.alias_offset_bytes = 0u;
      value.source = 0u;
      value.reset_node = resource::NoNode;
    }
    std::vector<std::uint8_t> marks(info.resources.size());
    for (const graph::Resource &value : info.resources) {
      std::uint32_t cursor = value.id;
      while (cursor != 0u && marks[cursor - 1u] == 0u) {
        marks[cursor - 1u] = 1u;
        cursor = info.resources[cursor - 1u].parent;
      }
      if (cursor != 0u && marks[cursor - 1u] == 1u) {
        return Status::fail(Reason::GraphInvalid);
      }
      cursor = value.id;
      while (cursor != 0u && marks[cursor - 1u] == 1u) {
        marks[cursor - 1u] = 2u;
        cursor = info.resources[cursor - 1u].parent;
      }
    }
    const auto valid_domain = [&](const Domain &domain) {
      return (domain.count == 0u ||
              (domain.count <= info.resources.size() &&
               count_value(info.resources[domain.count - 1u]))) &&
             (domain.predicate == 0u ||
              (domain.predicate <= info.resources.size() &&
               count_value(info.resources[domain.predicate - 1u]))) &&
             (domain.predicate != 0u || domain.expected == 0u);
    };
    if (std::any_of(nodes.begin(), nodes.end(), [&](const MemoryNode &node) {
          return !valid_domain(node.domain);
        })) {
      return Status::fail(Reason::GraphInvalid);
    }

    for (std::size_t node_index = 0u; node_index < info.nodes.size();
         ++node_index) {
      const graph::Node &node = info.nodes[node_index];
      if (node.index != node_index) {
        return Status::fail(Reason::GraphInvalid);
      }
      for (const graph::Access &access : node.accesses) {
        if (access.resource == 0u || access.resource > lifetimes.size() ||
            (access.mode != resource::AccessMode::Read &&
             access.mode != resource::AccessMode::Write)) {
          return Status::fail(Reason::GraphInvalid);
        }
        std::size_t &count = counts[access.resource - 1u];
        if (count == std::numeric_limits<std::size_t>::max()) {
          return Status::fail(Reason::GraphCapacity);
        }
        ++count;
        Lifetime &lifetime = lifetimes[access.resource - 1u];
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
              access.mode == resource::AccessMode::Write &&
              covers_value(info.resources[access.resource - 1u], access);
          lifetime.first_write_complete =
              lifetime.first_write_dense &&
              nodes[node.index].write == Write::Full;
          lifetime.domain = nodes[node.index].domain;
          if (lifetime.domain.empty() &&
              info.resources[access.resource - 1u].active != 0u) {
            lifetime.domain = Domain{
                .count = info.resources[access.resource - 1u].active,
            };
          }
          lifetime.first_write_domain =
              lifetime.first_write_dense && !lifetime.domain.empty() &&
              nodes[node.index].write == Write::Domain;
        } else if (lifetime.first_write == node.index) {
          const bool dense =
              access.mode == resource::AccessMode::Write &&
              covers_value(info.resources[access.resource - 1u], access);
          lifetime.first_write_dense = lifetime.first_write_dense && dense;
          lifetime.first_write_complete =
              lifetime.first_write_complete && dense &&
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

    std::vector<std::size_t> offsets(info.resources.size() + 1u);
    for (std::size_t index = 0u; index < counts.size(); ++index) {
      if (!size::add(offsets[index], counts[index], offsets[index + 1u])) {
        return Status::fail(Reason::GraphCapacity);
      }
    }
    std::vector<std::uint32_t> uses(offsets.back());
    std::copy(offsets.begin(), offsets.end() - 1u, counts.begin());
    for (const graph::Node &node : info.nodes) {
      for (const graph::Access &access : node.accesses) {
        uses[counts[access.resource - 1u]++] = node.index;
      }
    }

    std::vector<Live> live(info.nodes.size());
    std::vector<std::uint32_t> arena;
    arena.reserve(info.resources.size());
    for (std::size_t index = 0u; index < info.resources.size(); ++index) {
      graph::Resource &value = info.resources[index];
      const Lifetime lifetime = lifetimes[index];
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
      if (!add(live[lifetime.first].start, value.bytes) ||
          !add(live[lifetime.last].stop, value.bytes)) {
        return Status::fail(Reason::GraphCapacity);
      }
      if (lifetime.first_write != lifetime.first ||
          !lifetime.first_write_dense ||
          lifetime.first_read == lifetime.first) {
        return Status::fail(Reason::GraphInvalid);
      }
      bool reusable = lifetime.first_write_complete;
      if (!reusable && lifetime.first_write_domain) {
        reusable = true;
        for (std::size_t cursor = offsets[index]; cursor < offsets[index + 1u];
             ++cursor) {
          const std::uint32_t node = uses[cursor];
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
      arena.push_back(value.id);
    }

    const auto stored = [&](const std::uint32_t id) noexcept {
      const std::size_t index = id - 1u;
      return info.resources[index].visibility == graph::Visibility::Internal &&
             lifetimes[index].first != resource::NoNode;
    };
    for (const graph::Node &node : info.nodes) {
      if (node.operation != graph::Operation::Map ||
          nodes[node.index].write != Write::Full ||
          nodes[node.index].inplace != Inplace::Pointwise) {
        continue;
      }
      std::uint32_t output = 0u;
      for (const graph::Access &access : node.accesses) {
        if (access.mode != resource::AccessMode::Write ||
            !stored(access.resource) ||
            lifetimes[access.resource - 1u].first != node.index ||
            !covers_value(info.resources[access.resource - 1u], access)) {
          continue;
        }
        if (output != 0u) {
          output = 0u;
          break;
        }
        output = access.resource;
      }
      if (output == 0u) {
        continue;
      }
      const graph::Resource &target = info.resources[output - 1u];
      std::uint32_t source = 0u;
      for (const graph::Access &access : node.accesses) {
        if (access.mode != resource::AccessMode::Read ||
            !stored(access.resource) ||
            lifetimes[access.resource - 1u].first >= node.index ||
            lifetimes[access.resource - 1u].last != node.index ||
            !covers_value(info.resources[access.resource - 1u], access)) {
          continue;
        }
        const graph::Resource &candidate = info.resources[access.resource - 1u];
        if (shape(candidate) != shape(target)) {
          continue;
        }
        source =
            source == 0u ? access.resource : std::min(source, access.resource);
      }
      if (source != 0u) {
        info.resources[output - 1u].source = source;
      }
    }

    std::uint64_t live_bytes = 0u;
    for (std::size_t node = 0u; node < info.nodes.size(); ++node) {
      if (node != 0u) {
        if (live[node - 1u].stop > live_bytes) {
          return Status::fail(Reason::GraphInvalid);
        }
        live_bytes -= live[node - 1u].stop;
      }
      if (!kernel::checked::add(live_bytes, live[node].start, live_bytes)) {
        return Status::fail(Reason::GraphCapacity);
      }
      info.memory.live_bytes = std::max(info.memory.live_bytes, live_bytes);
    }

    if (!arena.empty()) {
      std::sort(arena.begin(), arena.end(),
                [&](const std::uint32_t left, const std::uint32_t right) {
                  const Lifetime left_lifetime = lifetimes[left - 1u];
                  const Lifetime right_lifetime = lifetimes[right - 1u];
                  return std::tie(left_lifetime.first, left) <
                         std::tie(right_lifetime.first, right);
                });
      Layout destructive;
      Layout ordinary;
      if (!pack(info.resources, lifetimes, arena, page_bytes, true,
                destructive) ||
          !pack(info.resources, lifetimes, arena, page_bytes, false,
                ordinary)) {
        return Status::fail(Reason::GraphCapacity);
      }
      const bool destructive_layout =
          std::tie(destructive.bytes, destructive.count) <=
          std::tie(ordinary.bytes, ordinary.count);
      const Layout &layout = destructive_layout ? destructive : ordinary;
      for (const std::uint32_t id : arena) {
        graph::Resource &value = info.resources[id - 1u];
        if (layout.owners[id - 1u] == 0u) {
          return Status::fail(Reason::GraphInvalid);
        }
        value.alias_group = layout.owners[id - 1u];
        value.alias_offset_bytes = layout.offsets[id - 1u];
        if (!destructive_layout) {
          value.source = 0u;
        }
      }
      if (!add(info.memory.physical_bytes, layout.bytes) ||
          !add(info.memory.allocation_count, layout.count)) {
        return Status::fail(Reason::GraphCapacity);
      }
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
}

} // namespace rund::compute::detail::resource_detail
