#include "local.hpp"

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace rund::compute::detail::resource_detail::memory_detail {
namespace {

[[nodiscard]] auto shape(const graph::Resource &value) noexcept {
  return std::tuple{value.type,     value.integer_bits,  value.fraction_bits,
                    value.rounding, value.overflow,      value.approximation,
                    value.elements, value.element_bytes, value.bytes};
}

} // namespace

void identify_sources(graph::Info &info,
                      const std::span<const MemoryNode> nodes,
                      const Work &work) noexcept {
  const auto stored = [&](const std::uint32_t id) noexcept {
    const std::size_t index = id - 1u;
    return info.resources[index].visibility == graph::Visibility::Internal &&
           work.lifetimes[index].first != resource::NoNode;
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
          work.lifetimes[access.resource - 1u].first != node.index ||
          !whole_value(info.resources[access.resource - 1u], access)) {
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
          work.lifetimes[access.resource - 1u].first >= node.index ||
          work.lifetimes[access.resource - 1u].last != node.index ||
          !whole_value(info.resources[access.resource - 1u], access)) {
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
}

} // namespace rund::compute::detail::resource_detail::memory_detail
