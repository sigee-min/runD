#include "../memory.hpp"

#include "local.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::compute::detail::resource_detail::memory_detail {

bool whole_value(const graph::Resource &value,
                 const graph::Access &access) noexcept {
  return access.offset_bytes == 0u && access.size_bytes == value.bytes &&
         access.element_bytes == value.element_bytes &&
         access.element_count == value.elements &&
         access.stride_bytes == value.element_bytes;
}

bool count_value(const graph::Resource &value) noexcept {
  return value.elements == 1u &&
         (value.type == graph::Value::U32 || value.type == graph::Value::U64);
}

Status validate(graph::Info &info, const std::span<const MemoryNode> nodes,
                const std::uint64_t page_bytes, Work &work) {
  if (nodes.size() != info.nodes.size() || page_bytes < Alignment ||
      page_bytes % Alignment != 0u) {
    return Status::fail(Reason::GraphInvalid);
  }

  info.memory = {};
  work.lifetimes.resize(info.resources.size());
  work.counts.resize(info.resources.size());
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
  return Status::success();
}

} // namespace rund::compute::detail::resource_detail::memory_detail
