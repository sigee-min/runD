#include "local.hpp"

namespace rund_node_graph_services::memory_detail {

Resource ResourceOf(const std::uint32_t id, const Value type,
                    const std::uint64_t elements,
                    const std::uint64_t width) noexcept {
  return Resource{.id = id,
                  .type = type,
                  .visibility = Visibility::Internal,
                  .elements = elements,
                  .element_bytes = width,
                  .bytes = elements * width};
}

Access AccessOf(const Resource &value, const AccessMode mode) noexcept {
  return Access{.resource = value.id,
                .mode = mode,
                .size_bytes = value.bytes,
                .element_bytes = value.element_bytes,
                .element_count = value.elements,
                .stride_bytes = value.element_bytes};
}

bool Overlaps(const Resource &left, const Resource &right) noexcept {
  return left.alias_group == right.alias_group &&
         left.alias_offset_bytes < right.alias_offset_bytes + right.bytes &&
         right.alias_offset_bytes < left.alias_offset_bytes + left.bytes;
}

bool SamePlan(const Info &left, const Info &right) noexcept {
  if (left.memory != right.memory ||
      left.resources.size() != right.resources.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < left.resources.size(); ++index) {
    const Resource &first = left.resources[index];
    const Resource &second = right.resources[index];
    if (first.active != second.active || first.parent != second.parent ||
        first.source != second.source ||
        first.alias_group != second.alias_group ||
        first.alias_offset_bytes != second.alias_offset_bytes ||
        first.reset_node != second.reset_node) {
      return false;
    }
  }
  return true;
}

rund::compute::Status Plan(Info &info, const std::span<const MemoryNode> nodes,
                           const std::uint64_t page) noexcept {
  return rund::compute::detail::resource_detail::plan_memory(info, nodes, page);
}

} // namespace rund_node_graph_services::memory_detail
