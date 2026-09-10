#include "../model.hpp"

namespace rund::compute::detail::graph_detail::describe_detail {

void append_access(graph::Info &info, graph::Node &node,
                   const std::uint32_t resource,
                   const resource::AccessMode mode) {
  const graph::Resource &value = info.resources[resource - 1u];
  node.accesses.push_back(graph::Access{.resource = resource,
                                        .mode = mode,
                                        .offset_bytes = 0u,
                                        .size_bytes = value.bytes,
                                        .element_bytes = value.element_bytes,
                                        .element_count = value.elements,
                                        .stride_bytes = value.element_bytes});
}


} // namespace rund::compute::detail::graph_detail::describe_detail
