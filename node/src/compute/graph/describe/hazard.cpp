#include "model.hpp"

#include <rund/compute/resource/plan.hpp>

#include <algorithm>
#include <new>
#include <vector>

namespace rund::compute::detail::graph_detail::describe_detail {

Status build_hazards(graph::Info &info) {
  try {
    std::vector<resource::Resource> resources;
    std::vector<resource::Access> accesses;
    resources.reserve(info.resources.size());
    for (const graph::Resource &value : info.resources) {
      resources.push_back(resource::Resource{
          .id = value.id,
          .bytes = value.bytes,
          .alias_group = value.alias_group,
          .alias_offset_bytes = value.alias_offset_bytes,
      });
    }
    for (const graph::Node &node : info.nodes) {
      for (const graph::Access &access : node.accesses) {
        accesses.push_back(resource::Access{
            .node = node.index,
            .resource = access.resource,
            .mode = access.mode,
            .offset_bytes = access.offset_bytes,
            .element_bytes = access.element_bytes,
            .element_count = access.element_count,
            .stride_bytes = access.stride_bytes,
        });
      }
    }

    auto planned = resource::analyze(
        resources, accesses, static_cast<std::uint32_t>(info.nodes.size()));
    if (!planned) {
      return Status::fail(planned.reason());
    }
    for (std::size_t index = 0u; index < planned->lifetimes.size(); ++index) {
      const resource::Lifetime &lifetime = planned->lifetimes[index];
      graph::Resource &value = info.resources[index];
      value.first_use = lifetime.first_use;
      value.last_use = lifetime.last_use;
    }
    for (graph::Node &node : info.nodes) {
      node.dependencies.clear();
    }
    for (const resource::Dependency &dependency : planned->dependencies) {
      info.nodes[dependency.after_node].dependencies.push_back(
          dependency.before_node);
    }
    for (graph::Node &node : info.nodes) {
      std::sort(node.dependencies.begin(), node.dependencies.end());
    }
    std::vector<const resource::Barrier *> boundaries(info.nodes.size(),
                                                      nullptr);
    for (const resource::Barrier &barrier : planned->barriers) {
      if (barrier.after_node >= boundaries.size()) {
        return Status::fail(Reason::ResourceGraphIncomplete);
      }
      const resource::Barrier *&boundary = boundaries[barrier.after_node];
      const bool cross_resource =
          barrier.before_resource != barrier.after_resource;
      const bool boundary_crosses =
          boundary != nullptr &&
          boundary->before_resource != boundary->after_resource;
      if (boundary == nullptr || (!boundary_crosses && cross_resource) ||
          (boundary_crosses == cross_resource &&
           barrier.before_node > boundary->before_node)) {
        boundary = &barrier;
      }
    }
    info.barriers.clear();
    info.barriers.reserve(info.nodes.size());
    for (const resource::Barrier *const boundary : boundaries) {
      if (boundary == nullptr) {
        continue;
      }
      const resource::Barrier &barrier = *boundary;
      info.barriers.push_back(graph::Barrier{
          .alias_group = barrier.alias_group,
          .before_resource = barrier.before_resource,
          .after_resource = barrier.after_resource,
          .offset_bytes = barrier.offset_bytes,
          .size_bytes = barrier.size_bytes,
          .before_offset_bytes = barrier.before_offset_bytes,
          .before_element_bytes = barrier.before_element_bytes,
          .before_element_count = barrier.before_element_count,
          .before_stride_bytes = barrier.before_stride_bytes,
          .after_offset_bytes = barrier.after_offset_bytes,
          .after_element_bytes = barrier.after_element_bytes,
          .after_element_count = barrier.after_element_count,
          .after_stride_bytes = barrier.after_stride_bytes,
          .before_node = barrier.before_node,
          .after_node = barrier.after_node,
          .before = barrier.before,
          .after = barrier.after,
      });
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::ResourceGraphCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_detail::describe_detail
