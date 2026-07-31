#include "local.hpp"

#include "../../fixed/format.hpp"
#include "../../type.hpp"

#include <accel/graph/factory.hpp>

#include <limits>
#include <new>

namespace rund::compute::detail::graph_compile {

std::uint64_t logical(const ProgramState &program,
                      const std::uint32_t value) noexcept {
  return value == 0u || value > program.graph_value_routes.size() ? 0u : value;
}

namespace {

[[nodiscard]] kernel::BufferInit init(const Lowering &lowering,
                                      const std::size_t node,
                                      const std::uint32_t value,
                                      const kernel::BufferRole role) noexcept {
  if (role != kernel::BufferRole::Write || lowering.layout == nullptr ||
      value == 0u || value > lowering.layout->resources.size()) {
    return kernel::BufferInit::Preserve;
  }
  const graph::Resource &resource = lowering.layout->resources[value - 1u];
  return resource.requires_reset() && resource.reset_node == node
             ? kernel::BufferInit::Zero
             : kernel::BufferInit::Preserve;
}

} // namespace

std::optional<rund::AccelGraphBufferRef> resident(const Lowering &lowering,
                                                  const std::size_t node,
                                                  const std::uint32_t value,
                                                  const kernel::BufferRole role,
                                                  const char *const name) {
  if (lowering.program == nullptr || lowering.graph == nullptr || value == 0u ||
      value > lowering.graph->values.size() ||
      value > lowering.program->graph_value_routes.size()) {
    return std::nullopt;
  }
  const GraphValueRoute route =
      lowering.program->graph_value_routes[value - 1u];
  const rund::GraphBufferVisibility visibility =
      route.source == GraphBindSource::Internal
          ? rund::GraphBufferVisibility::Internal
          : rund::GraphBufferVisibility::External;
  const GraphValue &stored = lowering.graph->values[value - 1u];
  const rund::AccelBufferDesc shape{
      .scalar_width_bytes = type_bytes(stored.type),
      .count = stored.count,
      .usage = role == kernel::BufferRole::Read ? rund::BufferUsage::ReadOnly
                                                : rund::BufferUsage::WriteOnly,
  };
  return role == kernel::BufferRole::Read
             ? rund::AccelRead(shape, name, visibility, value)
             : rund::AccelWrite(shape, name, visibility, value,
                                init(lowering, node, value, role));
}

Status bind(Lowering &lowering, const std::size_t index) {
  if (lowering.layout == nullptr || index >= lowering.layout->nodes.size() ||
      lowering.layout->nodes[index].index != index) {
    return Status::fail(Reason::GraphInvalid);
  }
  const graph::Node &node = lowering.layout->nodes[index];
  const std::size_t begin = lowering.program->graph_bindings.size();
  try {
    if (lowering.cpu()) {
      lowering.cpu_refs.emplace_back();
      lowering.cpu_refs.back().reserve(node.accesses.size());
    }
    for (const graph::Access &access : node.accesses) {
      if (access.resource == 0u ||
          access.resource > lowering.graph->values.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      const kernel::BufferRole role = access.mode == resource::AccessMode::Read
                                          ? kernel::BufferRole::Read
                                          : kernel::BufferRole::Write;
      lowering.program->graph_bindings.push_back(GraphRunBinding{
          .value_index = access.resource - 1u,
          .role = role,
      });
      if (lowering.cpu()) {
        lowering.cpu_refs.back().push_back(kernel::GraphBufferRef{
            .logical_id = logical(*lowering.program, access.resource),
            .role = role,
            .init = init(lowering, index, access.resource, role),
        });
      }
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  if (lowering.cpu()) {
    lowering.program->cpu_graph->bind_begin[index] = begin;
    lowering.program->cpu_graph->bind_count[index] =
        lowering.program->graph_bindings.size() - begin;
  }
  return Status::success();
}

Status lower(Lowering &lowering) {
  if (lowering.graph == nullptr || lowering.layout == nullptr ||
      lowering.program == nullptr ||
      lowering.layout->nodes.size() != lowering.graph->steps.size()) {
    return Status::fail(Reason::GraphInvalid);
  }
  try {
    if (lowering.cpu()) {
      lowering.cpu_refs.reserve(lowering.graph->steps.size());
      lowering.cpu_nodes.reserve(lowering.graph->steps.size());
    } else {
      lowering.accel_refs.reserve(lowering.graph->steps.size());
      lowering.accel_nodes.reserve(lowering.graph->steps.size());
      lowering.barriers.assign(lowering.graph->steps.size(), 0u);
      for (const graph::Barrier &barrier : lowering.layout->barriers) {
        if (barrier.before_resource == barrier.after_resource) {
          continue;
        }
        if (barrier.after_node >= lowering.barriers.size()) {
          return Status::fail(Reason::GraphInvalid);
        }
        lowering.barriers[barrier.after_node] = 1u;
      }
    }
    for (std::size_t index = 0u; index < lowering.graph->steps.size();
         ++index) {
      const Status bound = bind(lowering, index);
      if (!bound) {
        return bound;
      }
      const GraphStep &step = lowering.graph->steps[index];
      const Status lowered =
          std::holds_alternative<MapStep>(step)
              ? map(lowering, index, std::get<MapStep>(step))
          : std::holds_alternative<ScanStep>(step)
              ? scan(lowering, index, std::get<ScanStep>(step))
              : primitive(lowering, index, std::get<GraphPrimitive>(step));
      if (!lowered) {
        return lowered;
      }
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  } catch (const std::out_of_range &) {
    return Status::fail(Reason::GraphBindingInvalid);
  }
  if (lowering.operation != lowering.operations.size()) {
    return Status::fail(Reason::GraphOperationInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_compile
