#include "../projection/internal.hpp"

#include <bit>

namespace rund::compute::detail::graph_reduce {

std::uint64_t graph_identity(const VirtualRunProjection &run) noexcept {
  return run.result_identity_hi ^ std::rotl(run.result_identity_lo, 1);
}

const residency::GraphMaterialization *
graph_materialization(const VirtualRunProjection &run,
                      const std::uint32_t resource) noexcept {
  for (std::size_t index = 0u; index < run.graph_resource_count; ++index) {
    if (run.graph_resources[index] == resource) {
      return &run.graph_materializations[index];
    }
  }
  return nullptr;
}

residency::GraphPersistIdentity
persist_identity(const VirtualRunProjection &run,
                 const residency::TiledGraphPlan &graph,
                 const std::size_t stage, const Ticket &ticket,
                 const residency::Identity plan) noexcept {
  residency::GraphPersistIdentity id{};
  if (stage >= graph.stages().size() ||
      run.input_count > id.input_backings.size() || ticket.count == 0u ||
      plan == residency::Identity{}) {
    return id;
  }
  id.plan = plan;
  id.topology_hi = run.result_identity_hi;
  id.topology_lo = run.result_identity_lo;
  id.stage = stage;
  id.operation = run.operation;
  id.frame_capacity = run.frame_capacity;
  id.page_count = run.active.graph.page_count();
  id.batch_count = run.active.graph.batch_count();
  id.input_page_bytes = run.input_page_bytes;
  id.output_page_bytes = run.output_page_bytes;
  id.input_payload_bytes = run.input_payload_bytes;
  id.output_payload_bytes = run.output_payload_bytes;
  id.input_frame_elements = run.input_frame_elements;
  id.input_count = static_cast<std::uint32_t>(run.input_count);
  id.output_backing = run.output_backing;
  id.output_version = run.output_version;
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    id.input_backings[index] = run.inputs[index].backing;
    id.input_versions[index] = run.inputs[index].version;
  }
  const auto &ports = graph.stages()[stage].ports;
  if (ports.empty()) {
    return {};
  }
  id.resource = ports.back().resource;
  for (std::size_t index = 0u; index < ports.size(); ++index) {
    if (ports[index].resource == id.resource &&
        ports[index].access == residency::Access::Write) {
      id.port = index;
      break;
    }
  }
  return id;
}

} // namespace rund::compute::detail::graph_reduce
