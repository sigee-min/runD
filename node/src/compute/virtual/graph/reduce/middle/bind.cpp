#include "internal.hpp"

#include <array>
#include <limits>

namespace rund::compute::detail::graph_reduce {
namespace {

[[nodiscard]] Status check_live(const Ticket &ticket,
                                const residency::GraphPortRequest &request,
                                const LiveResource &live,
                                const std::uint32_t capacity) noexcept {
  return live.dirty && live.count == ticket.count &&
                 live.region.count == capacity && live.region == request.region
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

[[nodiscard]] bool
external_backing_read(const residency::TiledGraphPort &port,
                      const residency::TiledGraphResource &resource) noexcept {
  return port.access == residency::Access::Read &&
         resource.kind == residency::GraphResourceKind::ExternalInput &&
         resource.persistence == residency::ResourcePersistence::Backing;
}

[[nodiscard]] Status supply_terminal_inputs(
    Ticket &ticket, const VirtualRunProjection &run,
    const residency::TiledGraphPlan &graph, const residency::Pool &pool,
    PrefetchController &prefetch, const std::size_t terminal_stage,
    const std::uint32_t capacity, bool &cleanup_failed) noexcept {
  if (terminal_stage > std::numeric_limits<std::uint32_t>::max()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto &ports = graph.stages()[terminal_stage].ports;
  std::array<std::uint32_t, residency::TiledGraphPortCapacity> resources{};
  std::size_t resource_count = 0u;
  for (const residency::TiledGraphPort &port : ports) {
    const residency::TiledGraphResource *const resource =
        graph.resource(port.resource);
    if (resource == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (!external_backing_read(port, *resource)) {
      continue;
    }
    for (std::size_t index = 0u; index < resource_count; ++index) {
      if (resources[index] == port.resource) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    if (resource_count == resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    resources[resource_count++] = port.resource;
  }
  if (resource_count == 0u) {
    return Status::success();
  }

  StageScratch scratch{};
  if (!project_stage_scratch(graph, run, pool, ticket, terminal_stage, capacity,
                             scratch) ||
      scratch.port_count != ports.size() ||
      scratch.epoch.ordinal != ticket.collective_epoch.ordinal) {
    return Status::fail(Reason::PipelineInvalid);
  }
  WavefrontCoordinate coordinate{
      .ordinal = scratch.epoch.ordinal,
      .batch = ticket.batch,
      .first_page = ticket.pages.first_page,
      .page_count = ticket.pages.page_count,
      .stage = static_cast<std::uint32_t>(terminal_stage),
  };
  for (std::size_t index = 0u; index < resource_count; ++index) {
    coordinate.resource = resources[index];
    bool failed_cleanup = false;
    const Status supplied = prefetch.supply_stage(
        ticket, scratch, coordinate, coordinate.resource, failed_cleanup);
    cleanup_failed = failed_cleanup || cleanup_failed;
    ticket.poison = failed_cleanup || ticket.poison;
    if (!supplied) {
      return supplied;
    }
  }
  return Status::success();
}

} // namespace

Status MiddleController::bind_terminal_input(Ticket &ticket) noexcept {
  if (terminal_stage_ >= graph_.stages().size() ||
      ticket.collective_request_count !=
          graph_.stages()[terminal_stage_].ports.size() ||
      ticket.collective_anchor_port >= ticket.collective_request_count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t port_index = 0u;
       port_index < ticket.collective_request_count; ++port_index) {
    const residency::TiledGraphPort &port =
        graph_.stages()[terminal_stage_].ports[port_index];
    if (port.access != residency::Access::Read) {
      continue;
    }
    const residency::TiledGraphResource *const resource =
        graph_.resource(port.resource);
    std::size_t resource_index = 0u;
    if (resource == nullptr ||
        !graph_resource_index(graph_, port.resource, resource_index)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (resource->kind == residency::GraphResourceKind::ExternalInput) {
      continue;
    }
    const LiveResource &live = ticket.live_resources[resource_index];
    const Status checked = check_live(
        ticket, ticket.collective_requests[port_index], live, capacity_);
    if (!checked) {
      return checked;
    }
  }
  if (ticket.collective != nullptr && ticket.collective->device != nullptr &&
      ticket.collective->device->backend == Backend::Cpu) {
    bool cleanup_failed = false;
    const Status supplied =
        supply_terminal_inputs(ticket, run_, graph_, pool_, prefetch_,
                               terminal_stage_, capacity_, cleanup_failed);
    ticket.poison = cleanup_failed || ticket.poison;
    if (!supplied) {
      return supplied;
    }
  }
  ticket.intermediate_region =
      ticket.collective_requests[ticket.collective_anchor_port].region;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
