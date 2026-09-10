#include "../internal.hpp"

#include "../../../../backing.hpp"
#include "../../../../run/backing.hpp"
#include "../../transfer.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

namespace rund::compute::detail::graph_reduce {
namespace {

[[nodiscard]] bool
same_materialization(const residency::GraphMaterialization &left,
                     const residency::GraphMaterialization &right) noexcept {
  return left.resource == right.resource && left.key == right.key &&
         left.page_bytes == right.page_bytes &&
         left.page_count == right.page_count &&
         left.boundary_extent == right.boundary_extent;
}

} // namespace

Status PrefetchController::project_stage(
    PrefetchLane &lane, const Ticket &ticket, const StageScratch &scratch,
    const WavefrontCoordinate &coordinate, const std::uint32_t resource,
    const residency::PoolPhysicalOwner *&input_owner) noexcept {
  lane = {};
  input_owner = nullptr;
  if (coordinate.batch != ticket.batch || coordinate.stage == 0u ||
      coordinate.stage >= graph_.stages().size() ||
      coordinate.first_page != ticket.pages.first_page ||
      coordinate.page_count != ticket.pages.page_count || ticket.count == 0u ||
      ticket.count > lane.sources.size() ||
      scratch.epoch.ordinal != coordinate.ordinal ||
      scratch.port_count != graph_.stages()[coordinate.stage].ports.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }

  std::size_t selected_port = scratch.port_count;
  std::size_t external_input_count = 0u;
  for (std::size_t port_index = 0u; port_index < scratch.port_count;
       ++port_index) {
    const residency::TiledGraphPort port =
        graph_.stages()[coordinate.stage].ports[port_index];
    const residency::TiledGraphResource *const port_resource =
        graph_.resource(port.resource);
    external_input_count += static_cast<std::size_t>(
        residency::reads(port.access) && port_resource != nullptr &&
        port_resource->kind == residency::GraphResourceKind::ExternalInput &&
        port_resource->persistence == residency::ResourcePersistence::Backing);
    if (port.resource != resource || port.access != residency::Access::Read) {
      continue;
    }
    if (selected_port != scratch.port_count) {
      return Status::fail(Reason::PipelineInvalid);
    }
    selected_port = port_index;
  }
  if (selected_port == scratch.port_count || external_input_count == 0u ||
      external_input_count > residency::execution::GraphPromoteSourceCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }

  const residency::GraphPortRequest &request = scratch.requests[selected_port];
  const residency::TiledGraphResource *const declared =
      graph_.resource(resource);
  const residency::GraphMaterialization *const materialization =
      graph_materialization(run_, resource);
  std::size_t input_index = 0u;
  if (declared == nullptr ||
      declared->kind != residency::GraphResourceKind::ExternalInput ||
      declared->persistence != residency::ResourcePersistence::Backing ||
      materialization == nullptr || !find_input(resource, input_index) ||
      request.materialization.resource != resource ||
      request.first_use > scratch.use_count ||
      request.use_count != ticket.count ||
      request.use_count > scratch.use_count - request.first_use ||
      !same_materialization(request.materialization, *materialization) ||
      !project_virtual_epoch(run_, ticket.batch, lane.byte_epoch) ||
      lane.byte_epoch.failed_page != ticket.pages.first_page ||
      lane.byte_epoch.page_count != ticket.pages.page_count) {
    return Status::fail(Reason::PipelineInvalid);
  }

  lane.pages = ticket.pages;
  lane.epoch = scratch.epoch;
  lane.materialization = request.materialization;
  lane.count = ticket.count;
  lane.batch = ticket.batch;
  lane.stage = coordinate.stage;
  lane.resource = resource;
  lane.input_index = input_index;
  for (std::size_t page = 0u; page < ticket.count; ++page) {
    lane.sources[page] = scratch.uses[request.first_use + page];
  }

  input_owner = pool_.graph_owner(declared->physical_id);
  if (input_owner == nullptr || input_owner->arena == nullptr) {
    lane = {};
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
