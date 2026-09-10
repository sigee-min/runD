#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::project(
    PrefetchLane &lane, const std::uint64_t batch,
    const std::size_t input_index, const bool speculative,
    residency::TiledGraphPort &input_port,
    const residency::PoolPhysicalOwner *&input_owner) noexcept {
  lane = {};
  lane.batch = batch;
  lane.stage = 0u;
  lane.input_index = input_index;
  lane.speculative = speculative;
  if (input_index >= input_count_ || input_index >= state_.input_count ||
      inputs_[input_index] == nullptr ||
      !run_.active.graph.batch(batch, lane.pages) ||
      lane.pages.page_count == 0u ||
      lane.pages.page_count > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }

  lane.count = static_cast<std::size_t>(lane.pages.page_count);
  std::array<residency::PageUse,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity>
      projected{};
  const std::size_t port_count = graph_.stages().front().ports.size();
  if (port_count < 2u || port_count > residency::TiledGraphPortCapacity ||
      port_count > projected.size() / lane.count ||
      !run_.active.graph.project(batch, 0u,
                                 std::span<residency::PageUse>{
                                     projected.data(), port_count * lane.count},
                                 lane.epoch) ||
      !project_virtual_epoch(run_, batch, lane.byte_epoch) ||
      lane.byte_epoch.failed_page != lane.pages.first_page ||
      lane.byte_epoch.page_count != lane.pages.page_count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t resource = state_.graph_input_resources[input_index];
  const auto found =
      std::find_if(graph_.stages().front().ports.begin(),
                   graph_.stages().front().ports.end(),
                   [resource](const residency::TiledGraphPort port) {
                     return port.resource == resource &&
                            port.access == residency::Access::Read;
                   });
  if (found == graph_.stages().front().ports.end()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t port_index =
      static_cast<std::size_t>(found - graph_.stages().front().ports.begin());
  std::copy(projected.begin() +
                static_cast<std::ptrdiff_t>(port_index * lane.count),
            projected.begin() +
                static_cast<std::ptrdiff_t>((port_index + 1u) * lane.count),
            lane.sources.begin());

  input_port = *found;
  lane.resource = input_port.resource;
  const residency::GraphMaterialization *const materialization =
      graph_materialization(run_, lane.resource);
  if (materialization == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  lane.materialization = *materialization;
  const residency::TiledGraphResource *const input_resource =
      graph_.resource(input_port.resource);
  input_owner = input_resource == nullptr
                    ? nullptr
                    : pool_.graph_owner(input_resource->physical_id);
  if (input_owner == nullptr || input_owner->arena == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
