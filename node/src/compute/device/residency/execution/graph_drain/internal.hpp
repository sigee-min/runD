#pragma once

#include "../graph_drain.hpp"
#include "../../registry.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>

namespace rund::compute::detail::residency::graph_drain_detail {

struct Projection final {
  std::array<PageUse, TiledGraphPortCapacity * execution::GraphDrainCapacity>
      uses{};
  Epoch epoch{};
  std::size_t use_count{};
};

[[nodiscard]] inline bool validate_projection(
    const std::shared_ptr<const ResidencyPlan> &owner,
    const TiledGraphInvocation &invocation, const std::uint64_t batch,
    const std::size_t stage, const std::uint32_t resource,
    const std::span<const PageUse> selected,
    const GraphMaterialization materialization, const FrameRegion source_region,
    const FrameRegion target_region, Projection &result) noexcept {
  result = {};
  if (owner == nullptr || !owner->graph_tiled() || !owner->identity() ||
      !invocation.owned_by(owner->tiled_graph()) || selected.empty() ||
      selected.size() > execution::GraphDrainCapacity ||
      source_region.tier != FrameTier::Device ||
      source_region.role != FrameRole::Output || source_region.count == 0u ||
      target_region.tier != FrameTier::Host ||
      target_region.role != FrameRole::Output || target_region.count == 0u ||
      selected.size() > source_region.count ||
      selected.size() > target_region.count ||
      materialization.resource != resource ||
      materialization.key.backing == 0u || materialization.key.page != 0u ||
      materialization.page_count != invocation.page_count()) {
    return false;
  }
  const TiledGraphPlan &plan = owner->tiled_graph();
  if (stage >= plan.stages().size()) {
    return false;
  }
  const TiledGraphResource *const declared = plan.resource(resource);
  PageRun run{};
  if (declared == nullptr ||
      declared->kind != GraphResourceKind::ExternalOutput ||
      ((declared->persistence == ResourcePersistence::Transient) !=
       (materialization.key.domain == CacheDomain::Transient)) ||
      ((declared->persistence == ResourcePersistence::Backing) !=
       (materialization.key.domain == CacheDomain::Backing)) ||
      declared->page_bytes != materialization.page_bytes ||
      !invocation.batch(batch, run) || run.page_count == 0u ||
      run.page_count > execution::GraphDrainCapacity) {
    return false;
  }
  const TiledGraphStage &node = plan.stages()[stage];
  if (node.ports.empty() || node.ports.size() > TiledGraphPortCapacity ||
      run.page_count > result.uses.size() / node.ports.size()) {
    return false;
  }
  const auto port = std::find_if(node.ports.begin(), node.ports.end(),
                                 [resource](const TiledGraphPort row) {
                                   return row.resource == resource &&
                                          row.access == Access::Write;
                                 });
  if (port == node.ports.end() ||
      std::find_if(std::next(port), node.ports.end(),
                   [resource](const TiledGraphPort row) {
                     return row.resource == resource &&
                            row.access == Access::Write;
                   }) != node.ports.end()) {
    return false;
  }
  result.use_count =
      static_cast<std::size_t>(run.page_count) * node.ports.size();
  if (!invocation.project(
          batch, stage,
          std::span<PageUse>{result.uses.data(), result.use_count},
          result.epoch)) {
    return false;
  }
  const std::size_t port_index =
      static_cast<std::size_t>(port - node.ports.begin());
  const std::size_t first =
      port_index * static_cast<std::size_t>(run.page_count);
  const auto exact = std::span<const PageUse>{
      result.uses.data() + static_cast<std::ptrdiff_t>(first),
      static_cast<std::size_t>(run.page_count)};
  return exact.size() == selected.size() &&
         std::equal(exact.begin(), exact.end(), selected.begin());
}

} // namespace rund::compute::detail::residency::graph_drain_detail
