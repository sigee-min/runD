#pragma once

#include "../graph_forecast.hpp"
#include "../../registry.hpp"
#include "../../registry/graph_forecast_owner.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>

namespace rund::compute::detail::residency::graph_forecast_detail {

struct Projection final {
  std::array<PageUse, TiledGraphPortCapacity * execution::GraphForecastCapacity>
      uses{};
  Epoch epoch{};
  const TiledGraphResource *resource{};
  std::size_t use_count{};
};

[[nodiscard]] inline bool
validate_projection(const std::shared_ptr<const ResidencyPlan> &owner,
                    const TiledGraphInvocation &invocation,
                    const std::uint64_t batch, const std::size_t stage,
                    const std::uint32_t resource,
                    const std::span<const PageUse> selected,
                    const GraphMaterialization materialization,
                    const FrameRegion region, Projection &result) noexcept {
  result = {};
  if (owner == nullptr || !owner->graph_tiled() || !owner->identity() ||
      !invocation.owned_by(owner->tiled_graph()) || selected.empty() ||
      selected.size() > execution::GraphForecastCapacity ||
      region.tier != FrameTier::Host || region.role != FrameRole::Input ||
      region.count == 0u || selected.size() > region.count ||
      materialization.resource != resource ||
      materialization.key.backing == 0u || materialization.key.page != 0u ||
      materialization.key.domain != CacheDomain::Backing ||
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
      declared->kind != GraphResourceKind::ExternalInput ||
      declared->persistence != ResourcePersistence::Backing ||
      declared->page_bytes != materialization.page_bytes ||
      !invocation.batch(batch, run) || run.page_count == 0u ||
      run.page_count > execution::GraphForecastCapacity) {
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
                                          row.access == Access::Read;
                                 });
  if (port == node.ports.end() ||
      std::find_if(std::next(port), node.ports.end(),
                   [resource](const TiledGraphPort row) {
                     return row.resource == resource &&
                            row.access == Access::Read;
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
  std::size_t cursor = 0u;
  for (const PageUse &supplied : selected) {
    while (cursor < exact.size() && exact[cursor] != supplied) {
      ++cursor;
    }
    if (cursor == exact.size()) {
      return false;
    }
    ++cursor;
  }
  result.resource = declared;
  return true;
}

} // namespace rund::compute::detail::residency::graph_forecast_detail
