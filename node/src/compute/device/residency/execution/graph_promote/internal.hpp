#pragma once

#include "../graph_promote.hpp"
#include "../../registry.hpp"
#include "../../registry/graph_promote_owner.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail::residency::graph_promote_detail {

struct Projection final {
  // One projection belongs to one source Forecast.  Aggregate capacity is
  // owned by Group/ticket storage; multiplying it here would duplicate the
  // second source's page budget in every projection.
  std::array<PageUse, TiledGraphPortCapacity * execution::GraphForecastCapacity>
      uses{};
  Epoch epoch{};
  std::size_t use_count{};
  std::size_t first_use{};
  std::size_t page_count{};
};

struct Group final {
  std::shared_ptr<const ResidencyPlan> owner;
  std::array<Projection, execution::GraphPromoteSourceCapacity> projections{};
  std::array<std::uint32_t, execution::GraphPromoteSourceCapacity> resources{};
  std::array<std::span<const execution::GraphForecastPage>,
             execution::GraphPromoteSourceCapacity>
      pages{};
  std::array<std::uint64_t, execution::GraphPromoteSourceCapacity> tokens{};
  std::size_t source_count{};
};

template <class Source>
[[nodiscard]] inline bool
validate_projection(const std::shared_ptr<const ResidencyPlan> &owner,
                    const Source &forecast,
                    const TiledGraphInvocation &invocation,
                    const std::uint64_t batch, const std::size_t stage,
                    const std::uint32_t resource, Projection &result) noexcept {
  result = {};
  if (!forecast || owner == nullptr || !owner->graph_tiled() ||
      !owner->identity() || forecast.plan() != owner->identity() ||
      !invocation.owned_by(owner->tiled_graph())) {
    return false;
  }
  const TiledGraphPlan &plan = owner->tiled_graph();
  PageRun run{};
  if (stage >= plan.stages().size() || !invocation.batch(batch, run) ||
      run.page_count == 0u ||
      run.page_count > execution::GraphForecastCapacity ||
      forecast.pages().empty() ||
      forecast.pages().size() > static_cast<std::size_t>(run.page_count)) {
    return false;
  }
  const TiledGraphResource *const declared = plan.resource(resource);
  const TiledGraphStage &node = plan.stages()[stage];
  if (declared == nullptr ||
      declared->kind != GraphResourceKind::ExternalInput ||
      declared->persistence != ResourcePersistence::Backing ||
      node.ports.empty() || node.ports.size() > TiledGraphPortCapacity ||
      run.page_count > result.uses.size() / node.ports.size()) {
    return false;
  }
  const auto port = std::find_if(node.ports.begin(), node.ports.end(),
                                 [resource](const TiledGraphPort value) {
                                   return value.resource == resource &&
                                          value.access == Access::Read;
                                 });
  if (port == node.ports.end() ||
      std::find_if(std::next(port), node.ports.end(),
                   [resource](const TiledGraphPort value) {
                     return value.resource == resource &&
                            value.access == Access::Read;
                   }) != node.ports.end()) {
    return false;
  }
  result.use_count =
      static_cast<std::size_t>(run.page_count) * node.ports.size();
  if (!invocation.project(
          batch, stage,
          std::span<PageUse>{result.uses.data(), result.use_count},
          result.epoch) ||
      result.epoch.ordinal != forecast.coordinate()) {
    return false;
  }
  result.first_use = static_cast<std::size_t>(port - node.ports.begin()) *
                     static_cast<std::size_t>(run.page_count);
  result.page_count = static_cast<std::size_t>(run.page_count);
  std::size_t source_index = 0u;
  for (std::size_t index = 0u; index < result.page_count; ++index) {
    if (result.uses[result.first_use + index].access != Access::Read) {
      return false;
    }
    if (source_index < forecast.pages().size() &&
        result.uses[result.first_use + index].key ==
            forecast.pages()[source_index].use) {
      ++source_index;
    }
  }
  return source_index == forecast.pages().size();
}

} // namespace rund::compute::detail::residency::graph_promote_detail
