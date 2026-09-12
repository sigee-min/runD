#pragma once

#include "../../registry.hpp"
#include "../../registry/graph_promote_owner.hpp"
#include "../graph_promote.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail::residency::graph_promote_detail {

struct Projection final {
  // A group has one invocation/batch/stage, regardless of its source count.
  std::array<PageUse, TiledGraphPortCapacity * execution::GraphForecastCapacity>
      uses{};
  Epoch epoch{};
  std::size_t page_count{};
};

struct Group final {
  std::shared_ptr<const ResidencyPlan> owner;
  Projection projection{};
  std::array<std::size_t, execution::GraphPromoteSourceCapacity> first_uses{};
  std::array<std::uint32_t, execution::GraphPromoteSourceCapacity> resources{};
  std::array<std::span<const execution::GraphForecastPage>,
             execution::GraphPromoteSourceCapacity>
      pages{};
  std::array<std::uint64_t, execution::GraphPromoteSourceCapacity> tokens{};
  std::size_t source_count{};
};

[[nodiscard]] inline bool
project_group(const std::shared_ptr<const ResidencyPlan> &owner,
              const TiledGraphInvocation &invocation, const std::uint64_t batch,
              const std::size_t stage, Projection &result) noexcept {
  result.epoch = {};
  result.page_count = 0u;
  if (owner == nullptr || !owner->graph_tiled() || !owner->identity() ||
      !invocation.owned_by(owner->tiled_graph())) {
    return false;
  }
  const TiledGraphPlan &plan = owner->tiled_graph();
  PageRun run{};
  if (stage >= plan.stages().size() || !invocation.batch(batch, run) ||
      run.page_count == 0u ||
      run.page_count > execution::GraphForecastCapacity) {
    return false;
  }
  const auto &ports = plan.stages()[stage].ports;
  if (ports.empty() || ports.size() > TiledGraphPortCapacity ||
      run.page_count > result.uses.size() / ports.size()) {
    return false;
  }
  const std::size_t page_count = static_cast<std::size_t>(run.page_count);
  if (!invocation.project(
          batch, stage,
          std::span<PageUse>{result.uses.data(), page_count * ports.size()},
          result.epoch)) {
    return false;
  }
  result.page_count = page_count;
  return true;
}

template <class Source>
[[nodiscard]] inline bool
validate_source(const std::shared_ptr<const ResidencyPlan> &owner,
                const Source &forecast, const std::size_t stage,
                const std::uint32_t resource, const Projection &projection,
                std::size_t &first_use) noexcept {
  first_use = 0u;
  if (!forecast || forecast.plan() != owner->identity() ||
      forecast.coordinate() != projection.epoch.ordinal ||
      forecast.pages().empty() ||
      forecast.pages().size() > projection.page_count) {
    return false;
  }
  const TiledGraphPlan &plan = owner->tiled_graph();
  const TiledGraphResource *const declared = plan.resource(resource);
  const auto &ports = plan.stages()[stage].ports;
  if (declared == nullptr ||
      declared->kind != GraphResourceKind::ExternalInput ||
      declared->persistence != ResourcePersistence::Backing) {
    return false;
  }
  const auto reads_resource = [resource](const TiledGraphPort port) {
    return port.resource == resource && port.access == Access::Read;
  };
  const auto port = std::find_if(ports.begin(), ports.end(), reads_resource);
  if (port == ports.end() || std::find_if(std::next(port), ports.end(),
                                          reads_resource) != ports.end()) {
    return false;
  }
  first_use =
      static_cast<std::size_t>(port - ports.begin()) * projection.page_count;
  std::size_t source_index = 0u;
  for (std::size_t index = 0u; index < projection.page_count; ++index) {
    const PageUse &use = projection.uses[first_use + index];
    if (use.access != Access::Read) {
      return false;
    }
    if (source_index < forecast.pages().size() &&
        use.key == forecast.pages()[source_index].use) {
      ++source_index;
    }
  }
  return source_index == forecast.pages().size();
}

} // namespace rund::compute::detail::residency::graph_promote_detail
