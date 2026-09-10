#pragma once

#include "../../model.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::compute::detail::residency {

struct GraphPlanningState final {
  const TiledGraphPlanInput &input;
  std::uint64_t frames{};
  std::vector<TiledGraphResource> resources{};
  std::vector<TiledGraphStage> stages{};
  std::vector<TiledGraphPhysicalClass> physical_classes{};
  std::uint64_t page_bytes{};
};

[[nodiscard]] const TiledGraphResource *
find_graph_resource(const GraphPlanningState &, std::uint32_t resource) noexcept;

[[nodiscard]] std::size_t
index_graph_resource(const GraphPlanningState &, std::uint32_t resource) noexcept;

[[nodiscard]] Failure initialize_graph(GraphPlanningState &);
[[nodiscard]] Failure freeze_graph_liveness(GraphPlanningState &);
[[nodiscard]] Failure assign_graph_physical_classes(GraphPlanningState &);
[[nodiscard]] Failure seal_graph_dependencies(GraphPlanningState &);

} // namespace rund::compute::detail::residency
