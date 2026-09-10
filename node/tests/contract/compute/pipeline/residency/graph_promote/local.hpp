#pragma once

#include "../local.hpp"

#include "src/compute/device/residency/execution/graph_promote.hpp"
#include "src/compute/device/residency/registry/graph_forecast_owner.hpp"
#include "src/compute/device/residency/registry/graph_promote_owner.hpp"
#include "src/compute/device/residency/registry.hpp"
#include "src/compute/pipeline/residency/planner.hpp"

#include <cstddef>
#include <span>

namespace rund_node_test_pipeline_residency::graph_promote {

namespace residency = rund::compute::detail::residency;
namespace execution = rund::compute::detail::residency::execution;

[[nodiscard]] bool terminal_forecast(residency::Authority &,
                                     execution::GraphForecast &);
[[nodiscard]] bool terminal_promote(residency::Authority &,
                                    execution::GraphPromote &, bool);

[[nodiscard]] int CheckWide();
[[nodiscard]] int CheckGroup();
[[nodiscard]] int CheckSingle();

} // namespace rund_node_test_pipeline_residency::graph_promote
