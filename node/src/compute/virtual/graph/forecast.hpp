#pragma once

#include "../../device/residency/execution/graph_forecast.hpp"
#include "../../device/residency/prefetch.hpp"

namespace rund::compute::detail {

// Authenticate one returned backing-service receipt against the exact typed
// Graph Forecast capability. The capability remains live after this terminal
// observation; its caller releases it only after the Host frame consumer has
// returned.
[[nodiscard]] Status
terminal_graph_prefetch(residency::Authority &,
                        residency::execution::GraphForecast &,
                        const residency::PrefetchReceipt &) noexcept;

// Close an issued capability when no backing callback was dispatched.
[[nodiscard]] bool reject_graph_prefetch(residency::Authority &,
                                         residency::execution::GraphForecast &&,
                                         Status) noexcept;

} // namespace rund::compute::detail
