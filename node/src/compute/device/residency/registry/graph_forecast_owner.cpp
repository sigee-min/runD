#include "graph_forecast_owner.hpp"

namespace rund::compute::detail::residency {

GraphForecastOwner::GraphForecastOwner(Authority &authority) noexcept
    : authority_(authority) {}

GraphForecastOwner Authority::graph_forecasts() noexcept {
  return GraphForecastOwner{*this};
}

} // namespace rund::compute::detail::residency
