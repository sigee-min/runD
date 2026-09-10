#include "../graph_ready.hpp"

#include "../../registry/graph_forecast_owner.hpp"

#include <utility>

namespace rund::compute::detail::residency {

bool GraphForecastOwner::retire_graph_forecast(
    execution::GraphForecast &&forecast, execution::GraphReady &ready) noexcept {
  if (!forecast || ready || forecast.owner_ != &authority_) {
    return false;
  }
  if (!forecast.terminalled_) {
    bool settled = release_graph_forecast(std::move(forecast));
    if (!settled) {
      settled = abort_graph_forecast(forecast);
    }
    return settled;
  }
  if (!forecast.completion_ ||
      forecast.terminal_ != execution::TerminalKind::Known ||
      forecast.completion_may_write_) {
    bool settled = release_graph_forecast(std::move(forecast));
    if (!settled) {
      settled = abort_graph_forecast(forecast);
    }
    return settled;
  }
  if (!authority_.complete(forecast.token_, true)) {
    bool settled = release_graph_forecast(std::move(forecast));
    if (!settled) {
      settled = abort_graph_forecast(forecast);
    }
    return settled;
  }
  ready.owner_ = &authority_;
  ready.plan_owner_ = std::move(forecast.plan_owner_);
  ready.pages_ = forecast.pages_;
  ready.plan_ = forecast.plan_;
  ready.coordinate_ = forecast.coordinate_;
  ready.page_count_ = forecast.page_count_;
  forecast.clear();
  return true;
}

} // namespace rund::compute::detail::residency
