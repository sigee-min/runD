#include "internal.hpp"

#include <algorithm>

namespace rund::compute::detail::residency {

bool GraphForecastOwner::quarantine_graph_forecast(
    execution::GraphForecast &ticket) noexcept {
  std::lock_guard lock{authority_.gate_};
  const auto holder = std::find_if(
      authority_.cpu_graph_state_.graph_forecast_quarantine.begin(),
      authority_.cpu_graph_state_.graph_forecast_quarantine.end(),
      [](const execution::GraphForecast &candidate) { return !candidate; });
  if (holder == authority_.cpu_graph_state_.graph_forecast_quarantine.end() ||
      !validate_graph_forecast_locked(ticket)) {
    return false;
  }
  *holder = std::move(ticket);
  return true;
}

bool GraphForecastOwner::recover_graph_forecast_quarantine() noexcept {
  std::lock_guard lock{authority_.gate_};
  for (auto &holder : authority_.cpu_graph_state_.graph_forecast_quarantine) {
    if (!holder) {
      continue;
    }
    if (!validate_graph_forecast_locked(holder) ||
        !authority_.complete_locked(holder.token_, false, true, true)) {
      return false;
    }
    holder.clear();
  }
  return true;
}

bool GraphForecastOwner::release_graph_forecast(
    execution::GraphForecast &&ticket) noexcept {
  if (!ticket || ticket.owner_ != &authority_) {
    return false;
  }
  if (!ticket.terminalled_) {
    return abort_graph_forecast(ticket);
  }
  const bool success = static_cast<bool>(ticket.completion_);
  const bool invalidate =
      ticket.completion_may_write_ ||
      ticket.terminal_ == execution::TerminalKind::UnknownMayWrite;
  if (authority_.complete(ticket.token_, success, invalidate)) {
    ticket.clear();
    return true;
  }
  return abort_graph_forecast(ticket);
}

} // namespace rund::compute::detail::residency
