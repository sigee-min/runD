#pragma once

#include "../registry.hpp"

namespace rund::compute::detail::residency {

// Stateless owner for the callback-return-gated Graph HostReady protocol.
// Authority remains the sole owner of the gate, frame/epoch table, and fixed
// quarantine storage; this facet only borrows and authenticates that state.
class GraphForecastOwner final {
public:
  explicit GraphForecastOwner(Authority &) noexcept;

  GraphForecastOwner(const GraphForecastOwner &) noexcept = default;
  GraphForecastOwner &operator=(const GraphForecastOwner &) = delete;

  [[nodiscard]] AuthorityResult
  issue_graph_forecast(std::shared_ptr<const ResidencyPlan>,
                       const TiledGraphInvocation &, std::uint64_t batch,
                       std::size_t stage, std::uint32_t resource,
                       std::span<const PageUse>, GraphMaterialization,
                       FrameRegion, execution::GraphForecast &) noexcept;
  [[nodiscard]] bool terminal_graph_forecast(
      execution::GraphForecast &, Status, execution::TerminalKind,
      bool may_write,
      std::span<const execution::GraphForecastCompletion>) noexcept;
  [[nodiscard]] bool
  abort_graph_forecast(execution::GraphForecast &) noexcept;
  [[nodiscard]] bool
  quarantine_graph_forecast(execution::GraphForecast &) noexcept;
  [[nodiscard]] bool recover_graph_forecast_quarantine() noexcept;
  [[nodiscard]] bool
  release_graph_forecast(execution::GraphForecast &&) noexcept;
  [[nodiscard]] bool retire_graph_forecast(execution::GraphForecast &&,
                                           execution::GraphReady &) noexcept;

private:
  [[nodiscard]] bool validate_graph_forecast_locked(
      const execution::GraphForecast &) const noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency
