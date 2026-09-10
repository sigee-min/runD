#include "../forecast.hpp"
#include "../../../device/residency/registry.hpp"
#include "../../../device/residency/registry/graph_forecast_owner.hpp"

#include <array>

namespace rund::compute::detail {
namespace {

using residency::execution::GraphForecast;
using residency::execution::GraphForecastCapacity;
using residency::execution::GraphForecastCompletion;
using residency::execution::TerminalKind;

[[nodiscard]] bool terminal_invalid_receipt(residency::Authority &authority,
                                            GraphForecast &forecast) noexcept {
  std::array<GraphForecastCompletion, GraphForecastCapacity> completions{};
  const auto pages = forecast.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = GraphForecastCompletion{
        .key = pages[index].key,
        .bytes = 0u,
        .frame = pages[index].frame,
    };
  }
  return authority.graph_forecasts().terminal_graph_forecast(
      forecast, Status::fail(Reason::CompletionInvalid),
      TerminalKind::UnknownMayWrite, true,
      std::span<const GraphForecastCompletion>{completions.data(),
                                               pages.size()});
}

} // namespace

Status
terminal_graph_prefetch(residency::Authority &authority,
                        GraphForecast &forecast,
                        const residency::PrefetchReceipt &receipt) noexcept {
  const auto pages = forecast.pages();
  bool exact = forecast && receipt.token == forecast.token() &&
               receipt.pages.size() == pages.size();
  for (std::size_t index = 0u; exact && index < pages.size(); ++index) {
    const residency::PrefetchedPage page = receipt.pages[index];
    exact = page.key == pages[index].key &&
            page.physical_frame == pages[index].frame &&
            page.fetched == pages[index].fetch && page.frame != nullptr &&
            page.bytes == pages[index].bytes &&
            page.backing_bytes <= page.bytes;
  }
  if (!exact) {
    return terminal_invalid_receipt(authority, forecast)
               ? Status::fail(Reason::CompletionInvalid)
               : Status::fail(Reason::PipelineInvalid);
  }

  std::array<GraphForecastCompletion, GraphForecastCapacity> completions{};
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    const residency::PrefetchedPage page = receipt.pages[index];
    completions[index] = GraphForecastCompletion{
        .key = page.key,
        .bytes = page.fetched
                     ? (receipt.status ? page.bytes : page.backing_bytes)
                     : 0u,
        .frame = page.physical_frame,
    };
  }
  if (!authority.graph_forecasts().terminal_graph_forecast(
          forecast, receipt.status, TerminalKind::Known, !receipt.status,
          std::span<const GraphForecastCompletion>{completions.data(),
                                                   pages.size()})) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return receipt.status;
}

bool reject_graph_prefetch(residency::Authority &authority,
                           GraphForecast &&forecast,
                           const Status status) noexcept {
  if (!forecast || status) {
    return false;
  }
  std::array<GraphForecastCompletion, GraphForecastCapacity> completions{};
  const auto pages = forecast.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = GraphForecastCompletion{
        .key = pages[index].key,
        .bytes = 0u,
        .frame = pages[index].frame,
    };
  }
  return authority.graph_forecasts().terminal_graph_forecast(
             forecast, status, TerminalKind::Known, false,
             std::span<const GraphForecastCompletion>{completions.data(),
                                                      pages.size()}) &&
         authority.graph_forecasts().release_graph_forecast(std::move(forecast));
}

} // namespace rund::compute::detail
