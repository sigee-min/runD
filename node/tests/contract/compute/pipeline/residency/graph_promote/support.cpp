#include "local.hpp"

#include <algorithm>
#include <array>
#include <type_traits>

namespace rund_node_test_pipeline_residency::graph_promote {

static_assert(std::is_move_constructible_v<execution::GraphPromote>);
static_assert(!std::is_move_assignable_v<execution::GraphPromote>);
static_assert(!std::is_copy_constructible_v<execution::GraphPromote>);
static_assert(std::is_move_constructible_v<execution::GraphReady>);
static_assert(!std::is_move_assignable_v<execution::GraphReady>);
static_assert(!std::is_copy_constructible_v<execution::GraphReady>);

[[nodiscard]] bool terminal_forecast(residency::Authority &authority,
                                     execution::GraphForecast &forecast) {
  std::array<execution::GraphForecastCompletion,
             execution::GraphForecastCapacity>
      completions{};
  const auto pages = forecast.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = execution::GraphForecastCompletion{
        .key = pages[index].key,
        .bytes = pages[index].fetch ? pages[index].bytes : 0u,
        .frame = pages[index].frame,
    };
  }
  return authority.graph_forecasts().terminal_graph_forecast(
      forecast, rund::compute::Status::success(),
      execution::TerminalKind::Known, false,
      std::span<const execution::GraphForecastCompletion>{completions.data(),
                                                          pages.size()});
}

[[nodiscard]] bool terminal_promote(residency::Authority &authority,
                                    execution::GraphPromote &promote,
                                    const bool success) {
  std::array<execution::GraphPromoteCompletion, execution::GraphPromoteCapacity>
      completions{};
  const auto pages = promote.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = execution::GraphPromoteCompletion{
        .key = pages[index].key,
        .bytes = success ? pages[index].bytes : 0u,
        .source_frame = pages[index].source_frame,
        .target_frame = pages[index].target_frame,
    };
  }
  return authority.graph_promotes().terminal_graph_promote(
      promote,
      success
          ? rund::compute::Status::success()
          : rund::compute::Status::fail(rund::compute::Reason::TransferInvalid),
      execution::TerminalKind::Known, !success,
      std::span<const execution::GraphPromoteCompletion>{completions.data(),
                                                         pages.size()});
}

} // namespace rund_node_test_pipeline_residency::graph_promote
