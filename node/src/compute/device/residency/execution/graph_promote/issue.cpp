#include "internal.hpp"

#include <algorithm>

namespace rund::compute::detail::residency {

AuthorityResult GraphPromoteOwner::issue_graph_promote(
    execution::GraphForecast &&forecast, const TiledGraphInvocation &invocation,
    const std::uint64_t batch, const std::size_t stage,
    const std::uint32_t resource, const std::uint64_t destination_token,
    execution::GraphPromote &ticket) noexcept {
  const auto pages = forecast.pages();
  if (pages.empty() ||
      std::any_of(pages.begin(), pages.end(), [resource](const auto page) {
        return page.use.resource != resource;
      })) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  return issue_graph_promote_group(
      std::span<execution::GraphForecast>{&forecast, 1u}, invocation, batch,
      stage, destination_token, ticket);
}

AuthorityResult GraphPromoteOwner::issue_graph_promote_group(
    const std::span<execution::GraphForecast> forecasts,
    const TiledGraphInvocation &invocation, const std::uint64_t batch,
    const std::size_t stage, const std::uint64_t destination_token,
    execution::GraphPromote &ticket) noexcept {
  graph_promote_detail::Group group{};
  const AuthorityResult validated = validate_graph_promote_group(
      forecasts, invocation, batch, stage, destination_token, group);
  if (!validated) {
    return validated;
  }
  const AuthorityResult bound =
      bind_graph_promote_group(invocation, destination_token, group, ticket);
  if (bound) {
    for (execution::GraphForecast &forecast : forecasts) {
      forecast.clear();
    }
  }
  return bound;
}

AuthorityResult GraphPromoteOwner::issue_graph_promote_group(
    const std::span<execution::GraphReady> ready,
    const TiledGraphInvocation &invocation, const std::uint64_t batch,
    const std::size_t stage, const std::uint64_t destination_token,
    execution::GraphPromote &ticket) noexcept {
  graph_promote_detail::Group group{};
  const AuthorityResult validated = validate_graph_promote_group(
      ready, invocation, batch, stage, destination_token, group);
  if (!validated) {
    return validated;
  }
  const AuthorityResult bound =
      bind_graph_promote_group(invocation, destination_token, group, ticket);
  if (bound) {
    for (execution::GraphReady &source : ready) {
      source.clear();
    }
  }
  return bound;
}

} // namespace rund::compute::detail::residency
