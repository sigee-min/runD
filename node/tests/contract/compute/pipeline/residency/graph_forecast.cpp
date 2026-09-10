#include "local.hpp"

#include "src/compute/device/residency/execution/graph_forecast.hpp"
#include "src/compute/device/residency/registry/graph_forecast_owner.hpp"
#include "src/compute/device/residency/registry.hpp"
#include "src/compute/pipeline/residency/planner.hpp"

#include <array>
#include <memory>
#include <span>
#include <utility>

namespace {

namespace residency = rund::compute::detail::residency;
namespace execution = rund::compute::detail::residency::execution;

[[nodiscard]] bool terminal_success(residency::Authority &authority,
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

[[nodiscard]] bool
project_input(const residency::TiledGraphInvocation &invocation,
              const std::uint64_t batch, residency::PageUse &input,
              residency::Epoch &epoch) {
  std::array<residency::PageUse, 2u> uses{};
  if (!invocation.project(batch, 0u, uses, epoch)) {
    return false;
  }
  input = uses.front();
  return input.access == residency::Access::Read && input.key.resource == 1u;
}

} // namespace

namespace rund_node_test_pipeline_residency {

int CheckGraphForecast() {
  using rund::compute::detail::Type;
  using namespace residency;

  const TiledGraphPlanInput input{
      .page_count = 3u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .prefetch_distance = 2u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 2u * 64u + 40u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 2u * 64u + 40u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 8u,
                     .logical_bytes = 3u * 8u,
                     .kind = GraphResourceKind::ExternalOutput,
                     .persistence = ResourcePersistence::Transient}},
      .stages = {{.node = 0u,
                  .ports = {{.resource = 1u, .access = Access::Read},
                            {.resource = 2u, .access = Access::Write}}},
                 {.node = 1u,
                  .ports = {{.resource = 2u, .access = Access::Read},
                            {.resource = 3u, .access = Access::Write}}}},
  };
  PlanResult planned = PlanResidency(input);
  PlanResult isomorphic = PlanResidency(input);
  if (!planned || !isomorphic) {
    return 1;
  }
  auto owner = std::make_shared<ResidencyPlan>(std::move(planned.plan));
  auto foreign = std::make_shared<ResidencyPlan>(std::move(isomorphic.plan));
  TiledGraphInvocation invocation{};
  TiledGraphInvocation foreign_invocation{};
  const std::array<std::uint64_t, 3u> logical_bytes{2u * 64u + 40u,
                                                    2u * 64u + 40u, 3u * 8u};
  if (!owner->tiled_graph().active(3u, logical_bytes, invocation) ||
      !foreign->tiled_graph().active(3u, logical_bytes, foreign_invocation) ||
      !invocation.owned_by(owner->tiled_graph()) ||
      invocation.owned_by(foreign->tiled_graph())) {
    return 2;
  }

  Authority authority;
  std::uint32_t first = 99u;
  if (!authority.register_frames(FrameTier::Host, FrameRole::Input, 2u,
                                 first)) {
    return 3;
  }
  const FrameRegion bank_zero{.tier = FrameTier::Host,
                              .role = FrameRole::Input,
                              .first = first,
                              .count = 1u};
  const FrameRegion bank_one{.tier = FrameTier::Host,
                             .role = FrameRole::Input,
                             .first = first + 1u,
                             .count = 1u};
  const GraphMaterialization materialization{
      .resource = 1u,
      .key = {.backing = 17u,
              .version = 5u,
              .materialization_hi = 0x0123456789abcdefull,
              .materialization_lo = 0xfedcba9876543210ull,
              .domain = CacheDomain::Backing},
      .page_bytes = 64u,
      .page_count = 3u,
      .boundary_extent = 3u,
  };

  std::array<PageUse, 3u> uses{};
  std::array<Epoch, 3u> epochs{};
  for (std::uint64_t batch = 0u; batch < uses.size(); ++batch) {
    if (!project_input(invocation, batch, uses[batch], epochs[batch])) {
      return 4;
    }
  }
  execution::GraphForecast rejected{};
  if (authority.graph_forecasts().issue_graph_forecast(
          owner, foreign_invocation, 0u, 0u, 1u,
          std::span<const PageUse>{&uses[0], 1u}, materialization, bank_zero,
          rejected) ||
      rejected) {
    return 5;
  }

  execution::GraphForecast malformed{};
  if (!authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 0u, 0u, 1u,
          std::span<const PageUse>{&uses[0], 1u}, materialization, bank_zero,
          malformed)) {
    return 6;
  }
  const auto malformed_page = malformed.pages().front();
  const std::array malformed_completion{execution::GraphForecastCompletion{
      .key =
          CacheKey{.backing = malformed_page.key.backing + 1u,
                   .version = malformed_page.key.version,
                   .extent = malformed_page.key.extent,
                   .materialization_hi = malformed_page.key.materialization_hi,
                   .materialization_lo = malformed_page.key.materialization_lo,
                   .page = malformed_page.key.page,
                   .domain = malformed_page.key.domain},
      .bytes = malformed_page.bytes,
      .frame = malformed_page.frame,
  }};
  std::array<std::uint8_t, 1u> resident{};
  const std::array malformed_key{malformed_page.key};
  if (!authority.graph_forecasts().terminal_graph_forecast(
          malformed, rund::compute::Status::success(),
          execution::TerminalKind::Known, false, malformed_completion) ||
      !authority.graph_forecasts().release_graph_forecast(std::move(malformed)) ||
      !authority.probe(malformed_key, resident, bank_zero.first,
                       bank_zero.count) ||
      resident.front() != 0u) {
    return 7;
  }

  execution::GraphForecast current{};
  execution::GraphForecast future{};
  if (!authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 0u, 0u, 1u,
          std::span<const PageUse>{&uses[0], 1u}, materialization, bank_zero,
          current) ||
      !authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 1u, 0u, 1u,
          std::span<const PageUse>{&uses[1], 1u}, materialization, bank_one,
          future) ||
      current.plan() != owner->identity() ||
      future.plan() != owner->identity() ||
      current.coordinate() != epochs[0].ordinal ||
      future.coordinate() != epochs[1].ordinal ||
      current.pages().size() != 1u || future.pages().size() != 1u ||
      current.pages().front().key.materialization_hi !=
          materialization.key.materialization_hi ||
      current.pages().front().key.materialization_lo !=
          materialization.key.materialization_lo) {
    return 8;
  }

  // A future callback may return first. Terminal observation does not release
  // its physical capability, and neither bank can be reused before its exact
  // callback-return owner is explicitly released.
  if (!terminal_success(authority, future) ||
      !terminal_success(authority, current)) {
    return 9;
  }
  execution::GraphForecast blocked{};
  if (authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 2u, 0u, 1u,
          std::span<const PageUse>{&uses[2], 1u}, materialization, bank_zero,
          blocked) ||
      blocked) {
    return 10;
  }
  if (!authority.graph_forecasts().release_graph_forecast(std::move(current))) {
    return 11;
  }
  execution::GraphForecast reused{};
  if (!authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 2u, 0u, 1u,
          std::span<const PageUse>{&uses[2], 1u}, materialization, bank_zero,
          reused) ||
      reused.pages().front().frame != bank_zero.first ||
      reused.pages().front().bytes != 40u ||
      reused.pages().front().key.extent != materialization.boundary_extent) {
    return 12;
  }
  const std::weak_ptr<const ResidencyPlan> lifetime = owner;
  owner.reset();
  if (lifetime.expired() || !terminal_success(authority, reused) ||
      !authority.graph_forecasts().release_graph_forecast(std::move(reused)) ||
      !authority.graph_forecasts().release_graph_forecast(std::move(future))) {
    return 13;
  }
  if (!lifetime.expired()) {
    return 14;
  }

  return authority.release_frames(std::array{bank_zero, bank_one}) ? 0 : 15;
}

} // namespace rund_node_test_pipeline_residency
