#include "local.hpp"

#include <array>
#include <memory>
#include <span>
#include <utility>

namespace rund_node_test_pipeline_residency::graph_promote {

[[nodiscard]] int CheckSingle() {
  using rund::compute::detail::Type;
  using namespace residency;

  const TiledGraphPlanInput input{
      .page_count = 2u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 104u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 104u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 104u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 4u,
                     .type = Type::U64,
                     .page_bytes = 8u,
                     .logical_bytes = 16u,
                     .kind = GraphResourceKind::ExternalOutput,
                     .persistence = ResourcePersistence::Transient}},
      .stages = {{.node = 0u,
                  .ports = {{.resource = 1u, .access = Access::Read},
                            {.resource = 2u, .access = Access::Write}}},
                 {.node = 1u,
                  .ports = {{.resource = 1u, .access = Access::Read},
                            {.resource = 3u, .access = Access::Write}}},
                 {.node = 2u,
                  .ports = {{.resource = 2u, .access = Access::Read},
                            {.resource = 3u, .access = Access::Read},
                            {.resource = 4u, .access = Access::Write}}}},
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
  const std::array<std::uint64_t, 4u> logical_bytes{104u, 104u, 104u, 16u};
  if (!owner->tiled_graph().active(2u, logical_bytes, invocation) ||
      !foreign->tiled_graph().active(2u, logical_bytes, foreign_invocation)) {
    return 2;
  }

  Authority authority;
  std::uint32_t host_first = 99u;
  std::uint32_t input_first = 99u;
  std::uint32_t output_first = 99u;
  if (!authority.register_frames(FrameTier::Host, FrameRole::Input, 1u,
                                 host_first) ||
      !authority.register_frames(FrameTier::Device, FrameRole::Input, 1u,
                                 input_first) ||
      !authority.register_frames(FrameTier::Device, FrameRole::Intermediate, 1u,
                                 output_first)) {
    return 3;
  }
  const FrameRegion host{.tier = FrameTier::Host,
                         .role = FrameRole::Input,
                         .first = host_first,
                         .count = 1u};
  const FrameRegion device_input{.tier = FrameTier::Device,
                                 .role = FrameRole::Input,
                                 .first = input_first,
                                 .count = 1u};
  const FrameRegion device_output{.tier = FrameTier::Device,
                                  .role = FrameRole::Intermediate,
                                  .first = output_first,
                                  .count = 1u};
  const GraphMaterialization input_materialization{
      .resource = 1u,
      .key = {.backing = 17u,
              .version = 5u,
              .materialization_hi = 0x0123456789abcdefull,
              .materialization_lo = 0xfedcba9876543210ull,
              .domain = CacheDomain::Backing},
      .page_bytes = 64u,
      .page_count = 2u,
      .boundary_extent = 2u,
  };
  const GraphMaterialization output_materialization{
      .resource = 2u,
      .key = {.backing = 17u,
              .version = 5u,
              .materialization_hi = 0x1111222233334444ull,
              .materialization_lo = 0x5555666677778888ull,
              .domain = CacheDomain::Transient},
      .page_bytes = 64u,
      .page_count = 2u,
      .boundary_extent = 2u,
  };
  const GraphMaterialization later_output_materialization{
      .resource = 3u,
      .key = {.backing = 17u,
              .version = 5u,
              .materialization_hi = 0x9999aaaabbbbccccull,
              .materialization_lo = 0xddddeeeeffff0000ull,
              .domain = CacheDomain::Transient},
      .page_bytes = 64u,
      .page_count = 2u,
      .boundary_extent = 2u,
  };

  const auto begin_destination = [&](const std::uint64_t batch,
                                     std::array<PageUse, 2u> &uses,
                                     Epoch &epoch) -> AuthorityResult {
    if (!invocation.project(batch, 0u, uses, epoch)) {
      return {};
    }
    const TiledGraphStage &stage = owner->tiled_graph().stages().front();
    const std::array requests{
        GraphPortRequest{.program_port = stage.ports[0].program_port,
                         .first_use = 0u,
                         .use_count = 1u,
                         .materialization = input_materialization,
                         .region = device_input,
                         .cache_regions = {device_input},
                         .cache_region_count = 1u},
        GraphPortRequest{.program_port = stage.ports[1].program_port,
                         .first_use = 1u,
                         .use_count = 1u,
                         .materialization = output_materialization,
                         .region = device_output,
                         .cache_regions = {device_output},
                         .cache_region_count = 1u},
    };
    return authority.begin_graph_epoch(uses, requests, 0u, epoch.ordinal);
  };
  const auto begin_later_destination = [&](const std::uint64_t batch,
                                           std::array<PageUse, 2u> &uses,
                                           Epoch &epoch) -> AuthorityResult {
    if (!invocation.project(batch, 1u, uses, epoch)) {
      return {};
    }
    const TiledGraphStage &stage = owner->tiled_graph().stages()[1u];
    const std::array requests{
        GraphPortRequest{.program_port = stage.ports[0].program_port,
                         .first_use = 0u,
                         .use_count = 1u,
                         .materialization = input_materialization,
                         .region = device_input,
                         .cache_regions = {device_input},
                         .cache_region_count = 1u},
        GraphPortRequest{.program_port = stage.ports[1].program_port,
                         .first_use = 1u,
                         .use_count = 1u,
                         .materialization = later_output_materialization,
                         .region = device_output,
                         .cache_regions = {device_output},
                         .cache_region_count = 1u},
    };
    return authority.begin_graph_epoch(uses, requests, 0u, epoch.ordinal);
  };

  std::array<PageUse, 2u> uses{};
  Epoch epoch{};
  if (!invocation.project(0u, 0u, uses, epoch)) {
    return 4;
  }
  execution::GraphForecast forecast{};
  if (!authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 0u, 0u, 1u,
          std::span<const PageUse>{uses.data(), 1u}, input_materialization,
          host, forecast) ||
      !terminal_forecast(authority, forecast)) {
    return 5;
  }
  AuthorityResult destination = begin_destination(0u, uses, epoch);
  if (!destination) {
    return 4;
  }
  execution::GraphPromote rejected{};
  if (authority.graph_promotes().issue_graph_promote(
          std::move(forecast), foreign_invocation, 0u, 0u, 1u,
          destination.lease.token, rejected) ||
      rejected || !forecast) {
    return 6;
  }
  execution::GraphPromote promote{};
  if (!authority.graph_promotes().issue_graph_promote(
          std::move(forecast), invocation, 0u, 0u, 1u, destination.lease.token,
          promote) ||
      forecast || promote.plan() != owner->identity() ||
      promote.coordinate() != epoch.ordinal || promote.pages().size() != 1u ||
      promote.pages().front().bytes != 64u ||
      promote.pages().front().source_frame != host.first ||
      promote.pages().front().target_frame != device_input.first ||
      authority.resume(destination.lease.token) ||
      authority.graph_promotes().release_graph_promote(std::move(promote))) {
    return 7;
  }
  if (!terminal_promote(authority, promote, true) ||
      !authority.graph_promotes().release_graph_promote(std::move(promote)) ||
      !authority.resume(destination.lease.token) ||
      !authority.complete(destination.lease.token, true)) {
    return 8;
  }

  std::array<CacheKey, 1u> input_key{};
  std::array<std::uint8_t, 1u> resident{};
  if (!project_graph_cache_key(input_materialization, uses.front().key,
                               input_key.front()) ||
      !authority.probe(input_key, resident, device_input.first,
                       device_input.count) ||
      resident.front() != 1u) {
    return 9;
  }
  std::array<CacheKey, 1u> output_key{};
  if (!project_graph_cache_key(output_materialization, uses.back().key,
                               output_key.front())) {
    return 10;
  }
  const AuthorityResult discarded = authority.begin_discard(
      output_key, device_output.first, device_output.count);
  if (!discarded || !authority.discard(discarded.lease.token)) {
    return 11;
  }

  if (!invocation.project(1u, 0u, uses, epoch) ||
      !authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 1u, 0u, 1u,
          std::span<const PageUse>{uses.data(), 1u}, input_materialization,
          host, forecast) ||
      !terminal_forecast(authority, forecast)) {
    return 12;
  }
  destination = begin_destination(1u, uses, epoch);
  if (!destination || !authority.graph_promotes().issue_graph_promote(
                          std::move(forecast), invocation, 1u, 0u, 1u,
                          destination.lease.token, promote)) {
    return 12;
  }
  if (!terminal_promote(authority, promote, false) ||
      !authority.graph_promotes().release_graph_promote(std::move(promote)) ||
      authority.resume(destination.lease.token)) {
    return 13;
  }
  if (!project_graph_cache_key(input_materialization, uses.front().key,
                               input_key.front()) ||
      !authority.probe(input_key, resident, host.first, host.count) ||
      resident.front() != 1u ||
      !authority.probe(input_key, resident, device_input.first,
                       device_input.count) ||
      resident.front() != 0u) {
    return 14;
  }

  // A later independent stage uses the exact same external input. First
  // invalidate its clean Host hit through an authenticated failed Forecast,
  // then prove that the generic stage/resource Forecast and Promote pair can
  // re-fetch and activate the stage-1 destination without stage-0 authority.
  std::array<PageUse, 2u> later_uses{};
  Epoch later_epoch{};
  if (!invocation.project(1u, 1u, later_uses, later_epoch) ||
      !authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 1u, 1u, 1u,
          std::span<const PageUse>{later_uses.data(), 1u},
          input_materialization, host, forecast)) {
    return 15;
  }
  std::array<execution::GraphForecastCompletion, 1u> failed_forecast{{
      {.key = forecast.pages().front().key,
       .bytes = 0u,
       .frame = forecast.pages().front().frame},
  }};
  if (!authority.graph_forecasts().terminal_graph_forecast(
          forecast,
          rund::compute::Status::fail(rund::compute::Reason::TransferInvalid),
          execution::TerminalKind::Known, true, failed_forecast) ||
      !authority.graph_forecasts().release_graph_forecast(
          std::move(forecast)) ||
      !authority.graph_forecasts().issue_graph_forecast(
          owner, invocation, 1u, 1u, 1u,
          std::span<const PageUse>{later_uses.data(), 1u},
          input_materialization, host, forecast) ||
      !forecast.requires_backing() || !terminal_forecast(authority, forecast)) {
    return 16;
  }
  destination = begin_later_destination(1u, later_uses, later_epoch);
  if (!destination ||
      !authority.graph_promotes().issue_graph_promote(
          std::move(forecast), invocation, 1u, 1u, 1u, destination.lease.token,
          promote) ||
      promote.coordinate() != later_epoch.ordinal ||
      promote.pages().size() != 1u ||
      promote.pages().front().bytes != input_materialization.page_bytes) {
    return 17;
  }
  const std::weak_ptr<const ResidencyPlan> lifetime = owner;
  owner.reset();
  if (lifetime.expired() || !terminal_promote(authority, promote, true) ||
      lifetime.expired() ||
      !authority.graph_promotes().release_graph_promote(std::move(promote)) ||
      !lifetime.expired() || !authority.resume(destination.lease.token) ||
      !authority.complete(destination.lease.token, true)) {
    return 18;
  }
  std::array<CacheKey, 1u> later_output_key{};
  if (!project_graph_cache_key(later_output_materialization,
                               later_uses.back().key,
                               later_output_key.front())) {
    return 19;
  }
  const AuthorityResult later_discarded = authority.begin_discard(
      later_output_key, device_output.first, device_output.count);
  if (!later_discarded || !authority.discard(later_discarded.lease.token)) {
    return 20;
  }

  foreign.reset();
  const std::array regions{host, device_input, device_output};
  return authority.release_frames(regions) ? 0 : 21;
}

} // namespace rund_node_test_pipeline_residency::graph_promote
