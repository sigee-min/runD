#include "local.hpp"

#include <array>
#include <memory>
#include <span>
#include <utility>

namespace rund_node_test_pipeline_residency::graph_promote {

[[nodiscard]] int CheckGroup() {
  using rund::compute::detail::Type;
  using namespace residency;
  const TiledGraphPlanInput input{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 40u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 40u,
                     .kind = GraphResourceKind::ExternalInput,
                     .persistence = ResourcePersistence::Backing},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 40u,
                     .kind = GraphResourceKind::Internal,
                     .persistence = ResourcePersistence::Transient},
                    {.resource = 4u,
                     .type = Type::U64,
                     .page_bytes = 8u,
                     .logical_bytes = 8u,
                     .kind = GraphResourceKind::ExternalOutput,
                     .persistence = ResourcePersistence::Transient}},
      .stages = {{.node = 0u,
                  .ports = {{.resource = 1u, .access = Access::Read},
                            {.resource = 2u, .access = Access::Read},
                            {.resource = 3u, .access = Access::Write}}},
                 {.node = 1u,
                  .domain = StageDomain::TilePartial,
                  .ports = {{.resource = 3u, .access = Access::Read},
                            {.resource = 4u, .access = Access::Write}}}},
  };
  PlanResult planned = PlanResidency(input);
  if (!planned) {
    return 1;
  }
  auto owner = std::make_shared<ResidencyPlan>(std::move(planned.plan));
  TiledGraphInvocation invocation{};
  const std::array<std::uint64_t, 4u> logical_bytes{40u, 40u, 40u, 8u};
  if (!owner->tiled_graph().active(1u, logical_bytes, invocation)) {
    return 2;
  }

  Authority authority;
  std::array<std::uint32_t, 2u> host_first{};
  std::array<std::uint32_t, 2u> input_first{};
  std::uint32_t output_first = 0u;
  for (std::size_t index = 0u; index < 2u; ++index) {
    if (!authority.register_frames(FrameTier::Host, FrameRole::Input, 1u,
                                   host_first[index]) ||
        !authority.register_frames(FrameTier::Device, FrameRole::Input, 1u,
                                   input_first[index])) {
      return 3;
    }
  }
  if (!authority.register_frames(FrameTier::Device, FrameRole::Intermediate, 1u,
                                 output_first)) {
    return 3;
  }
  const std::array host_regions{
      FrameRegion{.tier = FrameTier::Host,
                  .role = FrameRole::Input,
                  .first = host_first[0u],
                  .count = 1u},
      FrameRegion{.tier = FrameTier::Host,
                  .role = FrameRole::Input,
                  .first = host_first[1u],
                  .count = 1u},
  };
  const std::array input_regions{
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Input,
                  .first = input_first[0u],
                  .count = 1u},
      FrameRegion{.tier = FrameTier::Device,
                  .role = FrameRole::Input,
                  .first = input_first[1u],
                  .count = 1u},
  };
  const FrameRegion output_region{.tier = FrameTier::Device,
                                  .role = FrameRole::Intermediate,
                                  .first = output_first,
                                  .count = 1u};
  const std::array materializations{
      GraphMaterialization{.resource = 1u,
                           .key = {.backing = 101u,
                                   .version = 3u,
                                   .materialization_hi = 0x1111111111111111ull,
                                   .materialization_lo = 0xaaaaaaaaaaaaaaaaull,
                                   .domain = CacheDomain::Backing},
                           .page_bytes = 64u,
                           .page_count = 1u,
                           .boundary_extent = 1u},
      GraphMaterialization{.resource = 2u,
                           .key = {.backing = 202u,
                                   .version = 7u,
                                   .materialization_hi = 0x2222222222222222ull,
                                   .materialization_lo = 0xbbbbbbbbbbbbbbbbull,
                                   .domain = CacheDomain::Backing},
                           .page_bytes = 64u,
                           .page_count = 1u,
                           .boundary_extent = 1u},
  };
  const GraphMaterialization output_materialization{
      .resource = 3u,
      .key = {.backing = 303u,
              .version = 1u,
              .materialization_hi = 0x3333333333333333ull,
              .materialization_lo = 0xccccccccccccccccull,
              .domain = CacheDomain::Transient},
      .page_bytes = 64u,
      .page_count = 1u,
      .boundary_extent = 1u,
  };

  std::array<PageUse, 3u> uses{};
  Epoch epoch{};
  if (!invocation.project(0u, 0u, uses, epoch)) {
    return 4;
  }
  std::array<execution::GraphForecast, 2u> forecasts{};
  for (std::size_t index = 0u; index < forecasts.size(); ++index) {
    if (!authority.graph_forecasts().issue_graph_forecast(
            owner, invocation, 0u, 0u, static_cast<std::uint32_t>(index + 1u),
            std::span<const PageUse>{&uses[index], 1u}, materializations[index],
            host_regions[index], forecasts[index])) {
      return 5;
    }
  }
  if (!terminal_forecast(authority, forecasts[0u])) {
    return 6;
  }
  execution::GraphPromote promote{};
  if (authority.graph_promotes().issue_graph_promote_group(
          forecasts, invocation, 0u, 0u, 77u, promote) ||
      promote || !forecasts[0u] || !forecasts[1u] ||
      !terminal_forecast(authority, forecasts[1u])) {
    return 7;
  }

  const TiledGraphStage &stage = owner->tiled_graph().stages().front();
  const std::array requests{
      GraphPortRequest{.program_port = stage.ports[0u].program_port,
                       .first_use = 0u,
                       .use_count = 1u,
                       .materialization = materializations[0u],
                       .region = input_regions[0u],
                       .cache_regions = {input_regions[0u]},
                       .cache_region_count = 1u},
      GraphPortRequest{.program_port = stage.ports[1u].program_port,
                       .first_use = 1u,
                       .use_count = 1u,
                       .materialization = materializations[1u],
                       .region = input_regions[1u],
                       .cache_regions = {input_regions[1u]},
                       .cache_region_count = 1u},
      GraphPortRequest{.program_port = stage.ports[2u].program_port,
                       .first_use = 2u,
                       .use_count = 1u,
                       .materialization = output_materialization,
                       .region = output_region,
                       .cache_regions = {output_region},
                       .cache_region_count = 1u},
  };
  AuthorityResult destination =
      authority.begin_graph_epoch(uses, requests, 0u, epoch.ordinal);
  if (!destination ||
      !authority.graph_promotes().issue_graph_promote_group(
          forecasts, invocation, 0u, 0u, destination.lease.token, promote) ||
      forecasts[0u] || forecasts[1u] || !promote ||
      promote.source_count() != 2u || promote.pages().size() != 2u ||
      promote.pages()[0u].use.resource != 1u ||
      promote.pages()[1u].use.resource != 2u ||
      authority.resume(destination.lease.token) ||
      authority.graph_promotes().release_graph_promote(std::move(promote))) {
    return 8;
  }
  if (!terminal_promote(authority, promote, true) ||
      !authority.graph_promotes().release_graph_promote(std::move(promote)) ||
      !authority.resume(destination.lease.token) ||
      !authority.complete(destination.lease.token, true)) {
    return 9;
  }

  std::array<CacheKey, 1u> output_key{};
  if (!project_graph_cache_key(output_materialization, uses.back().key,
                               output_key.front())) {
    return 10;
  }
  const AuthorityResult discarded = authority.begin_discard(
      output_key, output_region.first, output_region.count);
  if (!discarded || !authority.discard(discarded.lease.token)) {
    return 11;
  }
  const std::array regions{host_regions[0u], host_regions[1u],
                           input_regions[0u], input_regions[1u], output_region};
  return authority.release_frames(regions) ? 0 : 12;
}
} // namespace rund_node_test_pipeline_residency::graph_promote
