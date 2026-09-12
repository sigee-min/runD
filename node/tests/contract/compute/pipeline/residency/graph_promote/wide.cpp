#include "local.hpp"
#include "src/compute/device/residency/execution/graph_promote/internal.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <utility>

namespace rund_node_test_pipeline_residency::graph_promote {

[[nodiscard]] int CheckWide() {
  using rund::compute::detail::Type;
  using namespace residency;
  constexpr std::size_t InputCount = execution::GraphPromoteSourceCapacity;
  static_assert(InputCount == 7u);
  static_assert(sizeof(graph_promote_detail::Group) < 48u * 1024u);

  TiledGraphPlanInput input{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
  };
  for (std::size_t index = 0u; index < InputCount; ++index) {
    input.resources.push_back(TiledGraphResourceInput{
        .resource = static_cast<std::uint32_t>(index + 1u),
        .type = Type::U64,
        .page_bytes = 64u,
        .logical_bytes = 40u,
        .kind = GraphResourceKind::ExternalInput,
        .persistence = ResourcePersistence::Backing,
    });
  }
  input.resources.push_back(TiledGraphResourceInput{
      .resource = static_cast<std::uint32_t>(InputCount + 1u),
      .type = Type::U64,
      .page_bytes = 64u,
      .logical_bytes = 40u,
      .kind = GraphResourceKind::ExternalOutput,
      .persistence = ResourcePersistence::Transient,
  });
  TiledGraphStageInput stage{.node = 0u};
  for (std::size_t index = 0u; index < InputCount; ++index) {
    stage.ports.push_back(TiledGraphPortInput{
        .resource = static_cast<std::uint32_t>(index + 1u),
        .access = Access::Read,
    });
  }
  stage.ports.push_back(TiledGraphPortInput{
      .resource = static_cast<std::uint32_t>(InputCount + 1u),
      .access = Access::Write,
  });
  input.stages.push_back(std::move(stage));

  PlanResult planned = PlanResidency(input);
  if (!planned) {
    return 1;
  }
  auto owner = std::make_shared<ResidencyPlan>(std::move(planned.plan));
  TiledGraphInvocation invocation{};
  std::array<std::uint64_t, InputCount + 1u> logical_bytes{};
  logical_bytes.fill(40u);
  if (!owner->tiled_graph().active(1u, logical_bytes, invocation)) {
    return 2;
  }

  Authority authority;
  std::array<FrameRegion, InputCount> host_regions{};
  std::array<FrameRegion, InputCount> device_regions{};
  for (std::size_t index = 0u; index < InputCount; ++index) {
    std::uint32_t host = 0u;
    std::uint32_t device = 0u;
    if (!authority.register_frames(FrameTier::Host, FrameRole::Input, 1u,
                                   host) ||
        !authority.register_frames(FrameTier::Device, FrameRole::Input, 1u,
                                   device)) {
      return 3;
    }
    host_regions[index] = FrameRegion{.tier = FrameTier::Host,
                                      .role = FrameRole::Input,
                                      .first = host,
                                      .count = 1u};
    device_regions[index] = FrameRegion{.tier = FrameTier::Device,
                                        .role = FrameRole::Input,
                                        .first = device,
                                        .count = 1u};
  }
  std::uint32_t output_first = 0u;
  if (!authority.register_frames(FrameTier::Device, FrameRole::Output, 1u,
                                 output_first)) {
    return 3;
  }
  const FrameRegion output_region{.tier = FrameTier::Device,
                                  .role = FrameRole::Output,
                                  .first = output_first,
                                  .count = 1u};

  std::array<GraphMaterialization, InputCount> materializations{};
  for (std::size_t index = 0u; index < InputCount; ++index) {
    materializations[index] = GraphMaterialization{
        .resource = static_cast<std::uint32_t>(index + 1u),
        .key = {.backing = static_cast<std::uint64_t>(101u + index),
                .version = 3u,
                .materialization_hi = 0x1111111111111111ull + index,
                .materialization_lo = 0xaaaaaaaaaaaaaaa0ull + index,
                .domain = CacheDomain::Backing},
        .page_bytes = 64u,
        .page_count = 1u,
        .boundary_extent = 1u,
    };
  }
  const GraphMaterialization output_materialization{
      .resource = static_cast<std::uint32_t>(InputCount + 1u),
      .key = {.backing = 909u,
              .version = 1u,
              .materialization_hi = 0x8888888888888888ull,
              .materialization_lo = 0x7777777777777777ull,
              .domain = CacheDomain::Transient},
      .page_bytes = 64u,
      .page_count = 1u,
      .boundary_extent = 1u,
  };

  std::array<PageUse, InputCount + 1u> uses{};
  Epoch epoch{};
  if (!invocation.project(0u, 0u, uses, epoch)) {
    return 4;
  }
  std::array<execution::GraphReady, InputCount> ready{};
  for (std::size_t index = 0u; index < InputCount; ++index) {
    execution::GraphForecast forecast{};
    if (!authority.graph_forecasts().issue_graph_forecast(
            owner, invocation, 0u, 0u, static_cast<std::uint32_t>(index + 1u),
            std::span<const PageUse>{&uses[index], 1u}, materializations[index],
            host_regions[index], forecast) ||
        !terminal_forecast(authority, forecast) ||
        !authority.graph_forecasts().retire_graph_forecast(
            std::move(forecast), ready[index]) ||
        forecast || !ready[index]) {
      return 5;
    }
    if (index == 0u) {
      execution::GraphForecast blocked{};
      if (authority.graph_forecasts().issue_graph_forecast(
              owner, invocation, 0u, 0u, 2u,
              std::span<const PageUse>{&uses[1u], 1u}, materializations[1u],
              host_regions[0u], blocked) ||
          blocked) {
        return 5;
      }
    }
  }

  execution::GraphPromote invalid{};
  if (authority.graph_promotes().issue_graph_promote_group(
          ready, invocation, 0u, 0u, 77u,
                                          invalid) ||
      invalid ||
      std::any_of(ready.begin(), ready.end(),
                  [](const auto &source) { return !source; })) {
    return 6;
  }

  const TiledGraphStage &declared = owner->tiled_graph().stages().front();
  std::array<GraphPortRequest, InputCount + 1u> requests{};
  for (std::size_t index = 0u; index < InputCount; ++index) {
    requests[index] = GraphPortRequest{
        .program_port = declared.ports[index].program_port,
        .first_use = index,
        .use_count = 1u,
        .materialization = materializations[index],
        .region = device_regions[index],
        .cache_regions = {device_regions[index]},
        .cache_region_count = 1u,
    };
  }
  requests.back() = GraphPortRequest{
      .program_port = declared.ports.back().program_port,
      .first_use = InputCount,
      .use_count = 1u,
      .materialization = output_materialization,
      .region = output_region,
      .cache_regions = {output_region},
      .cache_region_count = 1u,
  };
  AuthorityResult destination =
      authority.begin_graph_epoch(uses, requests, 0u, epoch.ordinal);
  execution::GraphPromote promote{};
  if (!destination ||
      !authority.graph_promotes().issue_graph_promote_group(
          ready, invocation, 0u, 0u,
                                           destination.lease.token, promote) ||
      !promote || promote.source_count() != InputCount ||
      promote.pages().size() != InputCount) {
    return 7;
  }
  for (std::size_t index = 0u; index < InputCount; ++index) {
    if (ready[index] || promote.pages()[index].use.resource != index + 1u ||
        promote.pages()[index].source_frame != host_regions[index].first ||
        promote.pages()[index].target_frame != device_regions[index].first) {
      return 8;
    }
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
  std::array<FrameRegion, InputCount * 2u + 1u> regions{};
  std::copy(host_regions.begin(), host_regions.end(), regions.begin());
  std::copy(device_regions.begin(), device_regions.end(),
            regions.begin() + static_cast<std::ptrdiff_t>(InputCount));
  regions.back() = output_region;
  return authority.release_frames(regions) ? 0 : 12;
}

} // namespace rund_node_test_pipeline_residency::graph_promote
