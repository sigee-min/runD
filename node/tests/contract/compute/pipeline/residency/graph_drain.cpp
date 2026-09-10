#include "local.hpp"

#include "src/compute/device/residency/execution/graph_drain.hpp"
#include "src/compute/device/residency/registry/graph_drain_owner.hpp"
#include "src/compute/device/residency/registry.hpp"
#include "src/compute/pipeline/residency/planner.hpp"

#include <array>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace {

namespace residency = rund::compute::detail::residency;
namespace execution = rund::compute::detail::residency::execution;

static_assert(std::is_move_constructible_v<execution::GraphDrain>);
static_assert(!std::is_move_assignable_v<execution::GraphDrain>);
static_assert(!std::is_copy_constructible_v<execution::GraphDrain>);

[[nodiscard]] bool
project_output(const residency::TiledGraphInvocation &invocation,
               const std::uint64_t batch, residency::PageUse &output,
               residency::Epoch &epoch) {
  std::array<residency::PageUse, 2u> uses{};
  if (!invocation.project(batch, 1u, uses, epoch)) {
    return false;
  }
  output = uses.back();
  return output.access == residency::Access::Write && output.key.resource == 3u;
}

[[nodiscard]] bool seed_device_output(
    residency::Authority &authority, const residency::PageUse output,
    const residency::GraphMaterialization materialization,
    const residency::FrameRegion region, const std::uint64_t epoch) {
  const residency::AuthorityResult seeded =
      authority.begin_graph(std::span<const residency::PageUse>{&output, 1u},
                            materialization, region, epoch);
  return seeded && authority.activate(seeded.lease.token) &&
         authority.complete(seeded.lease.token, true);
}

[[nodiscard]] bool terminal_success(residency::Authority &authority,
                                    execution::GraphDrain &drain) {
  auto drains = authority.graph_drains();
  std::array<execution::GraphDrainCompletion, execution::GraphDrainCapacity>
      completions{};
  const auto pages = drain.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = execution::GraphDrainCompletion{
        .key = pages[index].key,
        .bytes = pages[index].bytes,
        .source_frame = pages[index].source_frame,
        .target_frame = pages[index].target_frame,
    };
  }
  return drains.terminal_graph_drain(
      drain, rund::compute::Status::success(), execution::TerminalKind::Known,
      false,
      std::span<const execution::GraphDrainCompletion>{completions.data(),
                                                       pages.size()});
}

[[nodiscard]] bool terminal_failure(residency::Authority &authority,
                                    execution::GraphDrain &drain) {
  auto drains = authority.graph_drains();
  std::array<execution::GraphDrainCompletion, execution::GraphDrainCapacity>
      completions{};
  const auto pages = drain.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = execution::GraphDrainCompletion{
        .key = pages[index].key,
        .bytes = 0u,
        .source_frame = pages[index].source_frame,
        .target_frame = pages[index].target_frame,
    };
  }
  return drains.terminal_graph_drain(
      drain,
      rund::compute::Status::fail(rund::compute::Reason::TransferInvalid),
      execution::TerminalKind::Known, true,
      std::span<const execution::GraphDrainCompletion>{completions.data(),
                                                       pages.size()});
}

} // namespace

namespace rund_node_test_pipeline_residency {

int CheckGraphDrain() {
  using rund::compute::detail::Type;
  using namespace residency;

  const TiledGraphPlanInput input{
      .page_count = 3u,
      .requested_frames = 1u,
      .max_frames = 1u,
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
  PlanResult foreign_plan = PlanResidency(input);
  if (!planned || !foreign_plan) {
    return 1;
  }
  auto owner = std::make_shared<ResidencyPlan>(std::move(planned.plan));
  auto foreign = std::make_shared<ResidencyPlan>(std::move(foreign_plan.plan));
  TiledGraphInvocation invocation{};
  TiledGraphInvocation foreign_invocation{};
  const std::array<std::uint64_t, 3u> logical_bytes{2u * 64u + 40u,
                                                    2u * 64u + 40u, 3u * 8u};
  if (!owner->tiled_graph().active(3u, logical_bytes, invocation) ||
      !foreign->tiled_graph().active(3u, logical_bytes, foreign_invocation)) {
    return 2;
  }

  Authority authority;
  auto drains = authority.graph_drains();
  std::uint32_t device_first = 99u;
  std::uint32_t host_first = 99u;
  if (!authority.register_frames(FrameTier::Device, FrameRole::Output, 1u,
                                 device_first) ||
      !authority.register_frames(FrameTier::Host, FrameRole::Output, 1u,
                                 host_first)) {
    return 3;
  }
  const FrameRegion device{.tier = FrameTier::Device,
                           .role = FrameRole::Output,
                           .first = device_first,
                           .count = 1u};
  const FrameRegion host{.tier = FrameTier::Host,
                         .role = FrameRole::Output,
                         .first = host_first,
                         .count = 1u};
  const GraphMaterialization materialization{
      .resource = 3u,
      .key = {.backing = 29u,
              .version = 7u,
              .materialization_hi = 0xaabbccddeeff0011ull,
              .materialization_lo = 0x1100ffeeddccbbaaull,
              .domain = CacheDomain::Transient},
      .page_bytes = 8u,
      .page_count = 3u,
  };
  PageUse output{};
  Epoch epoch{};
  if (!project_output(invocation, 0u, output, epoch) ||
      !seed_device_output(authority, output, materialization, device,
                          epoch.ordinal)) {
    return 4;
  }

  execution::GraphDrain rejected{};
  if (drains.issue_graph_drain(owner, foreign_invocation, 0u, 1u, 3u,
                               std::span<const PageUse>{&output, 1u},
                               materialization, device, host, rejected) ||
      rejected) {
    return 5;
  }
  execution::GraphDrain drain{};
  if (!drains.issue_graph_drain(owner, invocation, 0u, 1u, 3u,
                                std::span<const PageUse>{&output, 1u},
                                materialization, device, host, drain) ||
      drain.plan() != owner->identity() ||
      drain.coordinate() != epoch.ordinal || drain.pages().size() != 1u ||
      drain.pages().front().bytes != 8u ||
      drain.pages().front().source_frame != device.first ||
      drain.pages().front().target_frame != host.first ||
      drains.release_graph_drain(std::move(drain))) {
    return 6;
  }

  execution::GraphDrain blocked{};
  if (drains.issue_graph_drain(owner, invocation, 0u, 1u, 3u,
                               std::span<const PageUse>{&output, 1u},
                               materialization, device, host, blocked) ||
      blocked || !terminal_failure(authority, drain) ||
      !drains.release_graph_drain(std::move(drain))) {
    return 7;
  }
  std::array<CacheKey, 1u> keys{};
  std::array<std::uint8_t, 1u> resident{};
  if (!project_graph_cache_key(materialization, output.key, keys.front()) ||
      !authority.probe(keys, resident, device.first, device.count) ||
      resident.front() != 0u ||
      !authority.probe(keys, resident, host.first, host.count) ||
      resident.front() != 0u ||
      !seed_device_output(authority, output, materialization, device,
                          epoch.ordinal) ||
      !drains.issue_graph_drain(owner, invocation, 0u, 1u, 3u,
                                std::span<const PageUse>{&output, 1u},
                                materialization, device, host, drain) ||
      !terminal_success(authority, drain)) {
    return 8;
  }
  const std::weak_ptr<const ResidencyPlan> lifetime = owner;
  owner.reset();
  if (lifetime.expired() || !drains.release_graph_drain(std::move(drain))) {
    return 9;
  }
  if (!authority.probe(keys, resident, host.first, host.count) ||
      resident.front() != 1u) {
    return 10;
  }
  const AuthorityResult retired =
      authority.begin_discard(keys, host.first, host.count);
  if (!retired || !authority.discard(retired.lease.token) ||
      !lifetime.expired()) {
    return 11;
  }
  const std::array regions{device, host};
  return authority.release_frames(regions) ? 0 : 12;
}

} // namespace rund_node_test_pipeline_residency
