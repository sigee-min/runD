#include "local.hpp"

#include "src/compute/device/residency/execution/graph_drain.hpp"
#include "src/compute/device/residency/execution/graph_persist.hpp"
#include "src/compute/device/residency/registry/graph_drain_owner.hpp"
#include "src/compute/device/residency/registry/graph_persist_owner.hpp"
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

static_assert(std::is_move_constructible_v<execution::GraphPersist>);
static_assert(!std::is_move_assignable_v<execution::GraphPersist>);
static_assert(!std::is_copy_constructible_v<execution::GraphPersist>);
static_assert(execution::GraphPersistSlotCapacity == 2u);

[[nodiscard]] bool
project_output(const residency::TiledGraphInvocation &invocation,
               const std::uint64_t batch, residency::PageUse &output,
               residency::Epoch &epoch) {
  std::array<residency::PageUse, 2u> uses{};
  if (!invocation.project(batch, 1u, uses, epoch)) {
    return false;
  }
  output = uses.back();
  return output.access == residency::Access::Write &&
         output.key.resource == 3u && output.dirty.bytes != 0u;
}

[[nodiscard]] bool
seed_output(residency::Authority &authority, const residency::PageUse output,
            const residency::GraphMaterialization materialization,
            const residency::FrameRegion region, const std::uint64_t epoch) {
  const residency::AuthorityResult seeded =
      authority.begin_graph(std::span<const residency::PageUse>{&output, 1u},
                            materialization, region, epoch);
  return seeded && authority.activate(seeded.lease.token) &&
         authority.complete(seeded.lease.token, true);
}

[[nodiscard]] bool
drain_output(residency::Authority &authority,
             const std::shared_ptr<const residency::ResidencyPlan> &owner,
             const residency::TiledGraphInvocation &invocation,
             const std::uint64_t batch, const residency::PageUse output,
             const residency::GraphMaterialization materialization,
             const residency::FrameRegion source,
             const residency::FrameRegion target) {
  execution::GraphDrain drain{};
  auto drains = authority.graph_drains();
  if (!drains.issue_graph_drain(
          owner, invocation, batch, 1u, 3u,
          std::span<const residency::PageUse>{&output, 1u}, materialization,
          source, target, drain)) {
    return false;
  }
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
             drain, rund::compute::Status::success(),
             execution::TerminalKind::Known, false,
             std::span<const execution::GraphDrainCompletion>{
                 completions.data(), pages.size()}) &&
         drains.release_graph_drain(std::move(drain));
}

[[nodiscard]] bool terminal(residency::Authority &authority,
                            execution::GraphPersist &persist,
                            const rund::compute::Status status,
                            const execution::TerminalKind terminal_kind,
                            const bool may_write) {
  std::array<execution::GraphPersistCompletion, execution::GraphPersistCapacity>
      completions{};
  const auto pages = persist.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = execution::GraphPersistCompletion{
        .key = pages[index].key,
        .backing_offset = pages[index].backing_offset,
        .bytes = status ? pages[index].bytes : 0u,
        .frame = pages[index].frame,
    };
  }
  auto persists = authority.graph_persists();
  return persists.terminal_graph_persist(
      persist, status, terminal_kind, may_write,
      std::span<const execution::GraphPersistCompletion>{completions.data(),
                                                         pages.size()});
}

struct Fixture final {
  std::shared_ptr<const residency::ResidencyPlan> owner{};
  std::shared_ptr<const residency::ResidencyPlan> foreign{};
  residency::TiledGraphInvocation invocation{};
  residency::TiledGraphInvocation foreign_invocation{};
  residency::GraphMaterialization materialization{};
  std::array<residency::PageUse, 3u> outputs{};
  std::array<residency::Epoch, 3u> epochs{};

  [[nodiscard]] bool initialize() {
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
                       .page_bytes = 64u,
                       .logical_bytes = 2u * 64u + 40u,
                       .kind = GraphResourceKind::ExternalOutput,
                       .persistence = ResourcePersistence::Backing}},
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
      return false;
    }
    owner = std::make_shared<ResidencyPlan>(std::move(planned.plan));
    foreign = std::make_shared<ResidencyPlan>(std::move(isomorphic.plan));
    const std::array<std::uint64_t, 3u> logical_bytes{
        2u * 64u + 40u, 2u * 64u + 40u, 2u * 64u + 40u};
    if (!owner->tiled_graph().active(3u, logical_bytes, invocation) ||
        !foreign->tiled_graph().active(3u, logical_bytes, foreign_invocation)) {
      return false;
    }
    materialization = GraphMaterialization{
        .resource = 3u,
        .key = {.backing = 41u,
                .version = 9u,
                .materialization_hi = 0x1122334455667788ull,
                .materialization_lo = 0x8877665544332211ull,
                .domain = CacheDomain::Backing},
        .page_bytes = 64u,
        .page_count = 3u,
        .boundary_extent = 5u,
    };
    for (std::uint64_t batch = 0u; batch < outputs.size(); ++batch) {
      if (!project_output(invocation, batch, outputs[batch], epochs[batch])) {
        return false;
      }
    }
    return true;
  }
};

[[nodiscard]] residency::AuthorityResult
issue(residency::Authority &authority,
      const std::shared_ptr<const residency::ResidencyPlan> &owner,
      const residency::TiledGraphInvocation &invocation,
      const std::uint64_t batch, const residency::PageUse output,
      const residency::GraphMaterialization materialization,
      const residency::FrameRegion region, execution::GraphPersist &persist) {
  return authority.graph_persists().issue_graph_persist(
      owner, invocation, batch, 1u, 3u,
      std::span<const residency::PageUse>{&output, 1u}, materialization, region,
      persist);
}

} // namespace

namespace rund_node_test_pipeline_residency {

int CheckGraphPersist() {
  using namespace residency;

  Fixture fixture{};
  if (!fixture.initialize()) {
    return 1;
  }
  Authority authority;
  auto persists = authority.graph_persists();
  std::uint32_t device_first = 99u;
  std::uint32_t host_first = 99u;
  if (!authority.register_frames(FrameTier::Device, FrameRole::Output, 2u,
                                 device_first) ||
      !authority.register_frames(FrameTier::Host, FrameRole::Output, 2u,
                                 host_first)) {
    return 2;
  }
  const FrameRegion device_zero{.tier = FrameTier::Device,
                                .role = FrameRole::Output,
                                .first = device_first,
                                .count = 1u};
  const FrameRegion device_one{.tier = FrameTier::Device,
                               .role = FrameRole::Output,
                               .first = device_first + 1u,
                               .count = 1u};
  const FrameRegion bank_zero{.tier = FrameTier::Host,
                              .role = FrameRole::Output,
                              .first = host_first,
                              .count = 1u};
  const FrameRegion bank_one{.tier = FrameTier::Host,
                             .role = FrameRole::Output,
                             .first = host_first + 1u,
                             .count = 1u};
  if (!seed_output(authority, fixture.outputs[0], fixture.materialization,
                   device_zero, fixture.epochs[0].ordinal) ||
      !drain_output(authority, fixture.owner, fixture.invocation, 0u,
                    fixture.outputs[0], fixture.materialization, device_zero,
                    bank_zero) ||
      !seed_output(authority, fixture.outputs[1], fixture.materialization,
                   device_one, fixture.epochs[1].ordinal) ||
      !drain_output(authority, fixture.owner, fixture.invocation, 1u,
                    fixture.outputs[1], fixture.materialization, device_one,
                    bank_one)) {
    return 3;
  }

  execution::GraphPersist rejected{};
  if (issue(authority, fixture.owner, fixture.foreign_invocation, 0u,
            fixture.outputs[0], fixture.materialization, bank_zero, rejected) ||
      rejected) {
    return 4;
  }
  execution::GraphPersist current{};
  execution::GraphPersist future{};
  if (!issue(authority, fixture.owner, fixture.invocation, 0u,
             fixture.outputs[0], fixture.materialization, bank_zero, current)) {
    return 50;
  }
  const AuthorityResult future_result =
      issue(authority, fixture.owner, fixture.invocation, 1u,
            fixture.outputs[1], fixture.materialization, bank_one, future);
  if (!future_result) {
    return 51 + static_cast<int>(future_result.failure);
  }
  if (current.plan() != fixture.owner->identity() ||
      future.plan() != fixture.owner->identity() ||
      current.coordinate() != fixture.epochs[0].ordinal ||
      future.coordinate() != fixture.epochs[1].ordinal ||
      current.pages().size() != 1u || future.pages().size() != 1u ||
      current.pages().front().backing_offset != 0u ||
      future.pages().front().backing_offset != 64u ||
      current.pages().front().bytes != 64u ||
      future.pages().front().bytes != 64u) {
    return 5;
  }

  // A later callback may terminal first. Neither its fixed Persist slot nor
  // its exact Host frame is reusable until callback-return release.
  if (!terminal(authority, future, rund::compute::Status::success(),
                execution::TerminalKind::Known, false) ||
      !terminal(authority, current, rund::compute::Status::success(),
                execution::TerminalKind::Known, false)) {
    return 6;
  }
  execution::GraphPersist blocked{};
  if (issue(authority, fixture.owner, fixture.invocation, 2u,
            fixture.outputs[2], fixture.materialization, bank_zero, blocked) ||
      blocked || !persists.release_graph_persist(std::move(current)) ||
      !seed_output(authority, fixture.outputs[2], fixture.materialization,
                   device_zero, fixture.epochs[2].ordinal) ||
      !drain_output(authority, fixture.owner, fixture.invocation, 2u,
                    fixture.outputs[2], fixture.materialization, device_zero,
                    bank_zero)) {
    return 7;
  }
  execution::GraphPersist reused{};
  if (!issue(authority, fixture.owner, fixture.invocation, 2u,
             fixture.outputs[2], fixture.materialization, bank_zero, reused) ||
      reused.pages().front().frame != bank_zero.first ||
      reused.pages().front().backing_offset != 128u ||
      reused.pages().front().bytes != 40u ||
      !terminal(authority, reused, rund::compute::Status::success(),
                execution::TerminalKind::Known, false) ||
      !persists.release_graph_persist(std::move(reused)) ||
      !persists.release_graph_persist(std::move(future))) {
    return 8;
  }
  if (!authority.release_frames(
          std::array{device_zero, device_one, bank_zero, bank_one})) {
    return 9;
  }

  // Known no-write failure restores the exact Dirty row, so a bounded retry
  // needs neither another Drain nor a second cache authority.
  Fixture retry_fixture{};
  Authority retry_authority;
  auto retry_persists = retry_authority.graph_persists();
  std::uint32_t retry_first = 99u;
  if (!retry_fixture.initialize() ||
      !retry_authority.register_frames(FrameTier::Host, FrameRole::Output, 1u,
                                       retry_first)) {
    return 10;
  }
  const FrameRegion retry_region{.tier = FrameTier::Host,
                                 .role = FrameRole::Output,
                                 .first = retry_first,
                                 .count = 1u};
  if (!seed_output(retry_authority, retry_fixture.outputs[0],
                   retry_fixture.materialization, retry_region,
                   retry_fixture.epochs[0].ordinal)) {
    return 11;
  }
  execution::GraphPersist failed{};
  if (!issue(retry_authority, retry_fixture.owner, retry_fixture.invocation, 0u,
             retry_fixture.outputs[0], retry_fixture.materialization,
             retry_region, failed) ||
      !terminal(
          retry_authority, failed,
          rund::compute::Status::fail(rund::compute::Reason::TransferInvalid),
          execution::TerminalKind::Known, false) ||
      !retry_persists.release_graph_persist(std::move(failed))) {
    return 12;
  }
  execution::GraphPersist retry{};
  if (!issue(retry_authority, retry_fixture.owner, retry_fixture.invocation, 0u,
             retry_fixture.outputs[0], retry_fixture.materialization,
             retry_region, retry)) {
    return 13;
  }
  const std::weak_ptr<const ResidencyPlan> lifetime = retry_fixture.owner;
  retry_fixture.owner.reset();
  if (lifetime.expired() ||
      !terminal(retry_authority, retry, rund::compute::Status::success(),
                execution::TerminalKind::Known, false) ||
      !retry_persists.release_graph_persist(std::move(retry)) ||
      !lifetime.expired() ||
      !retry_authority.release_frames(std::array{retry_region})) {
    return 14;
  }

  // Unknown completion quarantines both the exact Writeback frame and its
  // fixed slot. Destruction cannot silently turn uncertain backing visibility
  // into a reusable physical row.
  Fixture unknown_fixture{};
  Authority unknown_authority;
  auto unknown_persists = unknown_authority.graph_persists();
  std::uint32_t unknown_first = 99u;
  if (!unknown_fixture.initialize() ||
      !unknown_authority.register_frames(FrameTier::Host, FrameRole::Output, 1u,
                                         unknown_first)) {
    return 13;
  }
  const FrameRegion unknown_region{.tier = FrameTier::Host,
                                   .role = FrameRole::Output,
                                   .first = unknown_first,
                                   .count = 1u};
  if (!seed_output(unknown_authority, unknown_fixture.outputs[0],
                   unknown_fixture.materialization, unknown_region,
                   unknown_fixture.epochs[0].ordinal)) {
    return 14;
  }
  execution::GraphPersist uncertain{};
  if (!issue(unknown_authority, unknown_fixture.owner,
             unknown_fixture.invocation, 0u, unknown_fixture.outputs[0],
             unknown_fixture.materialization, unknown_region, uncertain) ||
      !terminal(unknown_authority, uncertain,
                rund::compute::Status::fail(rund::compute::Reason::DeviceLost),
                execution::TerminalKind::UnknownMayWrite, true) ||
      !unknown_persists.release_graph_persist(std::move(uncertain))) {
    return 15;
  }
  execution::GraphPersist after_unknown{};
  return !issue(unknown_authority, unknown_fixture.owner,
                unknown_fixture.invocation, 0u, unknown_fixture.outputs[0],
                unknown_fixture.materialization, unknown_region,
                after_unknown) &&
                 !after_unknown &&
                 !unknown_authority.release_frames(std::array{unknown_region})
             ? 0
             : 16;
}

} // namespace rund_node_test_pipeline_residency
