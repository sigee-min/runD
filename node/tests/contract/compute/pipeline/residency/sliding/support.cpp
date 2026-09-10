#include "local.hpp"

#include "src/compute/device/residency/execution/sliding.hpp"
#include "src/compute/device/residency/registry.hpp"
#include "src/compute/pipeline/residency/planner.hpp"

#include <array>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

namespace rund_node_test_pipeline_residency::sliding_detail {

namespace execution = rund::compute::detail::residency::execution;
namespace residency = rund::compute::detail::residency;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::detail::Type;

[[nodiscard]] GraphFixture graph_fixture() {
  const residency::TiledGraphPlanInput graph_input{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .prefetch_distance = 1u,
      .resources = {{.resource = 1u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::ExternalInput,
                     .persistence = residency::ResourcePersistence::Backing},
                    {.resource = 2u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::ExternalOutput,
                     .persistence = residency::ResourcePersistence::Transient},
                    {.resource = 3u,
                     .type = Type::U64,
                     .page_bytes = 64u,
                     .logical_bytes = 64u,
                     .kind = residency::GraphResourceKind::ExternalOutput,
                     .persistence = residency::ResourcePersistence::Transient}},
      .stages =
          {{.node = 1u,
            .ports = {{.resource = 1u, .access = residency::Access::Read},
                      {.resource = 2u, .access = residency::Access::Write},
                      {.resource = 3u, .access = residency::Access::Write}}}},
  };
  residency::PlanResult graph_plan = residency::PlanResidency(graph_input);

  if (!graph_plan) {
    return {};
  }
  GraphFixture result{};
  result.owner =
      std::make_shared<residency::ResidencyPlan>(std::move(graph_plan.plan));
  result.bytes = {64u, 64u, 64u};
  return result;
}

static_assert(std::is_move_constructible_v<execution::SlidingFetch>);
static_assert(!std::is_move_assignable_v<execution::SlidingFetch>);
static_assert(!std::is_copy_constructible_v<execution::SlidingFetch>);
static_assert(std::is_move_constructible_v<execution::SlidingPromote>);
static_assert(!std::is_move_assignable_v<execution::SlidingPromote>);
static_assert(!std::is_copy_constructible_v<execution::SlidingPromote>);
static_assert(std::is_move_constructible_v<execution::SlidingNative>);
static_assert(!std::is_move_assignable_v<execution::SlidingNative>);
static_assert(!std::is_copy_constructible_v<execution::SlidingNative>);
static_assert(std::is_move_constructible_v<execution::SlidingDrain>);
static_assert(!std::is_move_assignable_v<execution::SlidingDrain>);
static_assert(!std::is_copy_constructible_v<execution::SlidingDrain>);
static_assert(std::is_move_constructible_v<execution::SlidingPersist>);
static_assert(!std::is_move_assignable_v<execution::SlidingPersist>);
static_assert(!std::is_copy_constructible_v<execution::SlidingPersist>);

[[nodiscard]] residency::FrameRegion region(const residency::FrameTier tier,
                                            const residency::FrameRole role,
                                            const std::uint32_t first,
                                            const std::uint32_t count) {
  return residency::FrameRegion{
      .tier = tier, .role = role, .first = first, .count = count};
}

[[nodiscard]] std::shared_ptr<const execution::Plan>
direct_plan(const std::uint64_t pages, const std::uint32_t capacity,
            const std::uint32_t host_capacity, const std::uint64_t input_retain,
            const std::uint64_t input_logical_bytes,
            const execution::FetchFill fill,
            const std::uint64_t input_page_bytes,
            const std::uint64_t input_payload_bytes,
            const std::uint64_t input_prefix_bytes,
            const std::uint64_t input_suffix_bytes,
            const std::uint64_t fill_element_bytes, const bool canonical_window,
            const std::uint64_t expanded_boundary_extent,
            const std::uint64_t canonical_boundary_extent) {
  const residency::GraphMaterialization canonical_input =
      canonical_window
          ? residency::GraphMaterialization{
                .resource = 1u,
                .key = residency::CacheKey{
                    .backing = 11u,
                    .version = 3u,
                    .materialization_hi = 0x43414e4f4e494341ull,
                    .materialization_lo = 0x4c5f504147455f31ull,
                },
                .page_bytes = input_payload_bytes,
                .page_count = pages,
                .boundary_extent = canonical_boundary_extent,
            }
          : residency::GraphMaterialization{};
  const execution::Request request{
      .page_count = pages,
      .frame_capacity = capacity,
      .prefetch_distance = 2u,
      .input =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{
                      .resource = 1u,
                      .key = residency::CacheKey{.backing = 11u, .version = 3u},
                      .page_bytes = input_page_bytes,
                      .page_count = pages,
                      .boundary_extent = expanded_boundary_extent,
                  },
              .access = residency::Access::Read,
              .logical_bytes = input_logical_bytes == 0u
                                   ? pages * input_payload_bytes
                                   : input_logical_bytes,
              .payload_bytes = input_payload_bytes,
              .read_prefix_bytes = input_prefix_bytes,
              .target_prefix_bytes = input_prefix_bytes,
              .read_suffix_bytes = input_suffix_bytes,
              .fill = fill,
              .fill_element_bytes = fill_element_bytes,
              .next_use_base = pages,
              .next_use_stride = 1u,
              .retain_until = input_retain,
          },
      .canonical_input = canonical_input,
      .output =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{
                      .resource = 2u,
                      .key = residency::CacheKey{.backing = 12u, .version = 4u},
                      .page_bytes = 16u,
                      .page_count = pages,
                  },
              .access = residency::Access::Write,
              .logical_bytes = pages * 16u,
              .payload_bytes = 16u,
          },
      .publication =
          execution::Publication{
              .backing = 12u,
              .version = 4u,
              .extent = residency::DirtyExtent{.bytes = pages * 16u},
          },
      .host_input = {region(residency::FrameTier::Host,
                            residency::FrameRole::Input, 0u, host_capacity),
                     region(residency::FrameTier::Host,
                            residency::FrameRole::Input, host_capacity,
                            host_capacity)},
      .device_input = {region(residency::FrameTier::Device,
                              residency::FrameRole::Input, 2u * host_capacity,
                              capacity),
                       region(residency::FrameTier::Device,
                              residency::FrameRole::Input,
                              2u * host_capacity + capacity, capacity)},
      .device_output = {region(residency::FrameTier::Device,
                               residency::FrameRole::Output,
                               2u * host_capacity + 2u * capacity, capacity),
                        region(residency::FrameTier::Device,
                               residency::FrameRole::Output,
                               2u * host_capacity + 3u * capacity, capacity)},
      .host_output =
          {region(residency::FrameTier::Host, residency::FrameRole::Output,
                  2u * host_capacity + 4u * capacity, host_capacity),
           region(residency::FrameTier::Host, residency::FrameRole::Output,
                  3u * host_capacity + 4u * capacity, host_capacity)},
  };
  const execution::SealResult sealed = execution::seal(request);
  return sealed ? std::make_shared<const execution::Plan>(sealed.plan)
                : nullptr;
}

[[nodiscard]] bool register_direct(residency::Authority &authority,
                                   const std::uint32_t capacity,
                                   const std::uint32_t host_capacity) {
  const std::array<std::pair<residency::FrameTier, residency::FrameRole>, 8u>
      owners{{
          {residency::FrameTier::Host, residency::FrameRole::Input},
          {residency::FrameTier::Host, residency::FrameRole::Input},
          {residency::FrameTier::Device, residency::FrameRole::Input},
          {residency::FrameTier::Device, residency::FrameRole::Input},
          {residency::FrameTier::Device, residency::FrameRole::Output},
          {residency::FrameTier::Device, residency::FrameRole::Output},
          {residency::FrameTier::Host, residency::FrameRole::Output},
          {residency::FrameTier::Host, residency::FrameRole::Output},
      }};
  const std::array<std::uint32_t, 8u> counts{
      host_capacity, host_capacity, capacity,      capacity,
      capacity,      capacity,      host_capacity, host_capacity,
  };
  std::uint32_t expected = 0u;
  for (std::size_t index = 0u; index < owners.size(); ++index) {
    std::uint32_t first = 0u;
    if (!authority.register_frames(owners[index].first, owners[index].second,
                                   counts[index], first) ||
        first != expected) {
      return false;
    }
    expected += counts[index];
  }
  return true;
}

[[nodiscard]] bool fetch(execution::Sliding &sliding, const std::uint64_t epoch,
                         EpochWork &work, execution::SlidingTicket *first) {
  if (!sliding.project(epoch, work.uses, work.projection)) {
    return false;
  }
  std::size_t fetched = 0u;
  for (std::size_t use = 0u; use < work.projection.use_count; ++use) {
    if (fetched == work.projection.fetch_count) {
      break;
    }
    execution::SlidingTicket ticket{};
    if (!sliding.issue_fetch(work.projection, work.uses, use, ticket)) {
      continue;
    }
    if (first != nullptr && fetched == 0u) {
      *first = ticket;
    }
    if (!sliding.fetch_terminal(ticket, Status::success(),
                                execution::TerminalKind::Known, true,
                                ticket.expected_bytes)) {
      return false;
    }
    ++fetched;
  }
  return fetched == work.projection.fetch_count;
}

[[nodiscard]] bool fetch_bound(residency::Authority &authority,
                               const execution::Plan &plan,
                               execution::Sliding &sliding,
                               const std::uint64_t epoch, EpochWork &work) {
  auto owner = authority.sliding();
  if (!sliding.project(epoch, work.uses, work.projection)) {
    return false;
  }
  std::size_t fetched = 0u;
  for (std::size_t use = 0u; use < work.projection.use_count; ++use) {
    if (fetched == work.projection.fetch_count) {
      break;
    }
    execution::SlidingFetch ticket{};
    if (!owner.issue_execution_sliding_fetch(plan, sliding, work.projection,
                                              work.uses, use, ticket)) {
      continue;
    }
    const bool backing = ticket.requires_backing();
    if (backing &&
        !owner.terminal_execution_sliding_fetch(
            sliding, ticket, Status::success(), execution::TerminalKind::Known,
            true, ticket.backing_bytes())) {
      return false;
    }
    if (!owner.release_execution_sliding_fetch(sliding, std::move(ticket))) {
      return false;
    }
    ++fetched;
  }
  return fetched == work.projection.fetch_count;
}

[[nodiscard]] bool admit(execution::Sliding &sliding, EpochWork &work) {
  execution::SlidingTicket promote{};
  return sliding.issue_promote(work.projection, work.uses, promote) &&
         sliding.promote_terminal(promote, Status::success(),
                                  execution::TerminalKind::Known, true,
                                  work.projection.fetch_bytes) &&
         sliding.admit(work.projection, work.uses, work.native);
}

[[nodiscard]] bool admit_bound(residency::Authority &authority,
                               const execution::Plan &plan,
                               execution::Sliding &sliding, EpochWork &work) {
  auto owner = authority.sliding();
  execution::SlidingPromote promote{};
  if (!owner.issue_execution_sliding_promote(plan, sliding, work.projection,
                                             work.uses, promote)) {
    return false;
  }
  const std::uint64_t bytes = promote.expected_bytes();
  if (!owner.terminal_execution_sliding_promote(
          sliding, promote, Status::success(), execution::TerminalKind::Known,
          promote.transfer_mask() != 0u, bytes) ||
      sliding.quiescent() ||
      !owner.release_execution_sliding_promote(sliding, std::move(promote)) ||
      promote) {
    return false;
  }
  work.physical_native = std::make_shared<execution::SlidingNative>();
  return work.physical_native != nullptr &&
         owner.issue_execution_sliding_native(
             plan, sliding, work.projection, work.uses, *work.physical_native);
}

[[nodiscard]] bool drain_bound(residency::Authority &authority,
                               const execution::Plan &plan,
                               execution::Sliding &sliding, EpochWork &work,
                               const bool persist) {
  auto owner = authority.sliding();
  if (work.physical_native == nullptr ||
      !owner.terminal_execution_sliding_native(
          sliding, *work.physical_native, Status::success(),
          execution::TerminalKind::Known, true) ||
      !owner.release_execution_sliding_native(
          sliding, std::move(*work.physical_native))) {
    return false;
  }
  std::size_t drained = 0u;
  for (std::size_t use = 0u; use < work.projection.use_count; ++use) {
    execution::SlidingDrain output{};
    if (!owner.issue_execution_sliding_drain(plan, sliding, work.projection,
                                              work.uses, use, output)) {
      continue;
    }
    if (!owner.terminal_execution_sliding_drain(
            sliding, output, Status::success(), execution::TerminalKind::Known,
            true, output.expected_bytes()) ||
        !owner.release_execution_sliding_drain(sliding, std::move(output))) {
      return false;
    }
    if (persist) {
      execution::SlidingPersist write{};
      if (!owner.issue_execution_sliding_persist(
              plan, sliding, work.projection, work.uses, use, write) ||
          !owner.terminal_execution_sliding_persist(
              sliding, write, Status::success(), execution::TerminalKind::Known,
              true, write.expected_bytes()) ||
          !owner.release_execution_sliding_persist(sliding, std::move(write))) {
        return false;
      }
    }
    ++drained;
  }
  return drained == work.projection.persist_count;
}

[[nodiscard]] bool drain(execution::Sliding &sliding, EpochWork &work,
                         const bool persist,
                         execution::SlidingTicket *retained) {
  if (!sliding.native_terminal(work.native, Status::success(),
                               execution::TerminalKind::Known, true)) {
    return false;
  }
  std::size_t drained = 0u;
  for (std::size_t use = 0u; use < work.projection.use_count; ++use) {
    execution::SlidingTicket output{};
    if (!sliding.issue_drain(work.native, use, output)) {
      continue;
    }
    if (!sliding.drain_terminal(output, Status::success(),
                                execution::TerminalKind::Known, true,
                                output.expected_bytes)) {
      return false;
    }
    if (retained != nullptr && drained == 0u) {
      *retained = output;
    }
    if (persist) {
      execution::SlidingTicket write{};
      if (!sliding.issue_persist(output, write) ||
          !sliding.persist_terminal(write, Status::success(),
                                    execution::TerminalKind::Known, true,
                                    write.expected_bytes)) {
        return false;
      }
    }
    ++drained;
  }
  return drained == work.projection.persist_count;
}

} // namespace rund_node_test_pipeline_residency::sliding_detail
