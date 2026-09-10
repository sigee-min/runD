#include "local.hpp"

#include "src/compute/device/residency/cycle/plan.hpp"
#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace rund_node_test_pipeline_residency {
namespace {

using namespace rund::compute::detail::residency::cycle;

[[nodiscard]] bool edge(const Plan &plan, const std::uint8_t before,
                        const std::uint8_t after) noexcept {
  for (const Edge candidate : plan.edges()) {
    if (candidate.before == before && candidate.after == after) {
      return true;
    }
  }
  return false;
}

} // namespace

[[nodiscard]] int CheckAuthorityCycleOwner() {
  using namespace rund::compute::detail::residency;
  Authority authority;
  std::uint32_t input_base = 99u;
  std::uint32_t output_base = 99u;
  if (!authority.register_frames(FrameTier::Device, FrameRole::Input, 4u,
                                 input_base) ||
      !authority.register_frames(FrameTier::Device, FrameRole::Output, 4u,
                                 output_base) ||
      input_base != 0u || output_base != 4u) {
    return 20;
  }
  const FrameRegion input_zero{.tier = FrameTier::Device,
                               .role = FrameRole::Input,
                               .first = input_base,
                               .count = 2u};
  const FrameRegion output_zero{.tier = FrameTier::Device,
                                .role = FrameRole::Output,
                                .first = output_base,
                                .count = 2u};
  const FrameRegion input_one{.tier = FrameTier::Device,
                              .role = FrameRole::Input,
                              .first = input_base + 2u,
                              .count = 2u};
  const FrameRegion output_one{.tier = FrameTier::Device,
                               .role = FrameRole::Output,
                               .first = output_base + 2u,
                               .count = 2u};
  const std::array input_uses{
      CacheUse{.key = {.backing = 201u, .version = 1u, .page = 0u},
               .access = Access::Read}};
  const std::array output_uses{
      CacheUse{.key = {.backing = 202u, .version = 1u, .page = 0u},
               .access = Access::Write,
               .dirty = {.offset = 0u, .bytes = 4u}}};
  const std::array next_input_uses{
      CacheUse{.key = {.backing = 201u, .version = 1u, .page = 1u},
               .access = Access::Read}};
  const std::array next_output_uses{
      CacheUse{.key = {.backing = 202u, .version = 1u, .page = 1u},
               .access = Access::Write,
               .dirty = {.offset = 0u, .bytes = 4u}}};
  const AuthorityResult first = authority.begin_transform(
      input_uses, input_zero, output_uses, output_zero);
  if (!first || !authority.activate(first.lease.token)) {
    return 21;
  }
  const AuthorityResult second = authority.begin_transform(
      next_input_uses, input_one, next_output_uses, output_one);
  if (!second || !authority.activate(second.lease.token)) {
    return 21;
  }
  const cycle::Flight first_flight{
      .ordinal = 0u,
      .token = first.lease.token,
      .bank = 0u,
      .input = {.domain = cycle::Domain::Device,
                .first = input_base,
                .count = 2u,
                .mask = 1u},
      .output = {.domain = cycle::Domain::Device,
                 .first = output_base,
                 .count = 2u,
                 .mask = 1u},
      .may_write = true};
  const cycle::Flight next_flight{
      .ordinal = 1u,
      .token = second.lease.token,
      .bank = 1u,
      .input = {.domain = cycle::Domain::Device,
                .first = input_base + 2u,
                .count = 2u,
                .mask = 1u},
      .output = {.domain = cycle::Domain::Device,
                 .first = output_base + 2u,
                 .count = 2u,
                 .mask = 1u},
      .may_write = true};
  CycleOwner cycles = authority.cycles();
  std::uint64_t cycle_token = 0u;
  if (!cycles.bind_cycle(first_flight, cycle_token) || cycle_token == 0u ||
      !cycles.advance_cycle(cycle_token, next_flight) ||
      !cycles.complete_cycle(cycle_token, first.lease.token, true) ||
      !cycles.complete_cycle(cycle_token, second.lease.token, true) ||
      !cycles.close_cycle(cycle_token) || !cycles.close_cycle(cycle_token)) {
    return 22;
  }
  return 0;
}

int CheckCycle() {
  const std::array<Epoch::Range, PhaseCapacity> reads{
      Epoch::Range{
          .domain = Domain::Host, .first = 100u, .count = 4u, .mask = 15u},
      Epoch::Range{.domain = Domain::HostVisible,
                   .first = 200u,
                   .count = 4u,
                   .mask = 15u},
      Epoch::Range{.domain = Domain::HostVisible,
                   .first = 300u,
                   .count = 4u,
                   .mask = 15u},
  };
  const std::array<Epoch::Range, PhaseCapacity> writes{
      Epoch::Range{.domain = Domain::HostVisible,
                   .first = 200u,
                   .count = 4u,
                   .mask = 15u},
      Epoch::Range{.domain = Domain::HostVisible,
                   .first = 300u,
                   .count = 4u,
                   .mask = 15u},
      Epoch::Range{.domain = Domain::HostVisible,
                   .first = 300u,
                   .count = 4u,
                   .mask = 15u},
  };
  const std::array<Epoch, 3u> epochs{
      Epoch{.ordinal = 8u,
            .bank = 0u,
            .tokens = {11u, 11u, 12u},
            .reads = reads,
            .writes = writes,
            .may_write = {false, true, false}},
      Epoch{.ordinal = 9u,
            .bank = 1u,
            .tokens = {21u, 21u, 22u},
            .reads = reads,
            .writes = writes,
            .may_write = {false, true, false}},
      Epoch{.ordinal = 10u,
            .bank = 0u,
            .tokens = {31u, 31u, 32u},
            .reads = reads,
            .writes = writes,
            .may_write = {false, true, false}},
  };
  const Result result = seal(epochs);
  if (!result || result.plan.nodes().size() != NodeCapacity ||
      result.plan.edges().size() != 7u) {
    return 1;
  }
  const auto nodes = result.plan.nodes();
  if (nodes[0].phase != Phase::Upload || nodes[0].source != Domain::Host ||
      nodes[0].target != Domain::HostVisible ||
      nodes[1].phase != Phase::Dispatch ||
      nodes[1].source != Domain::HostVisible ||
      nodes[1].target != Domain::HostVisible ||
      nodes[2].phase != Phase::Download ||
      nodes[2].source != Domain::HostVisible ||
      nodes[2].target != Domain::HostVisible || !nodes[1].may_write ||
      nodes[0].may_write || nodes[2].may_write || nodes[1].write != writes[1]) {
    return 2;
  }
  if (!edge(result.plan, 0u, 1u) || !edge(result.plan, 1u, 2u) ||
      !edge(result.plan, 3u, 4u) || !edge(result.plan, 4u, 5u) ||
      !edge(result.plan, 6u, 7u) || !edge(result.plan, 7u, 8u) ||
      !edge(result.plan, 2u, 7u) || edge(result.plan, 2u, 6u) ||
      edge(result.plan, 2u, 3u)) {
    return 3;
  }

  const Result one = seal(std::span<const Epoch>{epochs.data(), 1u});
  if (!one || one.plan.nodes().size() != 3u || one.plan.edges().size() != 2u) {
    return 4;
  }
  if (seal(std::span<const Epoch>{}).failure != Failure::Invalid) {
    return 5;
  }
  const std::array<Epoch, 4u> too_many{epochs[0], epochs[1], epochs[2],
                                       epochs[0]};
  if (seal(too_many).failure != Failure::Capacity) {
    return 6;
  }
  auto invalid = epochs;
  invalid[1].ordinal = 11u;
  if (seal(invalid).failure != Failure::Invalid) {
    return 7;
  }
  invalid = epochs;
  invalid[1].bank = 0u;
  if (seal(invalid).failure != Failure::Invalid) {
    return 8;
  }
  invalid = epochs;
  invalid[1].tokens[0] = 0u;
  if (seal(invalid).failure != Failure::Invalid) {
    return 9;
  }
  invalid = epochs;
  invalid[1].tokens[0] = invalid[0].tokens[2];
  if (seal(invalid).failure != Failure::Invalid) {
    return 10;
  }
  invalid = epochs;
  invalid[1].may_write[index(Phase::Dispatch)] = false;
  if (seal(invalid).failure != Failure::Invalid) {
    return 11;
  }
  const std::array<Epoch, 2u> overflow{
      Epoch{.ordinal = std::numeric_limits<std::uint64_t>::max(),
            .bank = 1u,
            .tokens = {41u, 41u, 42u},
            .reads = reads,
            .writes = writes,
            .may_write = {false, true, false}},
      Epoch{.ordinal = 0u,
            .bank = 0u,
            .tokens = {51u, 51u, 52u},
            .reads = reads,
            .writes = writes,
            .may_write = {false, true, false}},
  };
  if (seal(overflow).failure != Failure::Overflow) {
    return 12;
  }
  return CheckAuthorityCycleOwner();
}

} // namespace rund_node_test_pipeline_residency
