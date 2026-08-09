#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/prepared.hpp"
#include "src/compute/cpu/run/state.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {
template <class Seed, class Action, class Fold>
[[nodiscard]] int CheckComposition(rund::compute::Device &device,
                                   const rund::compute::Backend backend,
                                   const Seed &seed, const Action &action,
                                   const Fold &fold) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{kMaximum};
  auto increment =
      on(device)
          .map<std::uint32_t>("nested-window-downstream-repeat", 1u,
                              [](auto value) { return value + 1u; })
          .compile();
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto nested_output = device.buffer<std::uint32_t>(1u);
  auto derived = device.buffer<std::uint32_t>(1u);
  if (!increment || !outer || !queue || !domain || !count || !nested_output ||
      !derived) {
    return 1;
  }

  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *queue, *domain),
                                write_final(*nested_output))
      .template repeat<2u>(*increment, read(*nested_output),
                           write_final(*derived));
  const auto plan = builder.plan();
  const std::uint64_t logical_workspace =
      kOuter * (seed.graph().memory.logical_bytes +
                kInner * action.graph().memory.logical_bytes +
                fold.graph().memory.logical_bytes) +
      2u * increment->graph().memory.logical_bytes;
  const std::uint64_t live_workspace = std::max(
      {seed.graph().memory.live_bytes, action.graph().memory.live_bytes,
       fold.graph().memory.live_bytes, increment->graph().memory.live_bytes});
  if (!plan || plan->prepared_template_count != kPreparedTemplates + 2u ||
      plan->prepared_command_count != kCommands + 2u ||
      plan->barrier_count != kTemplates + 1u ||
      plan->logical_bytes !=
          plan->state_bytes + plan->prepared_bytes + logical_workspace ||
      plan->live_bytes !=
          plan->state_bytes + plan->prepared_bytes + live_workspace ||
      plan->physical_bytes !=
          plan->state_bytes + plan->prepared_bytes + plan->transient_bytes ||
      plan->physical_bytes != plan->peak_bytes) {
    std::fprintf(
        stderr,
        "nested composition plan backend=%u status=%u reason=%u "
        "templates=%llu commands=%llu barriers=%llu "
        "logical/live/physical=%llu/%llu/%llu workspace=%llu/%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(plan.ok()),
        static_cast<unsigned>(plan.reason()),
        static_cast<unsigned long long>(plan ? plan->prepared_template_count
                                             : 0u),
        static_cast<unsigned long long>(plan ? plan->prepared_command_count
                                             : 0u),
        static_cast<unsigned long long>(plan ? plan->barrier_count : 0u),
        static_cast<unsigned long long>(plan ? plan->logical_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->live_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->physical_bytes : 0u),
        static_cast<unsigned long long>(logical_workspace),
        static_cast<unsigned long long>(live_workspace));
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  std::array<std::uint32_t, 1u> nested_actual{};
  std::array<std::uint32_t, 1u> derived_actual{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  if (!prepared || !ran || !prepared->read(*nested_output, nested_actual) ||
      !prepared->read(*derived, derived_actual) ||
      nested_actual[0] != SerialOracle(kMaximum) ||
      derived_actual[0] != SerialOracle(kMaximum) + 2u ||
      stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.barrier_count != plan->barrier_count ||
      stats.pipeline.executed_outer_window_count != kOuter ||
      stats.pipeline.executed_inner_iteration_count != kOuter * kInner ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested composition run backend=%u prepared=%u status=%u reason=%u "
        "nested=%u/%u derived=%u/%u steps=%llu/%llu outer=%llu inner=%llu "
        "submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        nested_actual[0], SerialOracle(kMaximum), derived_actual[0],
        SerialOracle(kMaximum) + 2u,
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 3;
  }
  return 0;
}

[[nodiscard]] int CheckProductPlan(rund::compute::Device &device,
                                   const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t maximum = 33u;
  constexpr std::size_t tile = 1u;
  constexpr std::size_t inner = 33u;
  constexpr std::size_t outer = CeilDiv(maximum, tile);
  constexpr std::size_t routes = outer + inner + 3u;
  constexpr std::size_t prepared_templates = outer + 2u + 3u;
  constexpr std::size_t commands = outer * (inner + 2u);
  static_assert(outer * inner > PipelineIterationCapacity);
  static_assert(routes < PipelineIterationCapacity);

  auto seed = MakeNestedProductSeedProgram(device);
  auto action = MakeNestedActionProgram(device);
  auto fold = MakeNestedFoldProgram(device);
  std::array<std::uint32_t, maximum> queue_values{};
  for (std::size_t index = 0u; index < queue_values.size(); ++index) {
    queue_values[index] = static_cast<std::uint32_t>(index + 1u);
  }
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{maximum};
  auto outer_buffer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{queue_values});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.buffer<std::uint32_t>(1u);
  if (!seed || !action || !fold || !outer_buffer || !queue || !domain ||
      !count || !output) {
    return 1;
  }

  const auto body = tile_repeat<inner>(*seed, *action, *fold);
  auto builder = pipeline(device);
  builder.windows<maximum, tile>(body, rund::compute::window(*count),
                                 read(*outer_buffer, *queue, *domain),
                                 write_final(*output));
  const auto plan = builder.plan();
  constexpr std::uint64_t state_bytes = (outer + 5u) * sizeof(std::uint32_t);
  if (!plan || plan->outer_window_count != outer ||
      plan->tile_capacity != tile || plan->inner_iteration_count != inner ||
      plan->prepared_template_count != prepared_templates ||
      plan->prepared_command_count != commands ||
      plan->state_bytes != state_bytes || plan->resource_count != 11u ||
      plan->physical_bytes != plan->peak_bytes ||
      plan->logical_bytes <= plan->physical_bytes) {
    if (plan) {
      std::fprintf(
          stderr,
          "nested product plan outer=%llu tile=%llu inner=%llu "
          "templates=%llu commands=%llu state=%llu resources=%llu "
          "logical/physical=%llu/%llu\n",
          static_cast<unsigned long long>(plan->outer_window_count),
          static_cast<unsigned long long>(plan->tile_capacity),
          static_cast<unsigned long long>(plan->inner_iteration_count),
          static_cast<unsigned long long>(plan->prepared_template_count),
          static_cast<unsigned long long>(plan->prepared_command_count),
          static_cast<unsigned long long>(plan->state_bytes),
          static_cast<unsigned long long>(plan->resource_count),
          static_cast<unsigned long long>(plan->logical_bytes),
          static_cast<unsigned long long>(plan->physical_bytes));
    }
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  std::array<std::uint32_t, 1u> actual{};
  std::uint32_t expected = initial[0];
  for (const std::uint32_t item : queue_values) {
    expected += kDomainValues[item] + static_cast<std::uint32_t>(inner);
  }
  const bool unexpected_direct = backend == Backend::Metal &&
                                 stats.dispatches == 2u &&
                                 stats.pipeline.control_command_count == 1u;
  if (!prepared || !ran || !prepared->read(*output, actual) ||
      actual[0] != expected || unexpected_direct ||
      stats.pipeline.executed_outer_window_count != outer ||
      stats.pipeline.executed_inner_iteration_count != outer * inner ||
      stats.pipeline.prepared_command_count != commands ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested product fallback backend=%u prepared=%u status=%u/%u "
        "output=%u/%u dispatches=%llu control=%llu outer=%llu inner=%llu "
        "commands=%llu submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        actual[0], expected, static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(stats.pipeline.control_command_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 3;
  }
  return 0;
}

int CheckNestedComposition(rund::compute::Device &device,
                           const rund::compute::Backend backend,
                           const NestedSeed &seed, const NestedAction &action,
                           const NestedFold &fold) {
  return CheckComposition(device, backend, seed, action, fold);
}

} // namespace rund::node::test_contract::window
