#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "identity.hpp"
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
[[nodiscard]] int
CheckCount(rund::compute::Device &device, const rund::compute::Backend backend,
           const Seed &seed, const Action &action, const Fold &fold,
           const std::uint32_t count_value) {
  using namespace rund::compute;
  const std::array<std::uint32_t, 1u> initial{kOuterSeed};
  const std::array<std::uint32_t, 1u> count_values{count_value};
  const std::array<std::uint32_t, 1u> output_values{kSentinel};
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{output_values});
  if (!outer || !queue || !domain || !count || !output) {
    return 1;
  }

  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer, *queue, *domain),
                                   write_final(*output));
  const auto plan = builder.plan();
  if (!plan || !NestedPlanShape(*plan, seed, action, fold)) {
    if (plan) {
      std::fprintf(
          stderr,
          "nested plan backend=%u count=%u outer=%llu tile=%llu inner=%llu "
          "templates=%llu commands=%llu barriers=%llu resources=%llu "
          "state=%llu "
          "logical/live/physical=%llu/%llu/%llu\n",
          static_cast<unsigned>(backend), count_value,
          static_cast<unsigned long long>(plan->outer_window_count),
          static_cast<unsigned long long>(plan->tile_capacity),
          static_cast<unsigned long long>(plan->inner_iteration_count),
          static_cast<unsigned long long>(plan->prepared_template_count),
          static_cast<unsigned long long>(plan->prepared_command_count),
          static_cast<unsigned long long>(plan->barrier_count),
          static_cast<unsigned long long>(plan->resource_count),
          static_cast<unsigned long long>(plan->state_bytes),
          static_cast<unsigned long long>(plan->logical_bytes),
          static_cast<unsigned long long>(plan->live_bytes),
          static_cast<unsigned long long>(plan->physical_bytes));
    } else {
      std::fprintf(stderr,
                   "nested plan rejected backend=%u count=%u reason=%u\n",
                   static_cast<unsigned>(backend), count_value,
                   static_cast<unsigned>(plan.reason()));
    }
    return 2;
  }

  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *plan || !PreparedShape(*prepared)) {
    std::fprintf(stderr,
                 "nested prepare backend=%u count=%u status=%u reason=%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()));
    return 3;
  }
  if (backend == Backend::Cpu && count_value == kCounts.front() &&
      !CpuRouteOwnershipIsExact(*prepared)) {
    std::fprintf(stderr, "nested CPU route ownership mismatch\n");
    return 7;
  }
  const PipelineBindingIdentity frozen_bindings =
      CaptureBindingIdentity(*prepared, backend);
  if (!frozen_bindings.valid) {
    std::fprintf(
        stderr,
        "nested frozen bindings backend=%u count=%u valid=0 "
        "transactional=%u prepared=%u:%p alternate=%u:%p jobs=%llu/%llu/%llu\n",
        static_cast<unsigned>(backend), count_value,
        static_cast<unsigned>(frozen_bindings.transactional),
        static_cast<unsigned>(frozen_bindings.prepared_ok),
        frozen_bindings.prepared_owner,
        static_cast<unsigned>(frozen_bindings.alternate_prepared_ok),
        frozen_bindings.alternate_prepared_owner,
        static_cast<unsigned long long>(frozen_bindings.normal_jobs.size()),
        static_cast<unsigned long long>(frozen_bindings.alternate_jobs.size()),
        static_cast<unsigned long long>(frozen_bindings.jobs.size()));
    return 3;
  }

  if (count_value > kMaximum) {
    const Status failed = prepared->run();
    const Stats stats = prepared->stats();
    const bool bindings_unchanged =
        CaptureBindingIdentity(*prepared, backend) == frozen_bindings;
    const std::uint64_t submits = backend == Backend::Cpu ? 0u : 1u;
    if (failed || failed.reason() != Reason::BoundedCountInvalid ||
        prepared->generation() != 0u || stats.command_submits != submits ||
        stats.pipeline.verified_step_count != 0u ||
        stats.pipeline.failed_step_index != 0u ||
        stats.pipeline.failed_nested_phase != PipelineNestedPhase::Seed ||
        stats.pipeline.failed_outer_window != 0u ||
        stats.pipeline.failed_inner_iteration != PipelineStats::no_coordinate ||
        stats.pipeline.executed_outer_window_count != 0u ||
        stats.pipeline.executed_inner_iteration_count != 0u ||
        stats.control.overflow_ordinal != kMaximum || !bindings_unchanged) {
      std::fprintf(
          stderr,
          "nested overflow backend=%u status=%u reason=%u generation=%llu "
          "submits=%llu verified=%llu failed=%llu phase=%u "
          "coordinate=%llu/%llu "
          "outer=%llu inner=%llu ordinal=%llu bindings=%u\n",
          static_cast<unsigned>(backend), static_cast<unsigned>(failed.ok()),
          static_cast<unsigned>(failed.reason()),
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned long long>(stats.command_submits),
          static_cast<unsigned long long>(stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(stats.pipeline.failed_step_index),
          static_cast<unsigned>(stats.pipeline.failed_nested_phase),
          static_cast<unsigned long long>(stats.pipeline.failed_outer_window),
          static_cast<unsigned long long>(
              stats.pipeline.failed_inner_iteration),
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(stats.control.overflow_ordinal),
          static_cast<unsigned>(bindings_unchanged));
      return 4;
    }
    auto observer = on(device)
                        .map<std::uint32_t>("nested-window-observe", 1u,
                                            [](auto value) { return value; })
                        .compile();
    auto scratch = device.buffer<std::uint32_t>(1u);
    std::array<std::uint32_t, 1u> actual{};
    if (!observer || !scratch ||
        !Observe(*observer, *output, *scratch, actual) ||
        actual != output_values) {
      std::fprintf(stderr,
                   "nested overflow publication backend=%u actual=%u "
                   "expected=%u\n",
                   static_cast<unsigned>(backend), actual[0], kSentinel);
      return 5;
    }
    return 0;
  }

  const Status first = prepared->run();
  std::array<std::uint32_t, 1u> first_output{};
  const Status first_read = prepared->read(*output, first_output);
  const bool first_bindings_unchanged =
      CaptureBindingIdentity(*prepared, backend) == frozen_bindings;
  if (!first || !first_read || !first_bindings_unchanged ||
      first_output[0] != SerialOracle(count_value)) {
    std::fprintf(stderr,
                 "nested first backend=%u count=%u status=%u reason=%u "
                 "actual=%u expected=%u bindings=%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(first.ok()),
                 static_cast<unsigned>(first.reason()), first_output[0],
                 SerialOracle(count_value),
                 static_cast<unsigned>(first_bindings_unchanged));
    return 6;
  }

  const MemoryStats before = prepared->memory();
  const Status warm = prepared->run();
  const Stats warm_stats = prepared->stats();
  const MemoryStats after = prepared->memory();
  std::array<std::uint32_t, 1u> warm_output{};
  const Status warm_read = prepared->read(*output, warm_output);
  const bool warm_bindings_unchanged =
      CaptureBindingIdentity(*prepared, backend) == frozen_bindings;
  if (!warm || !RuntimeShape(warm_stats, backend, count_value) ||
      !rund_node_test_pipeline::SameMemory(before, after) ||
      prepared->generation() != 2u || !warm_read || !warm_bindings_unchanged ||
      warm_output[0] != SerialOracle(count_value)) {
    std::fprintf(
        stderr,
        "nested warm backend=%u count=%u status=%u reason=%u output=%u/%u "
        "generation=%llu outer=%llu/%llu inner=%llu/%llu "
        "templates=%llu commands=%llu dispatches=%llu control=%llu "
        "iterations=%llu/%llu alloc=%llu compile=%llu bindings=%u\n",
        static_cast<unsigned>(backend), count_value,
        static_cast<unsigned>(warm.ok()), static_cast<unsigned>(warm.reason()),
        warm_output[0], SerialOracle(count_value),
        static_cast<unsigned long long>(prepared->generation()),
        static_cast<unsigned long long>(
            warm_stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.prepared_template_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(warm_stats.dispatches),
        static_cast<unsigned long long>(
            warm_stats.pipeline.control_command_count),
        static_cast<unsigned long long>(warm_stats.control.iteration_count),
        static_cast<unsigned long long>(
            warm_stats.control.skipped_iteration_count),
        static_cast<unsigned long long>(warm_stats.buffer_allocations),
        static_cast<unsigned long long>(warm_stats.pipeline_compiles),
        static_cast<unsigned>(warm_bindings_unchanged));
    return 7;
  }
  return 0;
}

[[nodiscard]] int CheckNestedTerminal(rund::compute::Device &device,
                                      const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto seed = MakeNestedTerminalSeedProgram(device);
  auto action = on(device)
                    .input<std::uint32_t>(1u)
                    .branch([](auto value) {
                      auto index =
                          value.map("nested-window-terminal-action-index",
                                    [](auto current) { return current * 0u; });
                      return value.gather(index).scalar().map(
                          "nested-window-terminal-action",
                          [](auto current) { return current + 1u; });
                    })
                    .compile();
  auto fold = MakeNestedTerminalFoldProgram(device);
  if (!seed || !action || !fold) {
    return 1;
  }

  const auto run_case = [&]<std::size_t Maximum, std::size_t Tile>(
                            const std::uint32_t initial_terminal,
                            const std::uint32_t count_value,
                            const std::uint64_t active, Stats *const observed) {
    constexpr std::array<std::uint32_t, 1u> outer_values{kOuterSeed};
    const std::array<std::uint32_t, 1u> count_values{count_value};
    const std::array<std::uint32_t, 1u> terminal_values{initial_terminal};
    auto outer = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{outer_values});
    auto terminal = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{terminal_values});
    auto count = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{count_values});
    auto output = device.buffer<std::uint32_t>(1u);
    auto stopped = device.buffer<std::uint32_t>(1u);
    if (!outer || !terminal || !count || !output || !stopped) {
      return 1;
    }
    const auto body = tile_repeat<kInner>(*seed, *action, *fold);
    auto builder = pipeline(device);
    builder.windows<Maximum, Tile>(
        body, rund::compute::window(*count).until<1u>(7u),
        read(*outer, *terminal), write_final(*output, *stopped));
    const auto plan = builder.plan();
    if (!plan) {
      return 2;
    }
    auto prepared = std::move(builder)
                        .budget(MemoryBudget{.bytes = plan->peak_bytes})
                        .prepare();
    std::array<std::uint32_t, 1u> output_actual{};
    std::array<std::uint32_t, 1u> terminal_actual{};
    const Status ran =
        prepared ? prepared->run() : Status::fail(prepared.reason());
    const Stats stats = prepared ? prepared->stats() : Stats{};
    constexpr std::uint64_t outer_count = CeilDiv(Maximum, Tile);
    const std::uint32_t expected_output =
        active == 0u ? kOuterSeed : kOuterSeed + 1u + kInner;
    if (!prepared || !ran || prepared->generation() != 1u ||
        !prepared->read(*output, output_actual) ||
        !prepared->read(*stopped, terminal_actual) ||
        output_actual[0] != expected_output || terminal_actual[0] != 7u ||
        stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
        stats.pipeline.verified_step_count != 1u ||
        stats.pipeline.failed_step_index != PipelineStats::no_failed_step ||
        stats.pipeline.executed_outer_window_count != active ||
        stats.pipeline.skipped_outer_window_count != outer_count - active ||
        stats.pipeline.executed_inner_iteration_count != active * kInner ||
        stats.pipeline.skipped_inner_iteration_count !=
            (outer_count - active) * kInner ||
        stats.control.iteration_count != active ||
        stats.control.skipped_iteration_count != outer_count - active) {
      std::fprintf(
          stderr,
          "nested terminal backend=%u initial=%u prepared=%u status=%u "
          "reason=%u generation=%llu output=%u/%u terminal=%u "
          "outer=%llu/%llu inner=%llu/%llu control=%llu/%llu "
          "verified=%llu failed=%llu submits=%llu\n",
          static_cast<unsigned>(backend), initial_terminal,
          static_cast<unsigned>(prepared.ok()), static_cast<unsigned>(ran.ok()),
          static_cast<unsigned>(ran.reason()),
          static_cast<unsigned long long>(prepared ? prepared->generation()
                                                   : 0u),
          output_actual[0], expected_output, terminal_actual[0],
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              stats.pipeline.skipped_outer_window_count),
          static_cast<unsigned long long>(
              stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(
              stats.pipeline.skipped_inner_iteration_count),
          static_cast<unsigned long long>(stats.control.iteration_count),
          static_cast<unsigned long long>(
              stats.control.skipped_iteration_count),
          static_cast<unsigned long long>(stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(stats.command_submits));
      return 3;
    }
    if (observed != nullptr) {
      *observed = stats;
    }
    return 0;
  };

  const int initial =
      run_case.template operator()<kMaximum, kTile>(7u, kMaximum, 0u, nullptr);
  if (initial != 0) {
    return 10 + initial;
  }
  Stats produced_stats{};
  const int produced = run_case.template operator()<kMaximum, kTile>(
      0u, kMaximum, 1u, &produced_stats);
  if (produced != 0) {
    return 20 + produced;
  }
  // Two outer banks make the canonical terminal bank `second`. Stopping after
  // the first Fold leaves `first` current, so successful publication requires
  // the explicit current -> final seal before final -> target publication.
  const int sealed_early =
      run_case.template operator()<6u, 3u>(0u, 6u, 1u, nullptr);
  if (sealed_early != 0) {
    return 25 + sealed_early;
  }
  Stats single_stats{};
  const int single =
      run_case.template operator()<kTile, kTile>(0u, kTile, 1u, &single_stats);
  if (single != 0) {
    return 30 + single;
  }
  const ControlStats &produced_control = produced_stats.control;
  const ControlStats &single_control = single_stats.control;
  if (produced_control.generated_item_count !=
          single_control.generated_item_count ||
      produced_control.generated_capacity !=
          single_control.generated_capacity ||
      produced_control.indirect_dispatch_count !=
          single_control.indirect_dispatch_count ||
      produced_control.indirect_work_item_count !=
          single_control.indirect_work_item_count ||
      produced_control.conflict_count != single_control.conflict_count ||
      produced_control.overflow_ordinal != single_control.overflow_ordinal) {
    std::fprintf(
        stderr,
        "nested terminal payload backend=%u generated=%llu/%llu "
        "capacity=%llu/%llu indirect=%llu/%llu work=%llu/%llu "
        "conflict=%llu/%llu overflow=%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(produced_control.generated_item_count),
        static_cast<unsigned long long>(single_control.generated_item_count),
        static_cast<unsigned long long>(produced_control.generated_capacity),
        static_cast<unsigned long long>(single_control.generated_capacity),
        static_cast<unsigned long long>(
            produced_control.indirect_dispatch_count),
        static_cast<unsigned long long>(single_control.indirect_dispatch_count),
        static_cast<unsigned long long>(
            produced_control.indirect_work_item_count),
        static_cast<unsigned long long>(
            single_control.indirect_work_item_count),
        static_cast<unsigned long long>(produced_control.conflict_count),
        static_cast<unsigned long long>(single_control.conflict_count),
        static_cast<unsigned long long>(produced_control.overflow_ordinal),
        static_cast<unsigned long long>(single_control.overflow_ordinal));
    return 40;
  }
  return 0;
}

int CheckNestedCount(rund::compute::Device &device,
                     const rund::compute::Backend backend,
                     const NestedSeed &seed, const NestedAction &action,
                     const NestedFold &fold, const std::uint32_t count_value) {
  return CheckCount(device, backend, seed, action, fold, count_value);
}

} // namespace rund::node::test_contract::window
