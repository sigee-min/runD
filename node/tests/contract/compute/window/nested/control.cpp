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
template <std::size_t Maximum, std::size_t Tile>
[[nodiscard]] int CheckDormantAggregateRouteCase(
    rund::compute::Device &device,
    const std::array<std::uint64_t, 3u> expected_fold_occurrences,
    const bool expect_direct) {
  using namespace rund::compute;
  constexpr std::size_t outer_count = CeilDiv(Maximum, Tile);
  constexpr std::size_t template_count = outer_count + kInner + 3u;
  std::array<std::uint32_t, Maximum> queue_values{};
  std::copy_n(kQueue.begin(), Maximum, queue_values.begin());
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{
      static_cast<std::uint32_t>(Maximum)};

  auto seed = [&] {
    if constexpr (Maximum == 4u) {
      return MakeNestedSeed4Program(device);
    }
    return MakeNestedSeed8Program(device);
  }();
  auto action = MakeNestedActionProgram(device);
  auto fold = MakeNestedFoldProgram(device);
  auto outer = device.upload<std::uint32_t>(initial);
  auto queue = device.upload<std::uint32_t>(queue_values);
  auto domain = device.upload<std::uint32_t>(kDomainValues);
  auto count = device.upload<std::uint32_t>(count_values);
  if (!seed || !action || !fold || !outer || !queue || !domain || !count) {
    return 1;
  }

  std::uint32_t expected = kOuterSeed;
  for (const std::uint32_t item : queue_values) {
    expected += kDomainValues[item] + static_cast<std::uint32_t>(kInner);
  }
  for (const bool profile_steps : {false, true}) {
    auto output = device.buffer<std::uint32_t>(1u);
    if (!output) {
      return 2;
    }
    const auto body = tile_repeat<kInner>(*seed, *action, *fold);
    auto builder = pipeline(device);
    if (profile_steps) {
      builder.profile(PipelineProfile::Steps);
    }
    builder.windows<Maximum, Tile>(body, rund::compute::window(*count),
                                   read(*outer, *queue, *domain),
                                   write_final(*output));
    const auto plan = builder.plan();
    auto prepared = plan ? std::move(builder)
                               .budget(MemoryBudget{.bytes = plan->peak_bytes})
                               .prepare()
                         : Result<Pipeline>::fail(plan.reason());
    const Status ran =
        prepared ? prepared->run() : Status::fail(prepared.reason());
    const Stats stats = prepared ? prepared->stats() : Stats{};
    const bool direct =
        stats.dispatches == 2u && stats.pipeline.control_command_count == 1u;
    std::array<std::uint32_t, 1u> actual{};
    if (!plan || !prepared || !ran || !prepared->read(*output, actual) ||
        actual[0u] != expected || direct != expect_direct ||
        stats.pipeline.executed_outer_window_count != outer_count ||
        stats.pipeline.executed_inner_iteration_count != outer_count * kInner ||
        stats.command_submits != 1u) {
      std::fprintf(
          stderr,
          "nested dormant aggregate max=%llu tile=%llu profile=%u "
          "plan=%u prepared=%u run=%u/%u output=%u/%u dispatches=%llu "
          "control=%llu direct=%u/%u outer=%llu inner=%llu submits=%llu\n",
          static_cast<unsigned long long>(Maximum),
          static_cast<unsigned long long>(Tile),
          static_cast<unsigned>(profile_steps),
          static_cast<unsigned>(plan.ok()),
          static_cast<unsigned>(prepared.ok()), static_cast<unsigned>(ran.ok()),
          static_cast<unsigned>(ran.reason()), actual[0u], expected,
          static_cast<unsigned long long>(stats.dispatches),
          static_cast<unsigned long long>(stats.pipeline.control_command_count),
          static_cast<unsigned>(direct), static_cast<unsigned>(expect_direct),
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(stats.command_submits));
      return 3;
    }
    if (!profile_steps) {
      continue;
    }
    std::array<PipelineStepProfile, template_count> rows{};
    const auto profile = prepared->profile(rows);
    if (!profile || profile->written != rows.size() ||
        profile->total != rows.size()) {
      return 4;
    }
    const std::size_t fold_first = outer_count + kInner;
    for (std::size_t route = 0u; route < expected_fold_occurrences.size();
         ++route) {
      const std::uint64_t original =
          rows[fold_first + route].execution.original_dispatches;
      const bool expected_active = expected_fold_occurrences[route] != 0u;
      if ((original != 0u) != expected_active) {
        std::fprintf(
            stderr,
            "nested dormant aggregate profile max=%llu route=%llu "
            "occurrences=%llu original=%llu\n",
            static_cast<unsigned long long>(Maximum),
            static_cast<unsigned long long>(route),
            static_cast<unsigned long long>(expected_fold_occurrences[route]),
            static_cast<unsigned long long>(original));
        return 5;
      }
    }
  }
  return 0;
}

[[nodiscard]] int
CheckDormantAggregateRoutes(rund::compute::Device &device,
                            const rund::compute::Backend backend) {
  if (backend != rund::compute::Backend::Metal) {
    return 0;
  }
  if (const int result =
          CheckDormantAggregateRouteCase<4u, 4u>(device, {1u, 0u, 0u}, false);
      result != 0) {
    return 10 + result;
  }
  const int result =
      CheckDormantAggregateRouteCase<8u, 4u>(device, {1u, 1u, 0u}, true);
  return result == 0 ? 0 : 20 + result;
}

[[nodiscard]] int
CheckWorkspaceObservationCapacity(rund::compute::Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::size_t window_count = 22u;
  constexpr std::size_t rows_per_window = 5u;
  constexpr std::size_t row_count = window_count * rows_per_window;
  static_assert(3u * window_count > PipelineStepCapacity);
  static_assert(row_count < PipelineRouteCapacity);

  auto seed = on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto total, auto ordinal) {
                    (void)total;
                    return ordinal.scan(Scan::InclusiveSum)
                        .map("workspace-capacity-seed",
                             [](auto value) { return value + 1u; });
                  })
                  .compile();
  auto action = on(device)
                    .input<std::uint32_t>(1u)
                    .branch([](auto value) {
                      return value.scan(Scan::InclusiveSum)
                          .map("workspace-capacity-action",
                               [](auto current) { return current + 1u; });
                    })
                    .compile();
  auto fold = on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto outer, auto tile) {
                    auto prefix = tile.scan(Scan::InclusiveSum);
                    return outer.combine(
                        "workspace-capacity-fold", prefix,
                        [](auto left, auto right) { return left + right; });
                  })
                  .compile();
  constexpr std::array<std::uint32_t, 1u> initial{1u};
  constexpr std::array<std::uint32_t, 1u> inactive{0u};
  auto outer = device.upload<std::uint32_t>(initial);
  auto count = device.upload<std::uint32_t>(inactive);
  if (!seed || !action || !fold || !outer || !count) {
    return 1;
  }
  std::vector<rund::compute::Buffer<std::uint32_t>> outputs;
  outputs.reserve(window_count);
  for (std::size_t index = 0u; index < window_count; ++index) {
    auto output = device.buffer<std::uint32_t>(1u);
    if (!output) {
      return 1;
    }
    outputs.push_back(std::move(*output));
  }

  const auto body = tile_repeat<1u>(*seed, *action, *fold);
  auto builder = pipeline(device).profile(PipelineProfile::Steps);
  for (std::size_t index = 0u; index < window_count; ++index) {
    builder.windows<1u, 1u>(body, rund::compute::window(*count), read(*outer),
                            write_final(outputs[index]));
  }
  auto prepared = std::move(builder).prepare();
  const std::shared_ptr<PipelineState> state =
      prepared ? PipelineStateAccess::state(*prepared)
               : std::shared_ptr<PipelineState>{};
  if (!prepared || state == nullptr || state->steps.size() != row_count ||
      !prepared->run()) {
    return 2;
  }

  std::array<const JobWorkspace *, row_count> workspaces{};
  std::size_t workspace_count = 0u;
  for (const PipelineStep &step : state->steps) {
    if (step.job == nullptr || step.job->workspace == nullptr) {
      return 3;
    }
    const JobWorkspace *const workspace = step.job->workspace.get();
    if (std::find(workspaces.begin(), workspaces.begin() + workspace_count,
                  workspace) == workspaces.begin() + workspace_count) {
      workspaces[workspace_count++] = workspace;
    }
  }
  std::array<PipelineStepProfile, row_count> rows{};
  const auto profile = prepared->profile(rows);
  if (workspace_count <= PipelineStepCapacity || !profile ||
      profile->written != row_count || profile->total != row_count ||
      !::rund_node_test_pipeline::ProfileMemoryReconciles(*profile, rows) ||
      profile->memory.host.current ==
          std::numeric_limits<std::uint64_t>::max() ||
      profile->memory.resident.current ==
          std::numeric_limits<std::uint64_t>::max()) {
    return 4;
  }
  return 0;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] int CheckAggregateSeedFailures(
    rund::compute::Device &device, const rund::compute::Backend backend,
    const Seed &seed, const Action &action, const Fold &fold) {
  using namespace rund::compute;
  const auto same_execution = [](const PipelineStepStats &left,
                                 const PipelineStepStats &right) {
    return left.sample_count == right.sample_count &&
           left.original_dispatches == right.original_dispatches &&
           left.final_dispatches == right.final_dispatches &&
           left.barrier_count == right.barrier_count &&
           left.worker_count == right.worker_count &&
           left.participating_workers == right.participating_workers &&
           left.tile_count == right.tile_count &&
           left.tile_size == right.tile_size &&
           left.vector_chunks == right.vector_chunks &&
           left.tail_chunks == right.tail_chunks &&
           left.workgroup_count == right.workgroup_count &&
           left.work_item_count == right.work_item_count &&
           rund_node_test_pipeline::SameControlStats(left.control,
                                                     right.control);
  };
  const auto run_case = [&](const bool reduce_overflow) {
    constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
    constexpr std::array<std::uint32_t, 1u> output_values{kSentinel};
    const std::array<std::uint32_t, 1u> count_values{reduce_overflow ? 6u : 5u};
    std::array<std::uint32_t, kMaximum> queue_values{kQueue};
    std::array<std::uint32_t, kDomain> domain_values{kDomainValues};
    if (reduce_overflow) {
      domain_values[kQueue[4u]] = std::numeric_limits<std::uint32_t>::max();
      domain_values[kQueue[5u]] = 1u;
    } else {
      queue_values[4u] = static_cast<std::uint32_t>(kDomain);
    }

    auto outer = device.upload<std::uint32_t>(initial);
    auto queue = device.upload<std::uint32_t>(queue_values);
    auto domain = device.upload<std::uint32_t>(domain_values);
    auto count = device.upload<std::uint32_t>(count_values);
    auto output = device.upload<std::uint32_t>(output_values);
    auto observer =
        on(device)
            .map<std::uint32_t>("nested-aggregate-failure-observe", 1u,
                                [](auto value) { return value; })
            .compile();
    auto scratch = device.buffer<std::uint32_t>(1u);
    if (!outer || !queue || !domain || !count || !output || !observer ||
        !scratch) {
      return 1;
    }

    const auto body = tile_repeat<kInner>(seed, action, fold);
    auto builder = pipeline(device);
    builder.profile(PipelineProfile::Steps)
        .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                  read(*outer, *queue, *domain),
                                  write_final(*output));
    auto prepared = std::move(builder).prepare();
    std::array<std::uint32_t, 1u> actual{};
    const Status first_failed =
        prepared ? prepared->run() : Status::fail(prepared.reason());
    const Stats first_stats = prepared ? prepared->stats() : Stats{};
    std::array<PipelineStepProfile, kTemplates> first_rows{};
    const auto first_profile =
        prepared ? prepared->profile(first_rows)
                 : Result<PipelineProfileSnapshot>::fail(prepared.reason());
    const Reason expected_reason = reduce_overflow
                                       ? Reason::ReduceSumOverflow
                                       : Reason::GatherIndexOutOfRange;
    const std::uint64_t expected_ordinal =
        reduce_overflow ? std::numeric_limits<std::uint64_t>::max() : 0u;
    const std::uint64_t expected_generated =
        backend == Backend::Cpu ? 0u : (reduce_overflow ? 12u : 9u);
    const std::uint64_t expected_capacity =
        backend == Backend::Cpu ? 0u : (reduce_overflow ? 16u : 12u);
    const std::uint64_t expected_indirect =
        backend == Backend::Cpu ? 0u : (reduce_overflow ? 4u : 3u);
    const bool direct_shape = backend != Backend::Metal ||
                              first_stats.pipeline.control_command_count == 1u;
    if (!prepared || first_failed || first_failed.reason() != expected_reason ||
        !first_profile || first_profile->written != kTemplates ||
        first_profile->total != kTemplates || first_profile->truncated() ||
        prepared->generation() != 0u || prepared->poisoned() ||
        !Observe(*observer, *output, *scratch, actual) ||
        actual != output_values || !direct_shape ||
        first_stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
        first_stats.pipeline.verified_step_count != 0u ||
        first_stats.pipeline.failed_step_index != 0u ||
        first_stats.pipeline.failed_outer_window != 1u ||
        first_stats.pipeline.failed_inner_iteration !=
            PipelineStats::no_coordinate ||
        first_stats.pipeline.failed_nested_phase != PipelineNestedPhase::Seed ||
        first_stats.pipeline.executed_outer_window_count != 1u ||
        first_stats.pipeline.skipped_outer_window_count != 0u ||
        first_stats.pipeline.executed_inner_iteration_count != kInner ||
        first_stats.pipeline.skipped_inner_iteration_count != 0u ||
        first_stats.control.iteration_count != 1u ||
        first_stats.control.skipped_iteration_count != 0u ||
        first_stats.control.generated_item_count != expected_generated ||
        first_stats.control.generated_capacity != expected_capacity ||
        first_stats.control.indirect_dispatch_count != expected_indirect ||
        first_stats.control.indirect_work_item_count != expected_generated ||
        first_stats.control.overflow_ordinal != expected_ordinal ||
        first_stats.publication.discard_count != 1u ||
        first_profile->execution.pipeline.verified_step_count != 0u ||
        first_profile->execution.pipeline.failed_step_index != 0u) {
      std::fprintf(
          stderr,
          "nested aggregate seed failure backend=%u reduce=%u prepared=%u "
          "status=%u reason=%u/%u profile=%u/%u rows=%llu/%llu "
          "generation=%llu poison=%u output=%u/%u "
          "verified=%llu failed=%llu coords=%llu/%llu/%u outer=%llu/%llu "
          "inner=%llu/%llu control=%llu/%llu ordinal=%llu dispatch=%llu "
          "commands=%llu discard=%llu submits=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(reduce_overflow),
          static_cast<unsigned>(prepared.ok()),
          static_cast<unsigned>(first_failed.ok()),
          static_cast<unsigned>(first_failed.reason()),
          static_cast<unsigned>(expected_reason),
          static_cast<unsigned>(first_profile.ok()),
          static_cast<unsigned>(first_profile.reason()),
          static_cast<unsigned long long>(first_profile ? first_profile->written
                                                        : 0u),
          static_cast<unsigned long long>(first_profile ? first_profile->total
                                                        : 0u),
          static_cast<unsigned long long>(prepared ? prepared->generation()
                                                   : 0u),
          static_cast<unsigned>(prepared ? prepared->poisoned() : false),
          actual[0u], kSentinel,
          static_cast<unsigned long long>(
              first_stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(
              first_stats.pipeline.failed_outer_window),
          static_cast<unsigned long long>(
              first_stats.pipeline.failed_inner_iteration),
          static_cast<unsigned>(first_stats.pipeline.failed_nested_phase),
          static_cast<unsigned long long>(
              first_stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.skipped_outer_window_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.skipped_inner_iteration_count),
          static_cast<unsigned long long>(first_stats.control.iteration_count),
          static_cast<unsigned long long>(
              first_stats.control.skipped_iteration_count),
          static_cast<unsigned long long>(first_stats.control.overflow_ordinal),
          static_cast<unsigned long long>(first_stats.dispatches),
          static_cast<unsigned long long>(
              first_stats.pipeline.control_command_count),
          static_cast<unsigned long long>(
              first_stats.publication.discard_count),
          static_cast<unsigned long long>(first_stats.command_submits));
      return 2;
    }

    if (backend == Backend::Metal) {
      const PipelineStepStats &prefix = first_rows[0u].execution;
      const PipelineStepStats &failed = first_rows[1u].execution;
      const std::uint64_t failed_generated = reduce_overflow ? 4u : 1u;
      const std::uint64_t failed_capacity = reduce_overflow ? 8u : 4u;
      const std::uint64_t failed_indirect = reduce_overflow ? 2u : 1u;
      if (first_rows[0u].outer_window != 0u ||
          first_rows[1u].outer_window != 1u ||
          first_rows[0u].nested_phase != PipelineNestedPhase::Seed ||
          first_rows[1u].nested_phase != PipelineNestedPhase::Seed ||
          prefix.sample_count != 1u || prefix.final_dispatches != 2u ||
          prefix.workgroup_count != kOuter + 1u ||
          prefix.work_item_count == 0u ||
          prefix.control.generated_item_count != 2u * kTile ||
          prefix.control.generated_capacity != 2u * kTile ||
          prefix.control.indirect_dispatch_count != 2u ||
          prefix.control.indirect_work_item_count != 2u * kTile ||
          prefix.control.overflow_ordinal != ControlStats::no_overflow ||
          failed.sample_count != 1u || failed.original_dispatches == 0u ||
          failed.final_dispatches != 0u || failed.workgroup_count != 0u ||
          failed.work_item_count != 0u ||
          failed.control.generated_item_count != failed_generated ||
          failed.control.generated_capacity != failed_capacity ||
          failed.control.indirect_dispatch_count != failed_indirect ||
          failed.control.indirect_work_item_count != failed_generated ||
          failed.control.overflow_ordinal != expected_ordinal ||
          !rund_node_test_pipeline::TimingUnavailable(first_rows[0u].timing) ||
          !rund_node_test_pipeline::TimingUnavailable(first_rows[1u].timing)) {
        std::fprintf(
            stderr,
            "nested aggregate failure profile backend=%u reduce=%u "
            "prefix=%llu/%llu/%llu/%llu/%llu failed=%llu/%llu/%llu/%llu/"
            "%llu rows=%u/%u phases=%u/%u\n",
            static_cast<unsigned>(backend),
            static_cast<unsigned>(reduce_overflow),
            static_cast<unsigned long long>(prefix.sample_count),
            static_cast<unsigned long long>(prefix.final_dispatches),
            static_cast<unsigned long long>(prefix.workgroup_count),
            static_cast<unsigned long long>(
                prefix.control.generated_item_count),
            static_cast<unsigned long long>(prefix.control.overflow_ordinal),
            static_cast<unsigned long long>(failed.sample_count),
            static_cast<unsigned long long>(failed.final_dispatches),
            static_cast<unsigned long long>(
                failed.control.generated_item_count),
            static_cast<unsigned long long>(
                failed.control.indirect_dispatch_count),
            static_cast<unsigned long long>(failed.control.overflow_ordinal),
            first_rows[0u].outer_window, first_rows[1u].outer_window,
            static_cast<unsigned>(first_rows[0u].nested_phase),
            static_cast<unsigned>(first_rows[1u].nested_phase));
        return 3;
      }
      for (std::size_t index = 2u; index < first_rows.size(); ++index) {
        if (first_rows[index].execution.available() ||
            !rund_node_test_pipeline::TimingUnavailable(
                first_rows[index].timing)) {
          std::fprintf(stderr,
                       "nested aggregate failure suffix backend=%u reduce=%u "
                       "row=%llu work=%llu timing=%llu\n",
                       static_cast<unsigned>(backend),
                       static_cast<unsigned>(reduce_overflow),
                       static_cast<unsigned long long>(index),
                       static_cast<unsigned long long>(
                           first_rows[index].execution.sample_count),
                       static_cast<unsigned long long>(
                           first_rows[index].timing.sample_count));
          return 4;
        }
      }
    }

    const Status second_failed = prepared->run();
    const Stats second_stats = prepared->stats();
    std::array<PipelineStepProfile, kTemplates> second_rows{};
    const auto second_profile = prepared->profile(second_rows);
    if (second_failed || second_failed.reason() != expected_reason ||
        !second_profile || second_profile->written != kTemplates ||
        second_profile->total != kTemplates || second_profile->truncated() ||
        prepared->generation() != 0u || prepared->poisoned() ||
        second_stats.pipeline.verified_step_count != 0u ||
        second_stats.pipeline.failed_step_index != 0u ||
        second_stats.pipeline.failed_outer_window != 1u ||
        second_stats.pipeline.failed_nested_phase !=
            PipelineNestedPhase::Seed ||
        second_stats.publication.discard_count != 2u ||
        second_profile->execution.pipeline.verified_step_count != 0u ||
        second_profile->execution.pipeline.failed_step_index != 0u) {
      std::fprintf(
          stderr,
          "nested aggregate failure repeat backend=%u reduce=%u status=%u "
          "reason=%u/%u profile=%u/%u rows=%llu/%llu generation=%llu "
          "poison=%u verified=%llu failed=%llu outer=%llu phase=%u "
          "discard=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(reduce_overflow),
          static_cast<unsigned>(second_failed.ok()),
          static_cast<unsigned>(second_failed.reason()),
          static_cast<unsigned>(expected_reason),
          static_cast<unsigned>(second_profile.ok()),
          static_cast<unsigned>(second_profile.reason()),
          static_cast<unsigned long long>(
              second_profile ? second_profile->written : 0u),
          static_cast<unsigned long long>(second_profile ? second_profile->total
                                                         : 0u),
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned>(prepared->poisoned()),
          static_cast<unsigned long long>(
              second_stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(
              second_stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(
              second_stats.pipeline.failed_outer_window),
          static_cast<unsigned>(second_stats.pipeline.failed_nested_phase),
          static_cast<unsigned long long>(
              second_stats.publication.discard_count));
      return 5;
    }
    for (std::size_t index = 0u; index < first_rows.size(); ++index) {
      if (!same_execution(first_rows[index].execution,
                          second_rows[index].execution) ||
          (backend == Backend::Metal &&
           !rund_node_test_pipeline::TimingUnavailable(
               second_rows[index].timing))) {
        std::fprintf(
            stderr,
            "nested aggregate failure profile reset backend=%u reduce=%u "
            "row=%llu samples=%llu/%llu timing=%llu generated=%llu/%llu "
            "overflow=%llu/%llu\n",
            static_cast<unsigned>(backend),
            static_cast<unsigned>(reduce_overflow),
            static_cast<unsigned long long>(index),
            static_cast<unsigned long long>(
                first_rows[index].execution.sample_count),
            static_cast<unsigned long long>(
                second_rows[index].execution.sample_count),
            static_cast<unsigned long long>(
                second_rows[index].timing.sample_count),
            static_cast<unsigned long long>(
                first_rows[index].execution.control.generated_item_count),
            static_cast<unsigned long long>(
                second_rows[index].execution.control.generated_item_count),
            static_cast<unsigned long long>(
                first_rows[index].execution.control.overflow_ordinal),
            static_cast<unsigned long long>(
                second_rows[index].execution.control.overflow_ordinal));
        return 6;
      }
    }
    return 0;
  };

  const int gather = run_case(false);
  if (gather != 0) {
    return gather;
  }
  const int reduce = run_case(true);
  return reduce == 0 ? 0 : 10 + reduce;
}

int CheckNestedAggregateSeedFailures(rund::compute::Device &device,
                                     const rund::compute::Backend backend,
                                     const NestedSeed &seed,
                                     const NestedAction &action,
                                     const NestedFold &fold) {
  return CheckAggregateSeedFailures(device, backend, seed, action, fold);
}

} // namespace rund::node::test_contract::window
