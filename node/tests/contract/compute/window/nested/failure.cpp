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
[[nodiscard]] int CheckNestedFailureCase(
    rund::compute::Device &device, const rund::compute::Backend backend,
    const Seed &seed, const Action &action, const Fold &fold,
    const rund::compute::PipelineNestedPhase phase, const std::uint64_t inner,
    const std::uint64_t executed_inner) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{kMaximum};
  constexpr std::array<std::uint32_t, 1u> output_values{kSentinel};
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{output_values});
  auto observer = on(device)
                      .map<std::uint32_t>("nested-window-failure-observe", 1u,
                                          [](auto value) { return value; })
                      .compile();
  auto scratch = device.buffer<std::uint32_t>(1u);
  if (!outer || !count || !output || !observer || !scratch) {
    return 1;
  }
  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer), write_final(*output));
  const auto plan = builder.plan();
  if (!plan) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  std::array<std::uint32_t, 1u> actual{};
  const Status failed =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  if (!prepared || failed || failed.reason() != Reason::GatherIndexOutOfRange ||
      prepared->generation() != 0u || prepared->poisoned() ||
      !Observe(*observer, *output, *scratch, actual) ||
      actual != output_values ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
      stats.pipeline.verified_step_count != 0u ||
      stats.pipeline.failed_step_index != 0u ||
      stats.pipeline.failed_outer_window != 1u ||
      stats.pipeline.failed_inner_iteration != inner ||
      stats.pipeline.failed_nested_phase != phase ||
      stats.pipeline.executed_outer_window_count != 1u ||
      stats.pipeline.skipped_outer_window_count != 0u ||
      stats.pipeline.executed_inner_iteration_count != executed_inner ||
      stats.pipeline.skipped_inner_iteration_count != 0u ||
      stats.control.iteration_count != 1u ||
      stats.publication.discard_count != 1u) {
    std::fprintf(
        stderr,
        "nested failure backend=%u phase=%u prepared=%u status=%u reason=%u "
        "generation=%llu poison=%u output=%u/%u verified=%llu failed=%llu "
        "coords=%llu/%llu expected_inner=%llu outer=%llu/%llu "
        "inner_work=%llu/%llu control=%llu discard=%llu submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(phase),
        static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(failed.ok()),
        static_cast<unsigned>(failed.reason()),
        static_cast<unsigned long long>(prepared ? prepared->generation() : 0u),
        static_cast<unsigned>(prepared ? prepared->poisoned() : false),
        actual[0], kSentinel,
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.pipeline.failed_outer_window),
        static_cast<unsigned long long>(stats.pipeline.failed_inner_iteration),
        static_cast<unsigned long long>(inner),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(executed_inner),
        static_cast<unsigned long long>(stats.control.iteration_count),
        static_cast<unsigned long long>(stats.publication.discard_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 3;
  }
  return 0;
}

[[nodiscard]] int CheckNestedFailures(rund::compute::Device &device,
                                      const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto seed = MakeNestedFailureSeedProgram(device, false);
  auto seed_fault = MakeNestedFailureSeedProgram(device, true);
  auto action = on(device)
                    .map<std::uint32_t>("nested-window-failure-action", 1u,
                                        [](auto value) { return value + 1u; })
                    .compile();
  auto action_first = MakeNestedFailureActionProgram(device, 16u);
  auto action_middle = MakeNestedFailureActionProgram(device, 17u);
  auto action_last = MakeNestedFailureActionProgram(device, 18u);
  auto fold = MakeNestedFailureFoldProgram(device, false);
  auto fold_fault = MakeNestedFailureFoldProgram(device, true);
  if (!seed || !seed_fault || !action || !action_first || !action_middle ||
      !action_last || !fold || !fold_fault) {
    return 1;
  }
  constexpr std::uint64_t none = PipelineStats::no_coordinate;
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed_fault, *action, *fold,
                                 PipelineNestedPhase::Seed, none, kInner);
      result != 0) {
    return 10 + result;
  }
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed, *action_first, *fold,
                                 PipelineNestedPhase::Action, 0u, kInner);
      result != 0) {
    return 20 + result;
  }
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed, *action_middle, *fold,
                                 PipelineNestedPhase::Action, 1u, kInner + 1u);
      result != 0) {
    return 30 + result;
  }
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed, *action_last, *fold,
                                 PipelineNestedPhase::Action, 2u, kInner + 2u);
      result != 0) {
    return 40 + result;
  }
  const int fold_result =
      CheckNestedFailureCase(device, backend, *seed, *action, *fold_fault,
                             PipelineNestedPhase::Fold, none, kInner * 2u);
  return fold_result == 0 ? 0 : 50 + fold_result;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] int CheckNestedProfile(rund::compute::Device &device,
                                     const rund::compute::Backend backend,
                                     const Seed &seed, const Action &action,
                                     const Fold &fold) {
  using namespace rund::compute;
  constexpr std::uint32_t active_count = kTile + 1u;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{active_count};
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.buffer<std::uint32_t>(1u);
  if (!outer || !queue || !domain || !count || !output) {
    return 1;
  }

  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder.profile(PipelineProfile::Steps)
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *queue, *domain),
                                write_final(*output));
  const auto plan = builder.plan();
  if (!plan) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  const graph::Fingerprint seed_fingerprint = seed.fingerprint();
  const graph::Fingerprint action_fingerprint = action.fingerprint();
  const graph::Fingerprint fold_fingerprint = fold.fingerprint();
  const auto rows_exact = [&](const auto &candidate,
                              std::size_t &mismatch) noexcept {
    for (std::size_t index = 0u; index < candidate.size(); ++index) {
      const PipelineStepProfile &row = candidate[index];
      const bool seed_row = index < kOuter;
      const bool action_row = index >= kOuter && index < kOuter + kInner;
      const std::uint32_t expected_iteration =
          seed_row     ? static_cast<std::uint32_t>(index)
          : action_row ? static_cast<std::uint32_t>(index - kOuter)
                       : static_cast<std::uint32_t>(index - kOuter - kInner);
      const PipelineNestedPhase expected_phase =
          seed_row     ? PipelineNestedPhase::Seed
          : action_row ? PipelineNestedPhase::Action
                       : PipelineNestedPhase::Fold;
      const graph::Fingerprint expected_program =
          seed_row ? seed_fingerprint
                   : (action_row ? action_fingerprint : fold_fingerprint);
      if (row.index != 0u || row.iteration != expected_iteration ||
          row.outer_window_bound != kOuter ||
          row.inner_iteration_bound != kInner ||
          row.nested_phase != expected_phase ||
          row.program != expected_program ||
          row.outer_window != (seed_row ? expected_iteration
                                        : PipelineStepProfile::no_coordinate) ||
          row.inner_iteration != (action_row
                                      ? expected_iteration
                                      : PipelineStepProfile::no_coordinate)) {
        mismatch = index;
        return false;
      }
    }
    mismatch = candidate.size();
    return true;
  };
  std::array<PipelineStepProfile, kTemplates> cold_rows{};
  const auto cold_profile =
      prepared ? prepared->profile(cold_rows)
               : Result<PipelineProfileSnapshot>::fail(prepared.reason());
  std::size_t cold_mismatch = cold_rows.size();
  const bool cold_exact = cold_profile && cold_profile->written == kTemplates &&
                          cold_profile->total == kTemplates &&
                          rows_exact(cold_rows, cold_mismatch);
  std::array<std::uint32_t, 1u> actual{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  std::array<PipelineStepProfile, kTemplates> rows{};
  const auto profile =
      prepared ? prepared->profile(rows)
               : Result<PipelineProfileSnapshot>::fail(prepared.reason());
  if (!prepared || !cold_exact || !ran || !profile ||
      profile->written != kTemplates || profile->total != kTemplates ||
      !prepared->read(*output, actual) ||
      actual[0] != SerialOracle(active_count) ||
      (backend == Backend::Metal &&
       (profile->execution.dispatches != 2u ||
        profile->execution.pipeline.control_command_count != 1u)) ||
      profile->execution.pipeline.executed_outer_window_count != 2u ||
      profile->execution.pipeline.executed_inner_iteration_count !=
          2u * kInner ||
      profile->execution.command_submits !=
          (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested profile backend=%u prepared=%u cold=%u/%u rows=%llu/%llu "
        "mismatch=%llu run=%u/%u profile=%u/%u rows=%llu/%llu output=%u/%u "
        "outer=%llu inner=%llu submits=%llu dispatches=%llu control=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(cold_profile.ok()),
        static_cast<unsigned>(cold_profile.reason()),
        static_cast<unsigned long long>(cold_profile ? cold_profile->written
                                                     : 0u),
        static_cast<unsigned long long>(cold_profile ? cold_profile->total
                                                     : 0u),
        static_cast<unsigned long long>(cold_mismatch),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        static_cast<unsigned>(profile.ok()),
        static_cast<unsigned>(profile.reason()),
        static_cast<unsigned long long>(profile ? profile->written : 0u),
        static_cast<unsigned long long>(profile ? profile->total : 0u),
        actual[0], SerialOracle(active_count),
        static_cast<unsigned long long>(
            profile ? profile->execution.pipeline.executed_outer_window_count
                    : 0u),
        static_cast<unsigned long long>(
            profile ? profile->execution.pipeline.executed_inner_iteration_count
                    : 0u),
        static_cast<unsigned long long>(
            profile ? profile->execution.command_submits : 0u),
        static_cast<unsigned long long>(profile ? profile->execution.dispatches
                                                : 0u),
        static_cast<unsigned long long>(
            profile ? profile->execution.pipeline.control_command_count : 0u));
    return 3;
  }

  if (backend == Backend::Vulkan) {
    for (std::size_t inner = 0u; inner < kInner; ++inner) {
      const PipelineStepStats &execution = rows[kOuter + inner].execution;
      const std::uint64_t expected_physical = inner == 0u ? kOuter : 0u;
      if (execution.sample_count != 1u ||
          execution.original_dispatches != kOuter ||
          execution.final_dispatches != expected_physical ||
          execution.workgroup_count != expected_physical ||
          execution.work_item_count != expected_physical) {
        std::fprintf(
            stderr,
            "nested Vulkan transducer row=%llu samples=%llu authored=%llu "
            "physical=%llu/%llu groups=%llu items=%llu\n",
            static_cast<unsigned long long>(inner),
            static_cast<unsigned long long>(execution.sample_count),
            static_cast<unsigned long long>(execution.original_dispatches),
            static_cast<unsigned long long>(execution.final_dispatches),
            static_cast<unsigned long long>(expected_physical),
            static_cast<unsigned long long>(execution.workgroup_count),
            static_cast<unsigned long long>(execution.work_item_count));
        return 4;
      }
    }
  }

  if (backend == Backend::Metal) {
    for (std::size_t index = 0u; index < rows.size(); ++index) {
      const PipelineStepStats &execution = rows[index].execution;
      const bool physical_owner = index == 0u;
      const std::uint64_t expected_dispatches = physical_owner ? 2u : 0u;
      const std::uint64_t expected_workgroups =
          physical_owner ? kOuter + 1u : 0u;
      const std::uint32_t seed_live =
          index == 0u ? kTile : (index == 1u ? active_count - kTile : 0u);
      const bool seed_active = index < kOuter && seed_live != 0u;
      const std::uint64_t generated = 2u * seed_live;
      const std::uint64_t capacity = seed_active ? 2u * kTile : 0u;
      const std::uint64_t indirect = seed_active ? 2u : 0u;
      if (execution.sample_count != 1u || execution.original_dispatches == 0u ||
          execution.final_dispatches != expected_dispatches ||
          execution.workgroup_count != expected_workgroups ||
          (physical_owner ? execution.work_item_count == 0u
                          : execution.work_item_count != 0u) ||
          execution.control.generated_item_count != generated ||
          execution.control.generated_capacity != capacity ||
          execution.control.indirect_dispatch_count != indirect ||
          execution.control.indirect_work_item_count != generated ||
          execution.control.overflow_ordinal != ControlStats::no_overflow) {
        std::fprintf(
            stderr,
            "nested Metal aggregate profile row=%llu samples=%llu "
            "original=%llu physical=%llu groups=%llu items=%llu "
            "generated=%llu/%llu capacity=%llu/%llu indirect=%llu/%llu "
            "work=%llu/%llu overflow=%llu\n",
            static_cast<unsigned long long>(index),
            static_cast<unsigned long long>(execution.sample_count),
            static_cast<unsigned long long>(execution.original_dispatches),
            static_cast<unsigned long long>(execution.final_dispatches),
            static_cast<unsigned long long>(execution.workgroup_count),
            static_cast<unsigned long long>(execution.work_item_count),
            static_cast<unsigned long long>(
                execution.control.generated_item_count),
            static_cast<unsigned long long>(generated),
            static_cast<unsigned long long>(
                execution.control.generated_capacity),
            static_cast<unsigned long long>(capacity),
            static_cast<unsigned long long>(
                execution.control.indirect_dispatch_count),
            static_cast<unsigned long long>(indirect),
            static_cast<unsigned long long>(
                execution.control.indirect_work_item_count),
            static_cast<unsigned long long>(generated),
            static_cast<unsigned long long>(
                execution.control.overflow_ordinal));
        return 4;
      }
    }
  }

  std::size_t warm_mismatch = rows.size();
  if (!rows_exact(rows, warm_mismatch)) {
    std::fprintf(stderr, "nested profile row backend=%u mismatch=%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(warm_mismatch));
    return 5;
  }
  return 0;
}

int CheckNestedProfileCase(rund::compute::Device &device,
                           const rund::compute::Backend backend,
                           const NestedSeed &seed, const NestedAction &action,
                           const NestedFold &fold) {
  return CheckNestedProfile(device, backend, seed, action, fold);
}

} // namespace rund::node::test_contract::window
