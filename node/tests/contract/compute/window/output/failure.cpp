#include "../../allocation.hpp"
#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/output.hpp"
#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {
template <class Seed, class Fold>
[[nodiscard]] int CheckScatterConflicts(Device &device, const Backend backend,
                                        const Seed &seed, const Fold &fold,
                                        WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::array<std::uint32_t, kOutputTile>, 2u> values{{
      {5u, 5u, 0u, 0u},
      {5u, 7u, 11u, 13u},
  }};
  constexpr std::array<std::array<std::uint32_t, kOutputTile>, 2u> targets{{
      {3u, 3u, 0u, 1u},
      {3u, 3u, 3u, 3u},
  }};
  constexpr std::array<std::uint32_t, 2u> expected{10u, 36u};
  constexpr std::array<std::uint32_t, 1u> count_value{kOutputMaximum};
  constexpr std::array<std::uint32_t, 1u> initial{0u};
  std::array<std::uint64_t, 2u> output_hashes{};
  Fingerprint pipeline_fingerprint{};

  for (std::size_t scenario = 0u; scenario < values.size(); ++scenario) {
    std::array<std::uint32_t, kOutputMaximum> initial_window{};
    initial_window.fill(kOutputSentinel);
    auto count = device.upload<std::uint32_t>(count_value);
    auto outer = device.upload<std::uint32_t>(initial);
    auto input_values = device.upload<std::uint32_t>(values[scenario]);
    auto input_targets = device.upload<std::uint32_t>(targets[scenario]);
    auto final = device.upload<std::uint32_t>(initial);
    auto window_target = device.upload<std::uint32_t>(initial_window);
    if (!count || !outer || !input_values || !input_targets || !final ||
        !window_target) {
      return 1;
    }

    const auto body = tile_repeat<0u>(seed, fold);
    auto builder = pipeline(device);
    builder.windows<kOutputMaximum, kOutputTile>(
        body, rund::compute::window(*count),
        read(*outer, *input_values, *input_targets), write_final(*final),
        write_window(*window_target));
    const auto plan = builder.plan();
    if (!plan || plan->outer_window_count != kOutputOuter ||
        plan->prepared_template_count != kOutputTemplates ||
        plan->prepared_command_count != kOutputCommands ||
        plan->publish_count != kOutputOuter + 1u ||
        plan->publish_bytes != (kOutputMaximum + 1u) * sizeof(std::uint32_t)) {
      return 2;
    }
    auto prepared = std::move(builder).prepare();
    if (!prepared || !prepared->run()) {
      return 3;
    }
    std::array<std::uint32_t, 1u> first_final{};
    std::array<std::uint32_t, kOutputMaximum> first_window{};
    if (!prepared->read(*final, first_final) ||
        !prepared->read(*window_target, first_window)) {
      return 4;
    }
    const MemoryStats before_warm = prepared->memory();
    if (!prepared->run()) {
      return 5;
    }
    const Stats stats = prepared->stats();
    const MemoryStats after_warm = prepared->memory();
    std::array<std::uint32_t, 1u> actual_final{};
    std::array<std::uint32_t, kOutputMaximum> actual_window{};
    std::array<std::uint32_t, kOutputMaximum> expected_window{};
    expected_window[kOutputMaximum - 1u] = expected[scenario];
    if (!prepared->read(*final, actual_final) ||
        !prepared->read(*window_target, actual_window) ||
        prepared->generation() != 2u ||
        actual_final[0u] != expected[scenario] ||
        actual_window != expected_window || first_final != actual_final ||
        first_window != actual_window ||
        !NoAllocation(before_warm, after_warm) ||
        !OutputWarmSetupClean(stats) ||
        stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
        stats.pipeline.executed_outer_window_count != kOutputOuter ||
        stats.pipeline.skipped_outer_window_count != 0u ||
        stats.publication.generation != 2u ||
        stats.publication.discard_count != 0u) {
      std::fprintf(
          stderr,
          "window output scatter backend=%u scenario=%llu "
          "final=%u/%u high=%u generation=%llu outer=%llu "
          "discard=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(scenario), actual_final[0u],
          expected[scenario], actual_window[kOutputMaximum - 1u],
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(stats.publication.discard_count));
      return 6;
    }
    std::array<std::uint32_t, kOutputMaximum + 1u> raw{};
    raw[0u] = actual_final[0u];
    std::copy(actual_window.begin(), actual_window.end(), raw.begin() + 1u);
    output_hashes[scenario] = Hash(raw.data(), sizeof(raw));
    if (scenario == 0u) {
      pipeline_fingerprint = prepared->fingerprint();
    } else if (pipeline_fingerprint != prepared->fingerprint()) {
      return 7;
    }
  }

  const WindowOutputIdentity current{
      .scatter_seed = seed.fingerprint(),
      .scatter_fold = fold.fingerprint(),
      .scatter_pipeline = pipeline_fingerprint,
      .scatter_output = Hash(output_hashes.data(), sizeof(output_hashes)),
  };
  if (!current.scatter_seed || !current.scatter_fold ||
      !current.scatter_pipeline || current.scatter_output == 0u) {
    return 8;
  }
  if (identity.scatter_seed) {
    return identity.scatter_seed == current.scatter_seed &&
                   identity.scatter_fold == current.scatter_fold &&
                   identity.scatter_pipeline == current.scatter_pipeline &&
                   identity.scatter_output == current.scatter_output
               ? 0
               : 9;
  }
  identity.scatter_seed = current.scatter_seed;
  identity.scatter_fold = current.scatter_fold;
  identity.scatter_pipeline = current.scatter_pipeline;
  identity.scatter_output = current.scatter_output;
  return 0;
}

template <class Body>
[[nodiscard]] int
CheckLateFailure(Device &device, const Backend backend, const Body &body,
                 const std::uint32_t count_value, const Reason expected_reason,
                 const rund::compute::PipelineNestedPhase phase,
                 const std::uint64_t inner) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  const std::array<std::uint32_t, 1u> count_values{count_value};
  constexpr std::array<std::uint32_t, 1u> final_values{kOutputSentinel};
  std::array<std::uint32_t, kOutputMaximum> window_values{};
  window_values.fill(kOutputSentinel);
  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  auto final_observer =
      on(device)
          .map<std::uint32_t>("window-output-final-observe", 1u,
                              [](auto value) { return value; })
          .compile();
  auto final_scratch = device.buffer<std::uint32_t>(1u);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended || !final_observer || !final_scratch) {
    return 1;
  }
  auto builder = pipeline(device);
  builder.windows<kOutputMaximum, kOutputTile>(
      body, rund::compute::window(*count),
      read(*outer, *values, *lanes, *witness), write_final(*final),
      write_window(*appended));
  auto prepared = std::move(builder).prepare();
  const Status failed =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  std::array<std::uint32_t, kOutputMaximum> unread{};
  const Status poisoned_read = prepared ? prepared->read(*appended, unread)
                                        : Status::fail(prepared.reason());
  auto observed = final_observer->run(*final, *final_scratch);
  std::array<std::uint32_t, 1u> actual_final{};
  const bool final_unchanged =
      observed &&
      observed->read(*final_scratch, std::span<std::uint32_t>{actual_final}) &&
      actual_final == final_values;
  const Status rerun =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  if (!prepared || failed || failed.reason() != expected_reason ||
      prepared->generation() != 0u || !prepared->poisoned() || poisoned_read ||
      poisoned_read.reason() != Reason::BufferPoisoned || !final_unchanged ||
      rerun || rerun.reason() != Reason::PipelinePoisoned ||
      stats.pipeline.failed_outer_window != 1u ||
      stats.pipeline.failed_inner_iteration != inner ||
      stats.pipeline.failed_nested_phase != phase ||
      stats.pipeline.executed_outer_window_count != 1u ||
      stats.publication.discard_count != 1u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window output failure backend=%u phase=%u status=%u/%u/%u "
        "generation=%llu poison=%u read=%u/%u rerun=%u/%u final=%u/%u "
        "outer=%llu coord=%llu/%llu/%u discard=%llu submit=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(phase),
        static_cast<unsigned>(failed.ok()),
        static_cast<unsigned>(failed.reason()),
        static_cast<unsigned>(expected_reason),
        static_cast<unsigned long long>(prepared ? prepared->generation() : 0u),
        static_cast<unsigned>(prepared ? prepared->poisoned() : false),
        static_cast<unsigned>(poisoned_read.ok()),
        static_cast<unsigned>(poisoned_read.reason()),
        static_cast<unsigned>(rerun.ok()),
        static_cast<unsigned>(rerun.reason()), actual_final[0u],
        kOutputSentinel,
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(stats.pipeline.failed_outer_window),
        static_cast<unsigned long long>(stats.pipeline.failed_inner_iteration),
        static_cast<unsigned>(stats.pipeline.failed_nested_phase),
        static_cast<unsigned long long>(stats.publication.discard_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 2;
  }
  return 0;
}

template <class Seed, class Fold, class FoldTwo>
[[nodiscard]] int CheckAliases(Device &device, const Seed &seed,
                               const Fold &fold, const FoldTwo &fold_two) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> scalar{1u};
  auto outer = device.upload<std::uint32_t>(scalar);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(scalar);
  auto count = device.upload<std::uint32_t>(scalar);
  auto final = device.upload<std::uint32_t>(scalar);
  auto shared = device.buffer<std::uint32_t>(2u * kOutputMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !shared) {
    return 1;
  }
  auto first = shared->view(0u, kOutputMaximum);
  auto second = shared->view(kOutputMaximum, kOutputMaximum);
  if (!first || !second) {
    return 2;
  }
  const auto two = tile_repeat<0u>(seed, fold_two);
  auto duplicate = pipeline(device);
  duplicate.windows<kOutputMaximum, kOutputTile>(
      two, rund::compute::window(*count),
      read(*outer, *values, *lanes, *witness), write_final(*final),
      write_window(*first, *second));
  const auto duplicate_plan = duplicate.plan();
  if (duplicate_plan || duplicate_plan.reason() != Reason::BindingDuplicate) {
    std::fprintf(stderr, "window output duplicate reason=%u\n",
                 static_cast<unsigned>(duplicate_plan.reason()));
    return 3;
  }

  std::array<std::uint32_t, kOutputMaximum> aliased_values{};
  aliased_values[0u] = 1u;
  auto aliased = device.upload<std::uint32_t>(aliased_values);
  if (!aliased) {
    return 4;
  }
  auto count_view = aliased->view(0u, 1u);
  if (!count_view) {
    return 5;
  }
  const auto one = tile_repeat<0u>(seed, fold);
  auto overlap = pipeline(device);
  overlap.windows<kOutputMaximum, kOutputTile>(
      one, rund::compute::window(*count_view),
      read(*outer, *values, *lanes, *witness), write_final(*final),
      write_window(*aliased));
  const auto overlap_plan = overlap.plan();
  if (overlap_plan ||
      overlap_plan.reason() != Reason::BindingAliasUnsupported) {
    std::fprintf(stderr, "window output overlap reason=%u\n",
                 static_cast<unsigned>(overlap_plan.reason()));
  }
  return !overlap_plan &&
                 overlap_plan.reason() == Reason::BindingAliasUnsupported
             ? 0
             : 6;
}

int CheckOutputScatterConflicts(Device &device, const Backend backend,
                                const OutputScatterSeed &seed,
                                const OutputScatterFold &fold,
                                WindowOutputIdentity &identity) {
  return CheckScatterConflicts(device, backend, seed, fold, identity);
}
int CheckOutputLateFailures(Device &device, const Backend backend,
                            const OutputSeed &seed,
                            const OutputSeed &seed_fault,
                            const OutputAction &action_fault,
                            const OutputFold &fold,
                            const OutputFold &priority_fold) {
  using namespace rund::compute;
  const auto seed_failure = tile_repeat<0u>(seed_fault, fold);
  if (const int failure = CheckLateFailure(
          device, backend, seed_failure, 5u, Reason::GatherIndexOutOfRange,
          PipelineNestedPhase::Seed, PipelineStats::no_coordinate);
      failure != 0) {
    return 80 + failure;
  }
  const auto action_failure = tile_repeat<1u>(seed, action_fault, fold);
  if (const int failure = CheckLateFailure(device, backend, action_failure, 5u,
                                           Reason::GatherIndexOutOfRange,
                                           PipelineNestedPhase::Action, 0u);
      failure != 0) {
    return 90 + failure;
  }
  const auto priority_failure = tile_repeat<0u>(seed, priority_fold);
  if (const int failure = CheckLateFailure(
          device, backend, priority_failure, 8u, Reason::ScatterIndexOutOfRange,
          PipelineNestedPhase::Fold, PipelineStats::no_coordinate);
      failure != 0) {
    return 100 + failure;
  }
  return 0;
}
int CheckOutputAliases(Device &device, const OutputSeed &seed,
                       const OutputFold &fold, const OutputFoldTwo &fold_two) {
  return CheckAliases(device, seed, fold, fold_two);
}

} // namespace rund::node::test_contract::window
