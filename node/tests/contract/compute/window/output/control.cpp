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
template <class Seed, class Fold, class Advance>
[[nodiscard]] int
CheckTransactionalCountParity(Device &device, const Seed &seed,
                              const Fold &fold, const Advance &advance) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> first_count{
      static_cast<std::uint32_t>(kOutputTile)};
  constexpr std::array<std::uint32_t, 1u> second_count{0u};
  constexpr std::array<std::uint32_t, 1u> final_values{kOutputSentinel};
  std::array<std::uint32_t, kOutputMaximum> window_values{};
  window_values.fill(kOutputSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count_first = device.upload<std::uint32_t>(first_count);
  auto count_second = device.upload<std::uint32_t>(second_count);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !values || !lanes || !witness || !count_first ||
      !count_second || !final || !appended) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.state(*count_first, *count_second)
      .then(advance, read(*count_first), write(*count_second))
      .template windows<kOutputMaximum, kOutputTile>(
          body, rund::compute::window(*count_second),
          read(*outer, *values, *lanes, *witness), write_final(*final),
          write_window(*appended))
      .commit();
  auto prepared = std::move(builder).prepare();
  if (!prepared) {
    return 2;
  }
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(*prepared);
  const auto publication = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(value);
      });
  if (state == nullptr || publication == state->publications.end()) {
    return 3;
  }
  const auto &window = std::get<PipelineWindowPublicationPlan>(*publication);
  const std::uint32_t count_ordinal =
      state->windows[window.state].control.count.identity.resource_ordinal;
  if (count_ordinal >= state->resources.size() ||
      state->resources[count_ordinal].partner >= state->resources.size() ||
      state->resources[state->resources[count_ordinal].partner].partner !=
          count_ordinal) {
    return 4;
  }

  const auto check = [&](const std::uint32_t count_value) {
    const Status ran = prepared->run();
    std::array<std::uint32_t, 1u> actual_final{};
    std::array<std::uint32_t, kOutputMaximum> actual_window{};
    const auto expected_window = ExpectedWindow(kOutputValues, count_value);
    return ran && prepared->read(*final, actual_final) &&
           prepared->read(*appended, actual_window) &&
           actual_final[0u] == ExpectedFinal(kOutputValues, count_value) &&
           actual_window == expected_window;
  };
  if (!check(static_cast<std::uint32_t>(2u * kOutputTile))) {
    return 5;
  }
  if (!check(static_cast<std::uint32_t>(3u * kOutputTile)) ||
      prepared->generation() != 2u) {
    return 6;
  }
  return 0;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckStatePairPublicationTargetRejected(Device &device,
                                                          const Seed &seed,
                                                          const Fold &fold) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{
      static_cast<std::uint32_t>(kOutputMaximum)};
  constexpr std::array<std::uint32_t, 1u> pending_values{kOutputSentinel};
  std::array<std::uint32_t, kOutputMaximum> window_values{};
  window_values.fill(kOutputSentinel);

  auto published = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto pending = device.upload<std::uint32_t>(pending_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!published || !values || !lanes || !witness || !count || !pending ||
      !appended) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.state(*published, *pending)
      .template windows<kOutputMaximum, kOutputTile>(
          body, rund::compute::window(*count),
          read(*published, *values, *lanes, *witness), write_final(*pending),
          write_window(*appended))
      .commit();
  const auto plan = builder.plan();
  auto prepared = std::move(builder).prepare();
  return plan && !prepared && prepared.reason() == Reason::PipelineInvalid ? 0
                                                                           : 2;
}

[[nodiscard]] bool
OutputPreparedShape(const rund::compute::Pipeline &pipeline) {
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->steps.size() != kOutputTemplates ||
      state->windows.size() != 1u || state->publications.size() != 2u) {
    return false;
  }
  const PipelineWindow &window = state->windows[0u];
  const auto &shape = window.nested_shape;
  if (!window.nested() || shape.first() != 0u ||
      shape.end() != kOutputTemplates || shape.seed_first() != 0u ||
      shape.seed_count() != kOutputOuter ||
      shape.action_first() != kOutputOuter || shape.action_count() != 0u ||
      shape.fold_first() != kOutputOuter ||
      window.recurrent_output_count != 1u ||
      window.control.maximum != kOutputMaximum ||
      window.control.tile != kOutputTile) {
    return false;
  }
  for (std::size_t index = 0u; index < state->steps.size(); ++index) {
    const PipelineRoute expected = index < kOutputOuter
                                       ? PipelineRoute::NestedSeed
                                       : PipelineRoute::NestedFold;
    if (state->steps[index].route != expected ||
        state->steps[index].job == nullptr) {
      return false;
    }
  }
  const auto terminal = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineTerminalPublicationPlan>(value);
      });
  const auto appended = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(value);
      });
  if (terminal == state->publications.end() ||
      appended == state->publications.end()) {
    return false;
  }
  const auto &terminal_plan =
      std::get<PipelineTerminalPublicationPlan>(*terminal);
  const auto &window_plan = std::get<PipelineWindowPublicationPlan>(*appended);
  return window.control.final < terminal_plan.sources.size() &&
         terminal_plan.sources[window.control.final].identity.count == 1u &&
         window_plan.source.identity.count == kOutputTile &&
         window.control.count.identity.count == 1u &&
         window_plan.target.view.identity.stride_bytes ==
             window_plan.target.view.identity.element_bytes;
}

template <class Seed, class Fold>
[[nodiscard]] int
CheckCount(Device &device, const Backend backend, const Seed &seed,
           const Fold &fold, const std::uint32_t count_value,
           const std::array<std::uint32_t, kOutputMaximum> &values,
           const bool high_index_only, WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  const std::array<std::uint32_t, 3u> count_values{0x13579BDFu, count_value,
                                                   0x2468ACE0u};
  constexpr std::array<std::uint32_t, 1u> final_values{kOutputSentinel};
  std::array<std::uint32_t, kOutputWindowBacking> window_values{};
  window_values.fill(kOutputSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto input_values = device.upload<std::uint32_t>(values);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !input_values || !lanes || !witness || !count || !final ||
      !appended) {
    return 1;
  }
  auto count_view = count->view(1u, 1u);
  auto appended_view = appended->view(kOutputWindowOffset, kOutputMaximum);
  if (!count_view || !appended_view) {
    return 2;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.windows<kOutputMaximum, kOutputTile>(
      body, rund::compute::window(*count_view),
      read(*outer, *input_values, *lanes, *witness), write_final(*final),
      write_window(*appended_view));
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != kOutputOuter ||
      plan->tile_capacity != kOutputTile || plan->inner_iteration_count != 0u ||
      plan->prepared_template_count != kOutputTemplates ||
      plan->prepared_command_count != kOutputCommands ||
      plan->publish_count != kOutputOuter + 1u ||
      plan->publish_bytes != (kOutputMaximum + 1u) * sizeof(std::uint32_t)) {
    if (!plan) {
      std::fprintf(stderr, "window output plan backend=%u count=%u reason=%u\n",
                   static_cast<unsigned>(backend), count_value,
                   static_cast<unsigned>(plan.reason()));
    } else {
      std::fprintf(
          stderr,
          "window output plan backend=%u count=%u outer=%llu "
          "tile=%llu inner=%llu templates=%llu commands=%llu "
          "publish=%llu/%llu state=%llu transient=%llu\n",
          static_cast<unsigned>(backend), count_value,
          static_cast<unsigned long long>(plan->outer_window_count),
          static_cast<unsigned long long>(plan->tile_capacity),
          static_cast<unsigned long long>(plan->inner_iteration_count),
          static_cast<unsigned long long>(plan->prepared_template_count),
          static_cast<unsigned long long>(plan->prepared_command_count),
          static_cast<unsigned long long>(plan->publish_count),
          static_cast<unsigned long long>(plan->publish_bytes),
          static_cast<unsigned long long>(plan->state_bytes),
          static_cast<unsigned long long>(plan->transient_bytes));
    }
    return 3;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *plan ||
      !OutputPreparedShape(*prepared)) {
    std::fprintf(stderr,
                 "window output prepare backend=%u count=%u ok=%u reason=%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()));
    return 4;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kOutputMaximum> first_window{};
  std::array<std::uint32_t, kOutputWindowBacking> first_backing{};
  const Status first_run = prepared->run();
  const Status first_final_read =
      first_run ? prepared->read(*final, first_final) : first_run;
  const Status first_window_read =
      first_final_read ? prepared->read(*appended, first_backing)
                       : first_final_read;
  if (!first_run || !first_final_read || !first_window_read) {
    std::fprintf(stderr,
                 "window output first run backend=%u count=%u high=%u "
                 "run=%u/%u final=%u/%u window=%u/%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(high_index_only),
                 static_cast<unsigned>(first_run.ok()),
                 static_cast<unsigned>(first_run.reason()),
                 static_cast<unsigned>(first_final_read.ok()),
                 static_cast<unsigned>(first_final_read.reason()),
                 static_cast<unsigned>(first_window_read.ok()),
                 static_cast<unsigned>(first_window_read.reason()));
    return 5;
  }
  std::copy_n(first_backing.begin() + kOutputWindowOffset, kOutputMaximum,
              first_window.begin());
  const MemoryStats after_observation = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kOutputMaximum> actual_window{};
  std::array<std::uint32_t, kOutputWindowBacking> actual_backing{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_backing);
  if (window_read) {
    std::copy_n(actual_backing.begin() + kOutputWindowOffset, kOutputMaximum,
                actual_window.begin());
  }
  const std::size_t active = CeilDiv(count_value, kOutputTile);
  const auto expected_window = high_index_only
                                   ? ExpectedHighIndexWindow()
                                   : ExpectedWindow(values, count_value);
  const bool stats_match =
      stats.command_submits == (backend == Backend::Cpu ? 0u : 1u) &&
      stats.pipeline.step_count == 1u &&
      stats.pipeline.verified_step_count == 1u &&
      stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
      stats.pipeline.executed_outer_window_count == active &&
      stats.pipeline.skipped_outer_window_count == kOutputOuter - active &&
      stats.pipeline.executed_inner_iteration_count == 0u &&
      stats.pipeline.skipped_inner_iteration_count == 0u &&
      stats.pipeline.prepared_template_count == kOutputTemplates &&
      stats.pipeline.prepared_command_count == kOutputCommands &&
      stats.publication.generation == 2u &&
      stats.publication.discard_count == 0u && OutputWarmSetupClean(stats);
  if (!second || prepared->generation() != 2u || !stats_match ||
      !NoAllocation(after_observation, after_warm) || !final_read ||
      !window_read || first_final != actual_final ||
      first_window != actual_window ||
      actual_final[0u] != ExpectedFinal(values, count_value) ||
      actual_window != expected_window ||
      (high_index_only &&
       (!std::all_of(actual_window.begin(), actual_window.end() - 1u,
                     [](const std::uint32_t value) {
                       return value == kOutputSentinel;
                     }) ||
        actual_window.back() != kOutputHighIndexValue)) ||
      !std::all_of(
          actual_backing.begin(), actual_backing.begin() + kOutputWindowOffset,
          [](const std::uint32_t value) { return value == kOutputSentinel; }) ||
      !std::all_of(
          actual_backing.begin() + kOutputWindowOffset + kOutputMaximum,
          actual_backing.end(),
          [](const std::uint32_t value) { return value == kOutputSentinel; })) {
    std::fprintf(
        stderr,
        "window output run backend=%u count=%u status=%u/%u generation=%llu "
        "final=%u/%u first=%u/%u last=%u/%u outer=%llu/%llu "
        "skipped=%llu/%llu inner=%llu/%llu templates=%llu commands=%llu "
        "alloc=%llu submit=%llu\n",
        static_cast<unsigned>(backend), count_value,
        static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], ExpectedFinal(values, count_value), actual_window[0u],
        expected_window[0u], actual_window[kOutputMaximum - 1u],
        expected_window[kOutputMaximum - 1u],
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(active),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(kOutputOuter - active),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_template_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(stats.buffer_allocations),
        static_cast<unsigned long long>(stats.command_submits));
    return 6;
  }

  if (count_value == kOutputMaximum) {
    std::array<std::uint32_t, kOutputMaximum + 1u> identity_values{};
    identity_values[0u] = actual_final[0u];
    std::copy(actual_window.begin(), actual_window.end(),
              identity_values.begin() + 1u);
    const Fingerprint current_seed = seed.fingerprint();
    const Fingerprint current_fold = fold.fingerprint();
    const Fingerprint current_pipeline = prepared->fingerprint();
    const std::uint64_t current_output =
        Hash(identity_values.data(), sizeof(identity_values));
    if (!current_seed || !current_fold || !current_pipeline ||
        current_output == 0u) {
      return 7;
    }
    if (identity.seed) {
      if (identity.seed != current_seed) {
        return 8;
      }
    } else {
      identity.seed = current_seed;
    }
    Fingerprint &fold_identity =
        high_index_only ? identity.high_index_fold : identity.fold;
    Fingerprint &pipeline_identity =
        high_index_only ? identity.high_index_pipeline : identity.pipeline;
    std::uint64_t &output_identity =
        high_index_only ? identity.high_index_output : identity.output;
    if (fold_identity) {
      if (fold_identity != current_fold ||
          pipeline_identity != current_pipeline ||
          output_identity != current_output) {
        return 8;
      }
    } else {
      fold_identity = current_fold;
      pipeline_identity = current_pipeline;
      output_identity = current_output;
    }
  }
  return 0;
}

int CheckOutputTransactionalCountParity(Device &device, const OutputSeed &seed,
                                        const OutputFold &fold,
                                        const OutputUnary &advance) {
  return CheckTransactionalCountParity(device, seed, fold, advance);
}
int CheckOutputStatePairPublicationTargetRejected(Device &device,
                                                  const OutputSeed &seed,
                                                  const OutputFold &fold) {
  return CheckStatePairPublicationTargetRejected(device, seed, fold);
}
int CheckOutputCount(Device &device, const Backend backend,
                     const OutputSeed &seed, const OutputFold &fold,
                     const std::uint32_t count,
                     const std::array<std::uint32_t, kOutputMaximum> &values,
                     const bool high_index_only,
                     WindowOutputIdentity &identity) {
  return CheckCount(device, backend, seed, fold, count, values, high_index_only,
                    identity);
}

} // namespace rund::node::test_contract::window
