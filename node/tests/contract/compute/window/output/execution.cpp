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
template <class Seed, class Fold, class Downstream>
[[nodiscard]] int CheckDownstreamRead(Device &device, const Backend backend,
                                      const Seed &seed, const Fold &fold,
                                      const Downstream &downstream,
                                      WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::uint32_t count_value = 5u;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{count_value};
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
  auto sink = device.upload<std::uint32_t>(window_values);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended || !sink) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto baseline = pipeline(device);
  baseline.windows<kOutputMaximum, kOutputTile>(
      body, rund::compute::window(*count),
      read(*outer, *values, *lanes, *witness), write_final(*final),
      write_window(*appended));
  const auto baseline_plan = baseline.plan();

  auto builder = pipeline(device);
  builder
      .windows<kOutputMaximum, kOutputTile>(
          body, rund::compute::window(*count),
          read(*outer, *values, *lanes, *witness), write_final(*final),
          write_window(*appended))
      .then(downstream, read(*appended), write(*sink));
  const auto plan = builder.plan();
  if (!baseline_plan || !plan ||
      plan->prepared_template_count !=
          baseline_plan->prepared_template_count + 1u ||
      plan->prepared_command_count !=
          baseline_plan->prepared_command_count + 1u ||
      plan->barrier_count != baseline_plan->barrier_count + 1u ||
      plan->resource_count != baseline_plan->resource_count + 1u ||
      plan->persistent_bytes != baseline_plan->persistent_bytes +
                                    kOutputMaximum * sizeof(std::uint32_t) ||
      plan->state_bytes != baseline_plan->state_bytes ||
      plan->publish_count != baseline_plan->publish_count ||
      plan->publish_bytes != baseline_plan->publish_bytes) {
    std::fprintf(
        stderr,
        "window output downstream plan backend=%u status=%u/%u reason=%u/%u "
        "templates=%llu/%llu commands=%llu/%llu barriers=%llu/%llu "
        "resources=%llu/%llu persistent=%llu/%llu state=%llu/%llu "
        "publish=%llu/%llu/%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned>(baseline_plan.ok()),
        static_cast<unsigned>(plan.ok()),
        static_cast<unsigned>(baseline_plan.reason()),
        static_cast<unsigned>(plan.reason()),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->prepared_template_count : 0u),
        static_cast<unsigned long long>(plan ? plan->prepared_template_count
                                             : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->prepared_command_count : 0u),
        static_cast<unsigned long long>(plan ? plan->prepared_command_count
                                             : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->barrier_count : 0u),
        static_cast<unsigned long long>(plan ? plan->barrier_count : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->resource_count : 0u),
        static_cast<unsigned long long>(plan ? plan->resource_count : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->persistent_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->persistent_bytes : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->state_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->state_bytes : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->publish_count : 0u),
        static_cast<unsigned long long>(plan ? plan->publish_count : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->publish_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->publish_bytes : 0u));
    return 2;
  }

  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared) {
    return 3;
  }
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->windows.size() != 1u ||
      state->steps.size() != kOutputTemplates + 1u ||
      state->windows[0u].recurrent_output_count != 1u ||
      state->barriers.back() == 0u) {
    return 4;
  }
  const auto publication = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(value);
      });
  if (publication == state->publications.end()) {
    return 5;
  }
  const auto &window_publication =
      std::get<PipelineWindowPublicationPlan>(*publication);
  const std::uint32_t resource_index =
      window_publication.target.view.identity.resource_ordinal;
  if (resource_index >= state->resources.size()) {
    return 6;
  }
  const PipelineWindow &window = state->windows[0u];
  const bool has_publication_read_hazard = std::any_of(
      state->dependencies.begin(), state->dependencies.end(),
      [&](const PipelineDependency &dependency) {
        return dependency.resource == resource_index &&
               dependency.before >= window.nested_shape.fold_first() &&
               dependency.before < window.nested_shape.end() &&
               dependency.after == kOutputTemplates &&
               dependency.before_access == PipelineAccess::Write &&
               dependency.after_access == PipelineAccess::Read;
      });
  if (!has_publication_read_hazard) {
    return 7;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kOutputMaximum> first_window{};
  std::array<std::uint32_t, kOutputMaximum> first_sink{};
  if (!prepared->run() || !prepared->read(*final, first_final) ||
      !prepared->read(*appended, first_window) ||
      !prepared->read(*sink, first_sink)) {
    return 8;
  }
  const MemoryStats before_warm = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kOutputMaximum> actual_window{};
  std::array<std::uint32_t, kOutputMaximum> actual_sink{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_window);
  const Status sink_read = prepared->read(*sink, actual_sink);
  const auto expected_window = ExpectedWindow(kOutputValues, count_value);
  if (!second || !final_read || !window_read || !sink_read ||
      prepared->generation() != 2u ||
      actual_final[0u] != ExpectedFinal(kOutputValues, count_value) ||
      actual_window != expected_window || actual_sink != expected_window ||
      first_final != actual_final || first_window != actual_window ||
      first_sink != actual_sink || !NoAllocation(before_warm, after_warm) ||
      !OutputWarmSetupClean(stats) || stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.barrier_count != plan->barrier_count ||
      stats.pipeline.executed_outer_window_count != 2u ||
      stats.pipeline.skipped_outer_window_count != kOutputOuter - 2u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window output downstream run backend=%u status=%u/%u generation=%llu "
        "final=%u/%u window=%u/%u sink=%u/%u steps=%llu/%llu "
        "barriers=%llu/%llu outer=%llu/%llu skipped=%llu/%llu submit=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], ExpectedFinal(kOutputValues, count_value),
        actual_window[0u], expected_window[0u], actual_sink[0u],
        expected_window[0u],
        static_cast<unsigned long long>(stats.pipeline.step_count), 2ull,
        static_cast<unsigned long long>(stats.pipeline.barrier_count),
        static_cast<unsigned long long>(plan->barrier_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        2ull,
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(kOutputOuter - 2u),
        static_cast<unsigned long long>(stats.command_submits));
    return 9;
  }

  std::array<std::uint32_t, 2u * kOutputMaximum + 1u> raw{};
  raw[0u] = actual_final[0u];
  std::copy(actual_window.begin(), actual_window.end(), raw.begin() + 1u);
  std::copy(actual_sink.begin(), actual_sink.end(),
            raw.begin() + 1u + kOutputMaximum);
  const Fingerprint current_program = downstream.fingerprint();
  const Fingerprint current_pipeline = prepared->fingerprint();
  const std::uint64_t current_output = Hash(raw.data(), sizeof(raw));
  if (!current_program || !current_pipeline || current_output == 0u) {
    return 10;
  }
  if (identity.downstream_program) {
    if (identity.downstream_program != current_program ||
        identity.downstream_pipeline != current_pipeline ||
        identity.downstream_output != current_output) {
      return 11;
    }
  } else {
    identity.downstream_program = current_program;
    identity.downstream_pipeline = current_pipeline;
    identity.downstream_output = current_output;
  }

  auto write_conflict = pipeline(device);
  write_conflict
      .windows<kOutputMaximum, kOutputTile>(
          body, rund::compute::window(*count),
          read(*outer, *values, *lanes, *witness), write_final(*final),
          write_window(*appended))
      .then(downstream, read(*appended), write(*appended));
  const auto conflict_plan = write_conflict.plan();
  if (conflict_plan ||
      conflict_plan.reason() != Reason::BindingAliasUnsupported) {
    return 12;
  }
  return 0;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckZeroRecurrentPrefix(Device &device,
                                           const Backend backend,
                                           const Seed &seed, const Fold &fold,
                                           WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> count_values{0u};
  constexpr std::array<std::uint32_t, kOutputTile> tile_values{3u, 5u, 7u, 11u};
  constexpr std::array<std::uint32_t, 1u> final_values{kOutputSentinel};
  std::array<std::uint32_t, kOutputMaximum> window_values{};
  window_values.fill(kOutputSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto tile = device.upload<std::uint32_t>(tile_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !tile || !count || !final || !appended) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.windows<kOutputMaximum, kOutputTile>(
      body, rund::compute::window(*count), read(*outer, *tile),
      write_final(*final), write_window(*appended));
  const auto plan = builder.plan();
  if (!plan || plan->prepared_template_count != kOutputTemplates ||
      plan->prepared_command_count != kOutputCommands ||
      plan->publish_count != kOutputOuter + 1u ||
      plan->publish_bytes != (kOutputMaximum + 1u) * sizeof(std::uint32_t)) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared) {
    return 3;
  }
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->windows.size() != 1u ||
      state->windows[0u].recurrent_output_count != 1u ||
      state->steps[state->windows[0u].nested_shape.fold_first()].job ==
          nullptr) {
    return 4;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kOutputMaximum> first_window{};
  if (!prepared->run() || !prepared->read(*final, first_final) ||
      !prepared->read(*appended, first_window)) {
    return 5;
  }
  const MemoryStats before_warm = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kOutputMaximum> actual_window{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_window);
  if (!second || !final_read || !window_read || prepared->generation() != 2u ||
      actual_final != initial || actual_window != window_values ||
      first_final != actual_final || first_window != actual_window ||
      !NoAllocation(before_warm, after_warm) || !OutputWarmSetupClean(stats) ||
      stats.pipeline.step_count != 1u ||
      stats.pipeline.verified_step_count != 1u ||
      stats.pipeline.executed_outer_window_count != 0u ||
      stats.pipeline.skipped_outer_window_count != kOutputOuter ||
      stats.pipeline.executed_inner_iteration_count != 0u ||
      stats.pipeline.skipped_inner_iteration_count != 0u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window output zero prefix backend=%u status=%u/%u generation=%llu "
        "final=%u/%u window=%u/%u outer=%llu skipped=%llu/%llu submit=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], initial[0u], actual_window[0u], window_values[0u],
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(kOutputOuter),
        static_cast<unsigned long long>(stats.command_submits));
    return 6;
  }

  std::array<std::uint32_t, kOutputMaximum + 1u> raw{};
  raw[0u] = actual_final[0u];
  std::copy(actual_window.begin(), actual_window.end(), raw.begin() + 1u);
  const Fingerprint current_seed = seed.fingerprint();
  const Fingerprint current_fold = fold.fingerprint();
  const Fingerprint current_pipeline = prepared->fingerprint();
  const std::uint64_t current_output = Hash(raw.data(), sizeof(raw));
  if (!current_seed || !current_fold || !current_pipeline ||
      current_output == 0u) {
    return 7;
  }
  if (identity.zero_seed) {
    if (identity.zero_seed != current_seed ||
        identity.zero_fold != current_fold ||
        identity.zero_pipeline != current_pipeline ||
        identity.zero_output != current_output) {
      return 8;
    }
  } else {
    identity.zero_seed = current_seed;
    identity.zero_fold = current_fold;
    identity.zero_pipeline = current_pipeline;
    identity.zero_output = current_output;
  }
  return 0;
}

int CheckOutputDownstreamRead(Device &device, const Backend backend,
                              const OutputSeed &seed, const OutputFold &fold,
                              const OutputUnary &downstream,
                              WindowOutputIdentity &identity) {
  return CheckDownstreamRead(device, backend, seed, fold, downstream, identity);
}
int CheckOutputZeroRecurrentPrefix(Device &device, const Backend backend,
                                   const OutputZero &seed,
                                   const OutputZero &fold,
                                   WindowOutputIdentity &identity) {
  return CheckZeroRecurrentPrefix(device, backend, seed, fold, identity);
}

} // namespace rund::node::test_contract::window
