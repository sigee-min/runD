#include "../pipeline/local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {
namespace {

constexpr std::size_t kMaximum = 16u;
constexpr std::size_t kTile = 4u;
constexpr std::size_t kOuter = kMaximum / kTile;
constexpr std::size_t kTemplates = kOuter + 3u;
constexpr std::size_t kCommands = 2u * kOuter;
constexpr std::size_t kWindowOffset = 2u;
constexpr std::size_t kWindowBacking = kMaximum + 4u;
constexpr std::uint32_t kInitial = 7u;
constexpr std::uint32_t kSentinel = 0xA5A55A5Au;
constexpr std::uint32_t kWindowBias = 1000u;
constexpr std::uint32_t kHighIndexValue = 0xF0000001u;
constexpr std::array<std::uint32_t, kMaximum> kValues{
    11u, 22u,  22u,  44u,  55u,  66u,  77u,  88u,
    99u, 110u, 121u, 132u, 143u, 154u, 165u, 0xF0000001u};
constexpr std::array<std::uint32_t, kTile> kLanes{0u, 1u, 2u, 3u};
constexpr std::array<std::uint32_t, 6u> kCounts{0u, 1u, 3u, 4u, 5u, 16u};

static_assert(kOuter == 4u);
static_assert(kTemplates == 7u);
static_assert(kCommands == 8u);

template <bool Fault> [[nodiscard]] auto SeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kMaximum)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto lanes, auto witness, auto total,
                 auto ordinal) {
        auto current = resident<kMaximum, kTile>(total, ordinal);
        auto indices =
            lanes.combine("window-output-index", current.base(),
                          [](auto lane, auto base) { return lane + base; });
        auto tile = values.gather(indices);
        auto produced = [&] {
          if constexpr (Fault) {
            auto fault =
                ordinal.map("window-output-seed-fault-index", [](auto value) {
                  return select(value == 1u, 1u, 0u);
                });
            auto checked = witness.gather(fault).scalar();
            return tile.combine(
                "window-output-seed-fault", checked,
                [](auto value, auto check) { return value + check * 0u; });
          } else {
            return tile.map("window-output-seed-tile",
                            [](auto value) { return value; });
          }
        }();
        auto count = current.count().map("window-output-seed-count",
                                         [](auto value) { return value; });
        auto canonical_ordinal = current.ordinal().map(
            "window-output-seed-ordinal", [](auto value) { return value; });
        auto retained = witness.map("window-output-seed-witness",
                                    [](auto value) { return value; });
        return outputs(produced, count, canonical_ordinal, retained);
      })
      .compile();
}

template <bool Fault> [[nodiscard]] auto ActionProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto tile, auto count, auto ordinal, auto witness) {
        auto produced = [&] {
          if constexpr (Fault) {
            auto fault =
                ordinal.map("window-output-action-fault-index", [](auto value) {
                  return select(value == 1u, 1u, 0u);
                });
            auto checked = witness.gather(fault).scalar();
            return tile.combine(
                "window-output-action-fault", checked,
                [](auto value, auto check) { return value + check * 0u; });
          } else {
            return tile.map("window-output-action-tile",
                            [](auto value) { return value; });
          }
        }();
        auto next_count = count.map("window-output-action-count",
                                    [](auto value) { return value; });
        auto next_ordinal = ordinal.map("window-output-action-ordinal",
                                        [](auto value) { return value; });
        auto next_witness = witness.map("window-output-action-witness",
                                        [](auto value) { return value; });
        return outputs(produced, next_count, next_ordinal, next_witness);
      })
      .compile();
}

template <bool TwoWindows = false, bool HighIndexOnly = false>
[[nodiscard]] auto FoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto count, auto ordinal,
                 auto witness) {
        (void)witness;
        auto active =
            tile.indices().combine("window-output-active", count.scalar(),
                                   [](auto lane, auto logical) {
                                     return select(lane < logical, 1u, 0u);
                                   });
        auto masked = tile.combine("window-output-mask", active,
                                   [](auto value, auto enabled) {
                                     return select(enabled != 0u, value, 0u);
                                   });
        auto sum = masked.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto first = [&] {
          if constexpr (HighIndexOnly) {
            return tile.indices().combine(
                "window-output-high-index", ordinal.scalar(),
                [](auto lane, auto current) {
                  return select(lane == kTile - 1u,
                                select(current == kOuter - 1u, kHighIndexValue,
                                       kSentinel),
                                kSentinel);
                });
          } else {
            return tile.map("window-output-first",
                            [](auto value) { return value + kWindowBias; });
          }
        }();
        if constexpr (TwoWindows) {
          auto second = tile.combine(
              "window-output-second", active,
              [](auto value, auto enabled) { return value + enabled; });
          return outputs(next, first, second);
        } else {
          return outputs(next, first);
        }
      })
      .compile();
}

[[nodiscard]] auto PriorityFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto count, auto ordinal,
                 auto witness) {
        (void)count;
        (void)witness;
        auto targets = tile.indices().combine(
            "window-output-priority-target", ordinal.scalar(),
            [](auto lane, auto current) {
              // Outer 1 has two competing Scatter failures: target 4 is out
              // of range at source ordinal 1, while target 0 is duplicated at
              // source ordinal 2. The canonical failure key must retain the
              // higher-priority ordinal-1 range evidence even if the duplicate
              // is recorded first by a physically concurrent GPU lane.
              const auto failed =
                  select(lane == 1u, static_cast<std::uint32_t>(kTile),
                         select(lane == 2u, 0u, select(lane == 3u, 2u, lane)));
              return select(current == 1u, failed, lane);
            });
        auto sum = tile.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-priority-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto scattered = tile.scatter(targets, {.count = kTile});
        return outputs(next, scattered);
      })
      .compile();
}

[[nodiscard]] auto ScatterSeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto targets, auto total, auto ordinal) {
        (void)total;
        auto selected =
            values.combine("window-output-scatter-select", ordinal.scalar(),
                           [](auto value, auto outer) {
                             return select(outer == kOuter - 1u, value, 0u);
                           });
        auto retained = targets.map("window-output-scatter-target",
                                    [](auto target) { return target; });
        return outputs(selected, retained);
      })
      .compile();
}

[[nodiscard]] auto ScatterFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(kTile)
      .branch([](auto outer, auto values, auto targets) {
        auto sum = values.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-scatter-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto scattered = values.scatter_reduce(targets, kTile, Reduce::Sum);
        return outputs(next, scattered);
      })
      .compile();
}

[[nodiscard]] auto DownstreamProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("window-output-downstream", kMaximum,
                          [](auto value) { return value; })
      .compile();
}

[[nodiscard]] auto ZeroPrefixSeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto total, auto ordinal) {
        (void)ordinal;
        auto logical = total.map("window-output-zero-seed-count",
                                 [](auto value) { return value; });
        auto tile = values.map("window-output-zero-seed-tile",
                               [](auto value) { return value; });
        return outputs(logical, tile);
      })
      .compile();
}

[[nodiscard]] auto ZeroPrefixFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .branch([](auto outer, auto logical, auto tile) {
        (void)logical;
        auto next = outer.map("window-output-zero-fold-state",
                              [](auto value) { return value; });
        auto appended = tile.map("window-output-zero-fold-tile",
                                 [](auto value) { return value; });
        return outputs(next, appended);
      })
      .compile();
}

[[nodiscard]] constexpr std::uint32_t
ExpectedFinal(const std::array<std::uint32_t, kMaximum> &values,
              const std::uint32_t count) noexcept {
  std::uint32_t result = kInitial;
  for (std::size_t index = 0u; index < count; ++index) {
    result += values[index];
  }
  return result;
}

[[nodiscard]] constexpr std::array<std::uint32_t, kMaximum>
ExpectedWindow(const std::array<std::uint32_t, kMaximum> &values,
               const std::uint32_t count) noexcept {
  std::array<std::uint32_t, kMaximum> result{};
  result.fill(kSentinel);
  for (std::size_t index = 0u; index < count; ++index) {
    result[index] = values[index] + kWindowBias;
  }
  return result;
}

[[nodiscard]] constexpr std::array<std::uint32_t, kMaximum>
ExpectedHighIndexWindow() noexcept {
  std::array<std::uint32_t, kMaximum> result{};
  result.fill(kSentinel);
  result[kMaximum - 1u] = kHighIndexValue;
  return result;
}

[[nodiscard]] bool WarmSetupClean(const rund::compute::Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u && stats.uploaded_bytes == 0u &&
         stats.download_events == 0u && stats.downloaded_bytes == 0u &&
         stats.pipeline.rebinding_count == 0u;
}

[[nodiscard]] bool PreparedShape(const rund::compute::Pipeline &pipeline) {
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->steps.size() != kTemplates ||
      state->windows.size() != 1u || state->publications.size() != 2u) {
    return false;
  }
  const PipelineWindow &window = state->windows[0u];
  if (!window.nested || window.begin != 0u || window.end != kTemplates ||
      window.seed_first != 0u || window.seed_count != kOuter ||
      window.action_first != kOuter || window.action_count != 0u ||
      window.fold_first != kOuter || window.recurrent_output_count != 1u ||
      window.maximum != kMaximum || window.tile != kTile) {
    return false;
  }
  for (std::size_t index = 0u; index < state->steps.size(); ++index) {
    const PipelineRoute expected =
        index < kOuter ? PipelineRoute::NestedSeed : PipelineRoute::NestedFold;
    if (state->steps[index].route != expected ||
        state->steps[index].job == nullptr) {
      return false;
    }
  }
  const auto terminal =
      std::find_if(state->publications.begin(), state->publications.end(),
                   [](const PipelinePublish &value) {
                     return value.kind == PipelinePublishKind::Terminal;
                   });
  const auto appended =
      std::find_if(state->publications.begin(), state->publications.end(),
                   [](const PipelinePublish &value) {
                     return value.kind == PipelinePublishKind::Window;
                   });
  return terminal != state->publications.end() &&
         appended != state->publications.end() && terminal->count == 1u &&
         terminal->resident_count == nullptr && terminal->maximum == 0u &&
         terminal->tile == 0u && appended->count == kTile &&
         appended->resident_count == window.count &&
         appended->resident_count_offset == window.count_offset &&
         appended->maximum == kMaximum && appended->tile == kTile &&
         appended->target_stride == 1u;
}

template <class Seed, class Fold>
[[nodiscard]] int
CheckCount(Device &device, const Backend backend, const Seed &seed,
           const Fold &fold, const std::uint32_t count_value,
           const std::array<std::uint32_t, kMaximum> &values,
           const bool high_index_only, WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  const std::array<std::uint32_t, 3u> count_values{0x13579BDFu, count_value,
                                                   0x2468ACE0u};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kWindowBacking> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto input_values = device.upload<std::uint32_t>(values);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !input_values || !lanes || !witness || !count || !final ||
      !appended) {
    return 1;
  }
  auto count_view = count->view(1u, 1u);
  auto appended_view = appended->view(kWindowOffset, kMaximum);
  if (!count_view || !appended_view) {
    return 2;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(
      body, rund::compute::window(*count_view),
      read(*outer, *input_values, *lanes, *witness), write_final(*final),
      write_window(*appended_view));
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != kOuter ||
      plan->tile_capacity != kTile || plan->inner_iteration_count != 0u ||
      plan->prepared_template_count != kTemplates ||
      plan->prepared_command_count != kCommands ||
      plan->publish_count != kOuter + 1u ||
      plan->publish_bytes != (kMaximum + 1u) * sizeof(std::uint32_t)) {
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
  if (!prepared || prepared->plan() != *plan || !PreparedShape(*prepared)) {
    std::fprintf(stderr,
                 "window output prepare backend=%u count=%u ok=%u reason=%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()));
    return 4;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kMaximum> first_window{};
  std::array<std::uint32_t, kWindowBacking> first_backing{};
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
  std::copy_n(first_backing.begin() + kWindowOffset, kMaximum,
              first_window.begin());
  const MemoryStats after_observation = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kMaximum> actual_window{};
  std::array<std::uint32_t, kWindowBacking> actual_backing{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_backing);
  if (window_read) {
    std::copy_n(actual_backing.begin() + kWindowOffset, kMaximum,
                actual_window.begin());
  }
  const std::size_t active = CeilDiv(count_value, kTile);
  const auto expected_window = high_index_only
                                   ? ExpectedHighIndexWindow()
                                   : ExpectedWindow(values, count_value);
  const bool stats_match =
      stats.command_submits == (backend == Backend::Cpu ? 0u : 1u) &&
      stats.pipeline.step_count == 1u &&
      stats.pipeline.verified_step_count == 1u &&
      stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
      stats.pipeline.executed_outer_window_count == active &&
      stats.pipeline.skipped_outer_window_count == kOuter - active &&
      stats.pipeline.executed_inner_iteration_count == 0u &&
      stats.pipeline.skipped_inner_iteration_count == 0u &&
      stats.pipeline.prepared_template_count == kTemplates &&
      stats.pipeline.prepared_command_count == kCommands &&
      stats.publication.generation == 2u &&
      stats.publication.discard_count == 0u && WarmSetupClean(stats);
  if (!second || prepared->generation() != 2u || !stats_match ||
      !NoAllocation(after_observation, after_warm) || !final_read ||
      !window_read || first_final != actual_final ||
      first_window != actual_window ||
      actual_final[0u] != ExpectedFinal(values, count_value) ||
      actual_window != expected_window ||
      (high_index_only &&
       (!std::all_of(
            actual_window.begin(), actual_window.end() - 1u,
            [](const std::uint32_t value) { return value == kSentinel; }) ||
        actual_window.back() != kHighIndexValue)) ||
      !std::all_of(
          actual_backing.begin(), actual_backing.begin() + kWindowOffset,
          [](const std::uint32_t value) { return value == kSentinel; }) ||
      !std::all_of(actual_backing.begin() + kWindowOffset + kMaximum,
                   actual_backing.end(), [](const std::uint32_t value) {
                     return value == kSentinel;
                   })) {
    std::fprintf(
        stderr,
        "window output run backend=%u count=%u status=%u/%u generation=%llu "
        "final=%u/%u first=%u/%u last=%u/%u outer=%llu/%llu "
        "skipped=%llu/%llu inner=%llu/%llu templates=%llu commands=%llu "
        "rebind=%llu alloc=%llu submit=%llu\n",
        static_cast<unsigned>(backend), count_value,
        static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], ExpectedFinal(values, count_value), actual_window[0u],
        expected_window[0u], actual_window[kMaximum - 1u],
        expected_window[kMaximum - 1u],
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(active),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(kOuter - active),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_template_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(stats.pipeline.rebinding_count),
        static_cast<unsigned long long>(stats.buffer_allocations),
        static_cast<unsigned long long>(stats.command_submits));
    return 6;
  }

  if (count_value == kMaximum) {
    std::array<std::uint32_t, kMaximum + 1u> identity_values{};
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

template <class Seed, class Fold, class Downstream>
[[nodiscard]] int CheckDownstreamRead(Device &device, const Backend backend,
                                      const Seed &seed, const Fold &fold,
                                      const Downstream &downstream,
                                      WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::uint32_t count_value = 5u;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{count_value};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
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
  baseline.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                    read(*outer, *values, *lanes, *witness),
                                    write_final(*final),
                                    write_window(*appended));
  const auto baseline_plan = baseline.plan();

  auto builder = pipeline(device);
  builder
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *values, *lanes, *witness),
                                write_final(*final), write_window(*appended))
      .then(downstream, read(*appended), write(*sink));
  const auto plan = builder.plan();
  if (!baseline_plan || !plan ||
      plan->prepared_template_count !=
          baseline_plan->prepared_template_count + 1u ||
      plan->prepared_command_count !=
          baseline_plan->prepared_command_count + 1u ||
      plan->barrier_count != baseline_plan->barrier_count + 1u ||
      plan->resource_count != baseline_plan->resource_count + 1u ||
      plan->persistent_bytes !=
          baseline_plan->persistent_bytes + kMaximum * sizeof(std::uint32_t) ||
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
      state->steps.size() != kTemplates + 1u ||
      state->windows[0u].recurrent_output_count != 1u ||
      state->barriers.back() == 0u) {
    return 4;
  }
  const auto publication =
      std::find_if(state->publications.begin(), state->publications.end(),
                   [](const PipelinePublish &value) {
                     return value.kind == PipelinePublishKind::Window;
                   });
  if (publication == state->publications.end()) {
    return 5;
  }
  const auto resource =
      std::find_if(state->resources.begin(), state->resources.end(),
                   [&](const PipelineResource &value) {
                     return value.buffer == publication->target;
                   });
  if (resource == state->resources.end()) {
    return 6;
  }
  const auto resource_index =
      static_cast<std::uint32_t>(resource - state->resources.begin());
  const PipelineWindow &window = state->windows[0u];
  const bool has_publication_read_hazard =
      std::any_of(state->dependencies.begin(), state->dependencies.end(),
                  [&](const PipelineDependency &dependency) {
                    return dependency.resource == resource_index &&
                           dependency.before >= window.fold_first &&
                           dependency.before < window.end &&
                           dependency.after == kTemplates &&
                           dependency.before_access == PipelineAccess::Write &&
                           dependency.after_access == PipelineAccess::Read;
                  });
  if (!has_publication_read_hazard) {
    return 7;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kMaximum> first_window{};
  std::array<std::uint32_t, kMaximum> first_sink{};
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
  std::array<std::uint32_t, kMaximum> actual_window{};
  std::array<std::uint32_t, kMaximum> actual_sink{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_window);
  const Status sink_read = prepared->read(*sink, actual_sink);
  const auto expected_window = ExpectedWindow(kValues, count_value);
  if (!second || !final_read || !window_read || !sink_read ||
      prepared->generation() != 2u ||
      actual_final[0u] != ExpectedFinal(kValues, count_value) ||
      actual_window != expected_window || actual_sink != expected_window ||
      first_final != actual_final || first_window != actual_window ||
      first_sink != actual_sink || !NoAllocation(before_warm, after_warm) ||
      !WarmSetupClean(stats) || stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.barrier_count != plan->barrier_count ||
      stats.pipeline.executed_outer_window_count != 2u ||
      stats.pipeline.skipped_outer_window_count != kOuter - 2u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window output downstream run backend=%u status=%u/%u generation=%llu "
        "final=%u/%u window=%u/%u sink=%u/%u steps=%llu/%llu "
        "barriers=%llu/%llu outer=%llu/%llu skipped=%llu/%llu submit=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], ExpectedFinal(kValues, count_value),
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
        static_cast<unsigned long long>(kOuter - 2u),
        static_cast<unsigned long long>(stats.command_submits));
    return 9;
  }

  std::array<std::uint32_t, 2u * kMaximum + 1u> raw{};
  raw[0u] = actual_final[0u];
  std::copy(actual_window.begin(), actual_window.end(), raw.begin() + 1u);
  std::copy(actual_sink.begin(), actual_sink.end(),
            raw.begin() + 1u + kMaximum);
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
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *values, *lanes, *witness),
                                write_final(*final), write_window(*appended))
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
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> count_values{0u};
  constexpr std::array<std::uint32_t, kTile> tile_values{3u, 5u, 7u, 11u};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);

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
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer, *tile), write_final(*final),
                                   write_window(*appended));
  const auto plan = builder.plan();
  if (!plan || plan->prepared_template_count != kTemplates ||
      plan->prepared_command_count != kCommands ||
      plan->publish_count != kOuter + 1u ||
      plan->publish_bytes != (kMaximum + 1u) * sizeof(std::uint32_t)) {
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
      state->steps[state->windows[0u].fold_first].job == nullptr) {
    return 4;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kMaximum> first_window{};
  if (!prepared->run() || !prepared->read(*final, first_final) ||
      !prepared->read(*appended, first_window)) {
    return 5;
  }
  const MemoryStats before_warm = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kMaximum> actual_window{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_window);
  if (!second || !final_read || !window_read || prepared->generation() != 2u ||
      actual_final != initial || actual_window != window_values ||
      first_final != actual_final || first_window != actual_window ||
      !NoAllocation(before_warm, after_warm) || !WarmSetupClean(stats) ||
      stats.pipeline.step_count != 1u ||
      stats.pipeline.verified_step_count != 1u ||
      stats.pipeline.executed_outer_window_count != 0u ||
      stats.pipeline.skipped_outer_window_count != kOuter ||
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
        static_cast<unsigned long long>(kOuter),
        static_cast<unsigned long long>(stats.command_submits));
    return 6;
  }

  std::array<std::uint32_t, kMaximum + 1u> raw{};
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

template <class Seed, class Fold>
[[nodiscard]] int CheckScatterConflicts(Device &device, const Backend backend,
                                        const Seed &seed, const Fold &fold,
                                        WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::array<std::uint32_t, kTile>, 2u> values{{
      {5u, 5u, 0u, 0u},
      {5u, 7u, 11u, 13u},
  }};
  constexpr std::array<std::array<std::uint32_t, kTile>, 2u> targets{{
      {3u, 3u, 0u, 1u},
      {3u, 3u, 3u, 3u},
  }};
  constexpr std::array<std::uint32_t, 2u> expected{10u, 36u};
  constexpr std::array<std::uint32_t, 1u> count_value{kMaximum};
  constexpr std::array<std::uint32_t, 1u> initial{0u};
  std::array<std::uint64_t, 2u> output_hashes{};
  Fingerprint pipeline_fingerprint{};

  for (std::size_t scenario = 0u; scenario < values.size(); ++scenario) {
    std::array<std::uint32_t, kMaximum> initial_window{};
    initial_window.fill(kSentinel);
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
    builder.windows<kMaximum, kTile>(
        body, rund::compute::window(*count),
        read(*outer, *input_values, *input_targets), write_final(*final),
        write_window(*window_target));
    const auto plan = builder.plan();
    if (!plan || plan->outer_window_count != kOuter ||
        plan->prepared_template_count != kTemplates ||
        plan->prepared_command_count != kCommands ||
        plan->publish_count != kOuter + 1u ||
        plan->publish_bytes != (kMaximum + 1u) * sizeof(std::uint32_t)) {
      return 2;
    }
    auto prepared = std::move(builder).prepare();
    if (!prepared || !prepared->run()) {
      return 3;
    }
    std::array<std::uint32_t, 1u> first_final{};
    std::array<std::uint32_t, kMaximum> first_window{};
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
    std::array<std::uint32_t, kMaximum> actual_window{};
    std::array<std::uint32_t, kMaximum> expected_window{};
    expected_window[kMaximum - 1u] = expected[scenario];
    if (!prepared->read(*final, actual_final) ||
        !prepared->read(*window_target, actual_window) ||
        prepared->generation() != 2u ||
        actual_final[0u] != expected[scenario] ||
        actual_window != expected_window || first_final != actual_final ||
        first_window != actual_window ||
        !NoAllocation(before_warm, after_warm) || !WarmSetupClean(stats) ||
        stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
        stats.pipeline.executed_outer_window_count != kOuter ||
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
          expected[scenario], actual_window[kMaximum - 1u],
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(stats.publication.discard_count));
      return 6;
    }
    std::array<std::uint32_t, kMaximum + 1u> raw{};
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
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  const std::array<std::uint32_t, 1u> count_values{count_value};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);
  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
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
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer, *values, *lanes, *witness),
                                   write_final(*final),
                                   write_window(*appended));
  auto prepared = std::move(builder).prepare();
  const Status failed =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  std::array<std::uint32_t, kMaximum> unread{};
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
        static_cast<unsigned>(rerun.reason()), actual_final[0u], kSentinel,
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
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(scalar);
  auto count = device.upload<std::uint32_t>(scalar);
  auto final = device.upload<std::uint32_t>(scalar);
  auto shared = device.buffer<std::uint32_t>(2u * kMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !shared) {
    return 1;
  }
  auto first = shared->view(0u, kMaximum);
  auto second = shared->view(kMaximum, kMaximum);
  if (!first || !second) {
    return 2;
  }
  const auto two = tile_repeat<0u>(seed, fold_two);
  auto duplicate = pipeline(device);
  duplicate.windows<kMaximum, kTile>(two, rund::compute::window(*count),
                                     read(*outer, *values, *lanes, *witness),
                                     write_final(*final),
                                     write_window(*first, *second));
  const auto duplicate_plan = duplicate.plan();
  if (duplicate_plan || duplicate_plan.reason() != Reason::BindingDuplicate) {
    std::fprintf(stderr, "window output duplicate reason=%u\n",
                 static_cast<unsigned>(duplicate_plan.reason()));
    return 3;
  }

  std::array<std::uint32_t, kMaximum> aliased_values{};
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
  overlap.windows<kMaximum, kTile>(one, rund::compute::window(*count_view),
                                   read(*outer, *values, *lanes, *witness),
                                   write_final(*final), write_window(*aliased));
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

} // namespace

[[nodiscard]] int CheckWindowOutput(Device &device, const Backend backend,
                                    WindowOutputIdentity &identity) {
  auto seed = SeedProgram<false>(device);
  auto seed_fault = SeedProgram<true>(device);
  auto action_fault = ActionProgram<true>(device);
  auto fold = FoldProgram(device);
  auto high_index_fold = FoldProgram<false, true>(device);
  auto fold_two = FoldProgram<true>(device);
  auto priority_fold = PriorityFoldProgram(device);
  auto scatter_seed = ScatterSeedProgram(device);
  auto scatter_fold = ScatterFoldProgram(device);
  auto downstream = DownstreamProgram(device);
  auto zero_seed = ZeroPrefixSeedProgram(device);
  auto zero_fold = ZeroPrefixFoldProgram(device);
  if (!seed || !seed_fault || !action_fault || !fold || !high_index_fold ||
      !fold_two || !priority_fold || !scatter_seed || !scatter_fold ||
      !downstream || !zero_seed || !zero_fold) {
    return 1;
  }
  for (const std::uint32_t count : kCounts) {
    const int checked = CheckCount(device, backend, *seed, *fold, count,
                                   kValues, false, identity);
    if (checked != 0) {
      return 10 + checked;
    }
  }
  if (const int high_index =
          CheckCount(device, backend, *seed, *high_index_fold, kMaximum,
                     kValues, true, identity);
      high_index != 0) {
    return 20 + high_index;
  }
  if (const int scatter = CheckScatterConflicts(device, backend, *scatter_seed,
                                                *scatter_fold, identity);
      scatter != 0) {
    return 30 + scatter;
  }
  if (const int downstream_read = CheckDownstreamRead(
          device, backend, *seed, *fold, *downstream, identity);
      downstream_read != 0) {
    return 40 + downstream_read;
  }
  if (const int zero_prefix = CheckZeroRecurrentPrefix(
          device, backend, *zero_seed, *zero_fold, identity);
      zero_prefix != 0) {
    return 60 + zero_prefix;
  }
  const auto seed_failure = rund::compute::tile_repeat<0u>(*seed_fault, *fold);
  if (const int failure = CheckLateFailure(
          device, backend, seed_failure, 5u, Reason::GatherIndexOutOfRange,
          rund::compute::PipelineNestedPhase::Seed,
          rund::compute::PipelineStats::no_coordinate);
      failure != 0) {
    return 80 + failure;
  }
  const auto action_failure =
      rund::compute::tile_repeat<1u>(*seed, *action_fault, *fold);
  if (const int failure = CheckLateFailure(
          device, backend, action_failure, 5u, Reason::GatherIndexOutOfRange,
          rund::compute::PipelineNestedPhase::Action, 0u);
      failure != 0) {
    return 90 + failure;
  }
  const auto priority_failure =
      rund::compute::tile_repeat<0u>(*seed, *priority_fold);
  if (const int failure = CheckLateFailure(
          device, backend, priority_failure, 8u, Reason::ScatterIndexOutOfRange,
          rund::compute::PipelineNestedPhase::Fold,
          rund::compute::PipelineStats::no_coordinate);
      failure != 0) {
    return 100 + failure;
  }
  const int aliases = CheckAliases(device, *seed, *fold, *fold_two);
  return aliases == 0 ? 0 : 110 + aliases;
}

} // namespace rund::node::test_contract::window
