#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace runtime_compute_pipeline {
namespace {

[[nodiscard]] auto NestedSeed(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto total, auto ordinal) {
        (void)total;
        return ordinal.map("pipeline-nested-stats-seed",
                           [](auto value) { return value + 1u; });
      })
      .compile();
}

[[nodiscard]] auto NestedAction(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("pipeline-nested-stats-action", 1u,
                          [](auto value) { return value + 1u; })
      .compile();
}

[[nodiscard]] auto NestedFold(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile) {
        return outer.combine(
            "pipeline-nested-stats-fold", tile,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

} // namespace

int NestedWorkTotals(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t first_maximum = 5u;
  constexpr std::size_t first_tile = 3u;
  constexpr std::size_t first_inner = 2u;
  constexpr std::size_t first_outer =
      (first_maximum + first_tile - 1u) / first_tile;
  constexpr std::size_t second_maximum = 8u;
  constexpr std::size_t second_tile = 3u;
  constexpr std::size_t second_inner = 4u;
  constexpr std::size_t second_outer =
      (second_maximum + second_tile - 1u) / second_tile;
  constexpr std::array<std::uint32_t, 1u> first_initial{100u};
  constexpr std::array<std::uint32_t, 1u> second_initial{200u};
  constexpr std::array<std::uint32_t, 1u> first_count_values{0u};
  constexpr std::array<std::uint32_t, 1u> second_count_values{4u};
  constexpr std::uint64_t executed_outer = 2u;
  constexpr std::uint64_t skipped_outer =
      first_outer + (second_outer - executed_outer);
  constexpr std::uint64_t executed_inner = executed_outer * second_inner;
  constexpr std::uint64_t skipped_inner =
      first_outer * first_inner +
      (second_outer - executed_outer) * second_inner;
  constexpr std::uint32_t second_expected =
      second_initial[0u] + (1u + second_inner) + (2u + second_inner);

  auto seed = NestedSeed(device);
  auto action = NestedAction(device);
  auto fold = NestedFold(device);
  auto first_outer_seed = device.upload<std::uint32_t>(first_initial);
  auto second_outer_seed = device.upload<std::uint32_t>(second_initial);
  auto first_count = device.upload<std::uint32_t>(first_count_values);
  auto second_count = device.upload<std::uint32_t>(second_count_values);
  auto first_output = device.buffer<std::uint32_t>(1u);
  auto second_output = device.buffer<std::uint32_t>(1u);
  if (!seed || !action || !fold || !first_outer_seed || !second_outer_seed ||
      !first_count || !second_count || !first_output || !second_output) {
    return 1;
  }

  const auto first_body = tile_repeat<first_inner>(*seed, *action, *fold);
  const auto second_body = tile_repeat<second_inner>(*seed, *action, *fold);
  auto prepared = pipeline(device)
                      .windows<first_maximum, first_tile>(
                          first_body, rund::compute::window(*first_count),
                          read(*first_outer_seed), write_final(*first_output))
                      .windows<second_maximum, second_tile>(
                          second_body, rund::compute::window(*second_count),
                          read(*second_outer_seed), write_final(*second_output))
                      .prepare();
  if (!prepared) {
    return 2;
  }

  const Completion completed = session.compute(*prepared).submit().wait();
  const Stats stats = completed.stats();
  std::array<std::uint32_t, 1u> first_actual{};
  std::array<std::uint32_t, 1u> second_actual{};
  if (!completed ||
      !ReadExact(*prepared, *first_output,
                 std::span<std::uint32_t>{first_actual}) ||
      !ReadExact(*prepared, *second_output,
                 std::span<std::uint32_t>{second_actual}) ||
      first_actual != first_initial || second_actual[0u] != second_expected ||
      stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.executed_outer_window_count != executed_outer ||
      stats.pipeline.skipped_outer_window_count != skipped_outer ||
      stats.pipeline.executed_inner_iteration_count != executed_inner ||
      stats.pipeline.skipped_inner_iteration_count != skipped_inner ||
      stats.control.iteration_count != executed_outer ||
      stats.control.skipped_iteration_count != skipped_outer ||
      stats.command_submits != 0u) {
    std::fprintf(
        stderr,
        "runtime nested totals completion=%u output=%u/%u:%u/%u "
        "outer=%llu/%llu:%llu/%llu inner=%llu/%llu:%llu/%llu "
        "control=%llu/%llu submits=%llu\n",
        static_cast<unsigned>(completed.ok()), first_actual[0u],
        first_initial[0u], second_actual[0u], second_expected,
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(executed_outer),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(skipped_outer),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(executed_inner),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(skipped_inner),
        static_cast<unsigned long long>(stats.control.iteration_count),
        static_cast<unsigned long long>(stats.control.skipped_iteration_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 3;
  }
  return 0;
}

} // namespace runtime_compute_pipeline
