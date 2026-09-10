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

[[nodiscard]] int
CheckNestedAggregateStats(rund::compute::Device &device,
                          const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t first_maximum = 5u;
  constexpr std::size_t first_tile = 3u;
  constexpr std::size_t first_inner = 2u;
  constexpr std::size_t first_outer = CeilDiv(first_maximum, first_tile);
  constexpr std::size_t second_maximum = 8u;
  constexpr std::size_t second_tile = 3u;
  constexpr std::size_t second_inner = 4u;
  constexpr std::size_t second_outer = CeilDiv(second_maximum, second_tile);
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

  auto seed = MakeNestedTerminalSeedProgram(device);
  auto action = on(device)
                    .map<std::uint32_t>("nested-window-aggregate-action", 1u,
                                        [](auto value) { return value + 1u; })
                    .compile();
  auto fold = MakeNestedFailureFoldProgram(device, false);
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
  auto builder = pipeline(device);
  builder
      .windows<first_maximum, first_tile>(
          first_body, rund::compute::window(*first_count),
          read(*first_outer_seed), write_final(*first_output))
      .windows<second_maximum, second_tile>(
          second_body, rund::compute::window(*second_count),
          read(*second_outer_seed), write_final(*second_output));
  auto prepared = std::move(builder).prepare();
  std::array<std::uint32_t, 1u> first_actual{};
  std::array<std::uint32_t, 1u> second_actual{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  if (!prepared || !ran || !prepared->read(*first_output, first_actual) ||
      !prepared->read(*second_output, second_actual) ||
      first_actual != first_initial || second_actual[0u] != second_expected ||
      stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.executed_outer_window_count != executed_outer ||
      stats.pipeline.skipped_outer_window_count != skipped_outer ||
      stats.pipeline.executed_inner_iteration_count != executed_inner ||
      stats.pipeline.skipped_inner_iteration_count != skipped_inner ||
      stats.control.iteration_count != executed_outer ||
      stats.control.skipped_iteration_count != skipped_outer ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested aggregate backend=%u prepared=%u status=%u reason=%u "
        "outputs=%u/%u:%u/%u outer=%llu/%llu:%llu/%llu "
        "inner=%llu/%llu:%llu/%llu control=%llu/%llu submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        first_actual[0u], first_initial[0u], second_actual[0u], second_expected,
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
    return 2;
  }
  return 0;
}

} // namespace rund::node::test_contract::window
