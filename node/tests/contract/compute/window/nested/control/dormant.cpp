#include "../../../pipeline/local.hpp"
#include "../../local.hpp"
#include "../local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

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

} // namespace rund::node::test_contract::window
