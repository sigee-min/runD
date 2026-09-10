#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckActionFreeWindowOutput(rund::compute::Device &device) {
  constexpr std::size_t maximum = 6u;
  constexpr std::size_t tile = 2u;
  constexpr std::uint32_t sentinel = 0xA5A55A5Au;
  constexpr std::array<std::uint32_t, maximum> values{3u,  5u,  7u,
                                                      11u, 13u, 17u};
  constexpr std::array<std::uint32_t, tile> lanes{0u, 1u};
  constexpr std::array<std::uint32_t, 1u> outer_seed{19u};
  constexpr std::array<std::uint32_t, 1u> count_value{5u};
  constexpr std::array<std::uint32_t, maximum> empty{
      sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};

  auto seed = rund::compute::on(device)
                  .input<std::uint32_t>(maximum)
                  .zip_input<std::uint32_t>(tile)
                  .zip_input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto source, auto lane, auto total, auto ordinal) {
                    auto current =
                        rund::compute::resident<maximum, tile>(total, ordinal);
                    auto indices = lane.combine(
                        "pipeline-installed-window-index", current.base(),
                        [](auto local, auto base) { return local + base; });
                    auto selected = source.gather(indices);
                    auto active =
                        current.count().map("pipeline-installed-window-count",
                                            [](auto value) { return value; });
                    return rund::compute::outputs(selected, active);
                  })
                  .compile();
  auto fold =
      rund::compute::on(device)
          .input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(tile)
          .zip_input<std::uint32_t>(1u)
          .branch([](auto outer, auto selected, auto count) {
            auto enabled = selected.indices().combine(
                "pipeline-installed-window-active", count.scalar(),
                [](auto lane, auto active) {
                  return rund::compute::select(lane < active, 1u, 0u);
                });
            auto masked = selected.combine(
                "pipeline-installed-window-mask", enabled,
                [](auto value, auto active) {
                  return rund::compute::select(active != 0u, value, 0u);
                });
            auto sum = masked.reduce(rund::compute::Reduce::Sum);
            auto next = outer.combine(
                "pipeline-installed-window-fold", sum,
                [](auto state, auto value) { return state + value; });
            auto second = selected.map("pipeline-installed-window-second",
                                       [](auto value) { return value + 1u; });
            auto third = selected.map("pipeline-installed-window-third",
                                      [](auto value) { return value + 2u; });
            return rund::compute::outputs(next, selected, second, third);
          })
          .compile();
  auto source = device.upload<std::uint32_t>(values);
  auto lane = device.upload<std::uint32_t>(lanes);
  auto outer = device.upload<std::uint32_t>(outer_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto windows_first = device.upload<std::uint32_t>(empty);
  auto windows_second = device.upload<std::uint32_t>(empty);
  auto windows_third = device.upload<std::uint32_t>(empty);
  if (!seed || !fold || !source || !lane || !outer || !count || !final ||
      !windows_first || !windows_second || !windows_third) {
    return 1;
  }

  const auto body = rund::compute::tile_repeat<0u>(*seed, *fold);
  auto builder = rund::compute::pipeline(device);
  builder.windows<maximum, tile>(body, rund::compute::window(*count),
                                 rund::compute::read(*outer, *source, *lane),
                                 rund::compute::write_final(*final),
                                 rund::compute::write_window(*windows_first,
                                                             *windows_second,
                                                             *windows_third));
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != 3u ||
      plan->inner_iteration_count != 0u ||
      plan->prepared_template_count != 6u ||
      plan->prepared_command_count != 6u) {
    return 2;
  }
  auto prepared = std::move(builder).prepare();
  if (!prepared || !prepared->run()) {
    return 3;
  }
  std::array<std::uint32_t, 1u> observed_final{};
  std::array<std::uint32_t, maximum> observed_first{};
  std::array<std::uint32_t, maximum> observed_second{};
  std::array<std::uint32_t, maximum> observed_third{};
  if (!prepared->read(*final, observed_final) ||
      !prepared->read(*windows_first, observed_first) ||
      !prepared->read(*windows_second, observed_second) ||
      !prepared->read(*windows_third, observed_third)) {
    return 4;
  }
  return observed_final[0] == 58u &&
                 observed_first ==
                     std::array<std::uint32_t, maximum>{3u,  5u,  7u,
                                                        11u, 13u, sentinel} &&
                 observed_second ==
                     std::array<std::uint32_t, maximum>{4u,  6u,  8u,
                                                        12u, 14u, sentinel} &&
                 observed_third ==
                     std::array<std::uint32_t, maximum>{5u,  7u,  9u,
                                                        13u, 15u, sentinel} &&
                 prepared->stats().pipeline.executed_outer_window_count == 3u &&
                 prepared->stats().pipeline.executed_inner_iteration_count == 0u
             ? 0
             : 5;
}

} // namespace rund::package_example::pipeline
