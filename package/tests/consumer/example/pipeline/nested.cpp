#include "local.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckNestedResidentRecurrence(rund::compute::Device &device) {
  constexpr std::size_t maximum = 5u;
  constexpr std::size_t tile = 2u;
  constexpr std::size_t inner = 3u;
  constexpr std::size_t outer_windows = (maximum + tile - 1u) / tile;
  constexpr std::size_t prepared_templates = outer_windows + 2u + 3u;
  constexpr std::size_t prepared_commands = outer_windows * (inner + 2u);
  constexpr std::array<std::uint32_t, 1u> outer_seed{10u};
  constexpr std::array<std::uint32_t, 1u> count_value{maximum};

  auto seed = rund::compute::on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto count, auto ordinal) {
                    (void)count;
                    return ordinal.map("pipeline-installed-nested-seed",
                                       [](auto value) { return value + 1u; });
                  })
                  .compile();
  auto action = rund::compute::on(device)
                    .map<std::uint32_t>("pipeline-installed-nested-action", 1u,
                                        [](auto value) { return value + 1u; })
                    .compile();
  auto fold = rund::compute::on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto outer, auto local) {
                    return outer.combine(
                        "pipeline-installed-nested-fold", local,
                        [](auto left, auto right) { return left + right; });
                  })
                  .compile();
  auto outer = device.upload<std::uint32_t>(outer_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto output = device.buffer<std::uint32_t>(1u);
  if (!seed || !action || !fold || !outer || !count || !output) {
    return 1;
  }

  const auto body = rund::compute::tile_repeat<inner>(*seed, *action, *fold);
  auto builder = rund::compute::pipeline(device);
  builder.windows<maximum, tile>(body, rund::compute::window(*count),
                                 rund::compute::read(*outer),
                                 rund::compute::write_final(*output));
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != outer_windows ||
      plan->tile_capacity != tile || plan->inner_iteration_count != inner ||
      plan->prepared_template_count != prepared_templates ||
      plan->prepared_command_count != prepared_commands) {
    return 2;
  }

  auto prepared = std::move(builder).prepare();
  if (!prepared || !prepared->run()) {
    return 3;
  }
  std::array<std::uint32_t, 1u> actual{};
  const auto read = prepared->read(*output, actual);
  return read && actual[0] == 25u &&
                 prepared->stats().pipeline.executed_outer_window_count ==
                     outer_windows &&
                 prepared->stats().pipeline.executed_inner_iteration_count ==
                     outer_windows * inner
             ? 0
             : 4;
}

} // namespace rund::package_example::pipeline
