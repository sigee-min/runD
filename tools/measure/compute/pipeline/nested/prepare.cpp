#include "model.hpp"

#include <array>
#include <limits>
#include <numeric>

namespace rund::measure::compute::nested_repeat {
namespace {

template <std::size_t Max, std::size_t Width>
[[nodiscard]] auto SeedProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(Max)
      .zip_input<std::uint32_t>(Domain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Max, Width>(total, ordinal);
        auto active_ordinals = queue.gather(current.items());
        return domain.gather(active_ordinals).reduce(Reduce::Sum);
      })
      .compile();
}

[[nodiscard]] auto ActionProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-window-repeat-action", 1u,
                          [](auto value) { return value + 1u; })
      .compile();
}

[[nodiscard]] auto FoldProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile) {
        return outer.combine(
            "measure-window-repeat-fold", tile,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

} // namespace

std::optional<Fixture> Prepare(const ::rund::compute::Backend backend) {
  using namespace ::rund::compute;
  std::vector<std::uint32_t> queue_values(Maximum);
  for (std::size_t index = 0u; index < queue_values.size(); ++index) {
    queue_values[index] = static_cast<std::uint32_t>(index % Domain);
  }
  std::array<std::uint32_t, Domain> domain_values{};
  for (std::size_t index = 0u; index < domain_values.size(); ++index) {
    domain_values[index] = static_cast<std::uint32_t>(3u * index + 1u);
  }
  std::vector<std::uint32_t> tile_seeds(Outer);
  for (std::size_t outer = 0u; outer < Outer; ++outer) {
    std::uint32_t sum = 0u;
    for (std::size_t offset = 0u; offset < Tile; ++offset) {
      const std::size_t index = outer * Tile + offset;
      sum += domain_values[queue_values[index]];
    }
    tile_seeds[outer] = sum;
  }
  const std::uint64_t expected_wide =
      OuterSeed +
      std::accumulate(tile_seeds.begin(), tile_seeds.end(), std::uint64_t{}) +
      SerialSubmits;
  if (expected_wide > std::numeric_limits<std::uint32_t>::max()) {
    std::fprintf(stderr, "window repeat oracle overflow\n");
    return std::nullopt;
  }

  auto device = open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "window repeat %s open failed: %.*s\n", Name(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return std::nullopt;
  }
  auto seed_program = SeedProgram<Maximum, Tile>(*device);
  auto action_program = ActionProgram(*device);
  auto fold_program = FoldProgram(*device);
  auto observe_program =
      on(*device)
          .map<std::uint32_t>("measure-window-repeat-observe", Outer,
                              [](auto value) { return value; })
          .compile();
  if (!seed_program || !action_program || !fold_program || !observe_program) {
    std::fprintf(stderr, "window repeat %s program preparation failed\n",
                 Name(backend));
    return std::nullopt;
  }

  auto serial_seed = device->upload<std::uint32_t>(tile_seeds);
  auto serial_first = device->buffer<std::uint32_t>(Outer);
  auto serial_second = device->buffer<std::uint32_t>(Outer);
  auto serial_observed = device->buffer<std::uint32_t>(Outer);
  const std::array<std::uint32_t, 1u> count_value{
      static_cast<std::uint32_t>(Maximum)};
  const std::array<std::uint32_t, 1u> outer_value{OuterSeed};
  const std::array<std::uint32_t, 1u> output_sentinel{0xA5A55A5Au};
  auto queue = device->upload<std::uint32_t>(queue_values);
  auto domain = device->upload<std::uint32_t>(domain_values);
  auto count = device->upload<std::uint32_t>(count_value);
  auto outer_seed = device->upload<std::uint32_t>(outer_value);
  auto nested_output = device->upload<std::uint32_t>(output_sentinel);
  auto repeated_output = device->upload<std::uint32_t>(output_sentinel);
  auto sealed_output = device->upload<std::uint32_t>(output_sentinel);
  if (!serial_seed || !serial_first || !serial_second || !serial_observed ||
      !queue || !domain || !count || !outer_seed || !nested_output ||
      !repeated_output || !sealed_output) {
    std::fprintf(stderr, "window repeat %s buffer preparation failed\n",
                 Name(backend));
    return std::nullopt;
  }

  std::vector<Pipeline> serial_steps;
  serial_steps.reserve(Outer * 2u);
  for (std::size_t outer = 0u; outer < Outer; ++outer) {
    auto first = serial_first->view(outer, 1u);
    auto second = serial_second->view(outer, 1u);
    if (!first || !second) {
      std::fprintf(stderr, "window repeat %s serial view failed\n",
                   Name(backend));
      return std::nullopt;
    }
    auto forward = pipeline(*device)
                       .then(*action_program, read(*first), write(*second))
                       .prepare();
    auto backward = pipeline(*device)
                        .then(*action_program, read(*second), write(*first))
                        .prepare();
    if (!forward || !backward) {
      std::fprintf(stderr,
                   "window repeat %s serial pipeline preparation failed "
                   "outer=%zu\n",
                   Name(backend), outer);
      return std::nullopt;
    }
    serial_steps.emplace_back(std::move(*forward));
    serial_steps.emplace_back(std::move(*backward));
  }

  const auto body =
      tile_repeat<Inner>(*seed_program, *action_program, *fold_program);
  auto nested_builder = pipeline(*device);
  nested_builder.windows<Maximum, Tile>(body, window(*count),
                                        read(*outer_seed, *queue, *domain),
                                        write_final(*nested_output));
  auto nested_plan = nested_builder.plan();
  auto nested = std::move(nested_builder).prepare();
  auto repeated_builder = pipeline(*device);
  repeated_builder.windows<Maximum, Tile>(body, window(*count),
                                          read(*outer_seed, *queue, *domain),
                                          write_final(*repeated_output));
  auto repeated_plan = repeated_builder.plan();
  auto repeated = std::move(repeated_builder).prepare();
  auto sealed_builder = pipeline(*device);
  sealed_builder.sealed_repetitions<Repetitions>().windows<Maximum, Tile>(
      body, window(*count), read(*outer_seed, *queue, *domain),
      write_final(*sealed_output));
  auto sealed_plan = sealed_builder.plan();
  auto sealed = std::move(sealed_builder).prepare();
  if (!nested_plan || !nested ||
      nested_plan->prepared_template_count != Templates ||
      nested_plan->prepared_command_count != Commands || !repeated_plan ||
      !repeated || *repeated_plan != *nested_plan || !sealed_plan || !sealed ||
      *sealed_plan != *nested_plan) {
    std::fprintf(stderr, "window repeat %s nested preparation failed\n",
                 Name(backend));
    return std::nullopt;
  }

  return Fixture{
      backend,
      static_cast<std::uint32_t>(expected_wide),
      std::move(*device),
      std::move(*observe_program),
      std::move(*serial_seed),
      std::move(*serial_first),
      std::move(*serial_second),
      std::move(*serial_observed),
      std::move(*nested_output),
      std::move(*repeated_output),
      std::move(*sealed_output),
      std::move(serial_steps),
      std::move(*nested_plan),
      std::move(*nested),
      std::move(*repeated_plan),
      std::move(*repeated),
      std::move(*sealed_plan),
      std::move(*sealed),
  };
}

} // namespace rund::measure::compute::nested_repeat
