#pragma once

#include "model.hpp"

namespace rund::measure::compute::preparation_memory {

template <std::size_t Max, std::size_t Width>
[[nodiscard]] inline auto LargeSeedProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(Max)
      .zip_input<std::uint32_t>(Domain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Max, Width>(total, ordinal);
        auto active = queue.gather(current.items());
        auto values = domain.gather(active);
        for (std::size_t pass = 0u; pass < SeedScanMapPairs; ++pass) {
          values = values.scan(pass % 2u == 0u ? Scan::InclusiveSum
                                               : Scan::ExclusiveSum);
          values = values.map("measure-prepare-memory-seed-map",
                              [](auto value) { return value + 1u; });
        }
        return outputs(values.reduce(Reduce::Sum), values);
      })
      .compile();
}

[[nodiscard]] inline auto ActionProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(Tile)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto window, auto window_count) {
        auto next = value.map("measure-prepare-memory-action",
                              [](auto item) { return item + 1u; });
        auto retained = window.map("measure-prepare-memory-window-retain",
                                   [](auto item) { return item; });
        auto retained_count =
            window_count.map("measure-prepare-memory-window-count",
                             [](auto item) { return item; });
        return outputs(next, retained, retained_count);
      })
      .compile();
}

[[nodiscard]] inline auto FoldProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(Tile)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto window, auto window_count) {
        (void)window_count;
        auto next =
            outer.combine("measure-prepare-memory-fold", tile,
                          [](auto left, auto right) { return left + right; });
        auto published = window.map("measure-prepare-memory-window-publish",
                                    [](auto item) { return item; });
        return outputs(next, published);
      })
      .compile();
}

template <std::size_t Max, std::size_t Width>
[[nodiscard]] inline auto SecondSeedProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(Max)
      .zip_input<std::uint32_t>(Domain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Max, Width>(total, ordinal);
        auto active = queue.gather(current.items());
        auto values = domain.gather(active);
        for (std::size_t pass = 0u; pass < SeedScanMapPairs; ++pass) {
          values = values.scan(pass % 2u == 0u ? Scan::InclusiveSum
                                               : Scan::ExclusiveSum);
          values = values.map("measure-prepare-memory-second-seed-map",
                              [](auto value) { return value + 1u; });
        }
        return values.reduce(Reduce::Sum);
      })
      .compile();
}

[[nodiscard]] inline auto SecondActionProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-prepare-memory-second-action", 1u,
                          [](auto value) { return value + 1u; })
      .compile();
}

[[nodiscard]] inline auto SecondFoldProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile) {
        return outer.combine(
            "measure-prepare-memory-second-fold", tile,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

[[nodiscard]] inline auto ConsumeProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-prepare-memory-consume", Maximum,
                          [](auto value) { return value; })
      .compile();
}

[[nodiscard]] inline auto OrdinaryRecurrenceProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-prepare-memory-recurrence", Maximum,
                          [](auto value) { return value + 1u; })
      .compile();
}

[[nodiscard]] inline auto PublishProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto published) {
        return value.combine("measure-prepare-memory-publish", published,
                             [](auto next, auto) { return next; });
      })
      .compile();
}


} // namespace rund::measure::compute::preparation_memory
