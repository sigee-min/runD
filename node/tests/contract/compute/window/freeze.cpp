#include "local.hpp"

#include <array>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {

[[nodiscard]] int CheckWindowFreeze(Device &device, const Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t maximum = 10u;
  constexpr std::size_t tile = 3u;
  constexpr std::array<std::uint32_t, 1u> seed{7u};
  // Exercise all selector sources at the first stop: seed, first, and second.
  constexpr std::array<std::uint32_t, 3u> counts{0u, 1u, 4u};

  auto body = on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto state, auto count, auto ordinal) {
                    (void)count;
                    (void)ordinal;
                    return state.map("resident-window-freeze-witness",
                                     [](auto value) { return value + 1u; });
                  })
                  .compile();
  auto consume =
      on(device)
          .input<std::uint32_t>(1u)
          .branch([](auto state) {
            return state.map("resident-window-freeze-consume",
                             [](auto value) { return value + 100u; });
          })
          .compile();
  auto initial =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{seed});
  if (!body || !consume || !initial) {
    return 1;
  }
  for (const std::uint32_t item_count : counts) {
    const std::array<std::uint32_t, 1u> count_value{item_count};
    auto count = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{count_value});
    auto output = device.buffer<std::uint32_t>(1u);
    auto consumed = device.buffer<std::uint32_t>(1u);
    if (!count || !output || !consumed) {
      return 2;
    }
    auto builder = pipeline(device);
    builder
        .windows<maximum, tile>(*body, rund::compute::window(*count),
                                read(*initial), write(*output))
        .then(*consume, read(*output), write(*consumed));
    auto prepared = std::move(builder).prepare();
    if (!prepared) {
      return 3;
    }
    std::array<std::uint32_t, 1u> actual{};
    std::array<std::uint32_t, 1u> consumed_actual{};
    const Status ran = prepared->run();
    if (!ran || !prepared->read(*output, actual) ||
        !prepared->read(*consumed, consumed_actual)) {
      return 4;
    }
    const std::uint32_t iterations = (item_count + tile - 1u) / tile;
    const std::uint32_t expected = seed[0] + iterations;
    if (actual[0] != expected || consumed_actual[0] != expected + 100u) {
      std::fprintf(stderr,
                   "window freeze backend=%u count=%u actual=%u consumed=%u "
                   "expected=%u\n",
                   static_cast<unsigned>(backend), item_count, actual[0],
                   consumed_actual[0], expected);
      return 5;
    }
  }
  return 0;
}

} // namespace rund::node::test_contract::window
