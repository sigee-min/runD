#include "local.hpp"

#include <array>
#include <cstdio>
#include <span>

namespace rund::node::test_contract::window {

[[nodiscard]] int CheckWindowTerminal(Device &device, const Backend backend,
                                      MatrixIdentity &identity) {
  using namespace rund::compute;
  constexpr std::size_t maximum = 8u;
  constexpr std::size_t tile = 3u;
  constexpr std::array<std::uint32_t, 1u> state_seed{7u};
  constexpr std::array<std::uint32_t, 1u> terminal_seed{0u};
  constexpr std::array<std::uint32_t, 1u> count_seed{maximum};

  auto body = on(device)
                  .input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .zip_input<std::uint32_t>(1u)
                  .branch([](auto state, auto terminal, auto count,
                             auto ordinal) {
                    (void)count;
                    (void)ordinal;
                    auto next = state.map("window-terminal-state",
                                          [](auto value) {
                                            return value + 1u;
                                          });
                    auto stop = terminal.map("window-terminal-stop",
                                             [](auto value) {
                                               return value * 0u + 7u;
                                             });
                    return outputs(next, stop);
                  })
                  .compile();
  auto initial =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{state_seed});
  auto terminal = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{terminal_seed});
  auto count =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{count_seed});
  auto output = device.buffer<std::uint32_t>(1u);
  auto stopped = device.buffer<std::uint32_t>(1u);
  if (!body || !initial || !terminal || !count || !output || !stopped) {
    return 1;
  }

  auto prepared =
      pipeline(device)
          .windows<maximum, tile>(
              *body, rund::compute::window(*count).until<1u>(7u),
              read(*initial, *terminal), write(*output, *stopped))
          .prepare();
  if (!prepared) {
    return 2;
  }
  std::array<std::uint32_t, 1u> actual{};
  std::array<std::uint32_t, 1u> terminal_actual{};
  const Status ran = prepared->run();
  if (!ran || !prepared->read(*output, actual) ||
      !prepared->read(*stopped, terminal_actual)) {
    return 3;
  }
  if (actual[0] != state_seed[0] + 1u || terminal_actual[0] != 7u) {
    std::fprintf(stderr,
                 "window terminal backend=%u state=%u terminal=%u "
                 "expected_state=%u\n",
                 static_cast<unsigned>(backend), actual[0], terminal_actual[0],
                 state_seed[0] + 1u);
    return 4;
  }
  const std::array<std::uint32_t, 2u> raw{actual[0], terminal_actual[0]};
  const Fingerprint body_hash = body->fingerprint();
  const Fingerprint pipeline_hash = prepared->fingerprint();
  const std::uint64_t output_hash = Hash(raw.data(), sizeof(raw));
  if (!body_hash || !pipeline_hash || output_hash == 0u) {
    return 5;
  }
  if (identity.body) {
    if (identity.body != body_hash || identity.pipeline != pipeline_hash ||
        identity.output != output_hash) {
      return 6;
    }
  } else {
    identity.body = body_hash;
    identity.pipeline = pipeline_hash;
    identity.output = output_hash;
  }
  return 0;
}

} // namespace rund::node::test_contract::window
