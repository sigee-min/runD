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

  auto body =
      on(device)
          .input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .zip_input<std::uint32_t>(1u)
          .branch([](auto state, auto terminal, auto count, auto ordinal) {
            (void)count;
            (void)ordinal;
            auto next = state.map("window-terminal-state",
                                  [](auto value) { return value + 1u; });
            auto stop = terminal.map("window-terminal-stop", [](auto value) {
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
              read(*initial, *terminal), write_final(*output, *stopped))
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

  const auto check_seed_target_alias = [&](const bool early_stop) {
    auto aliased = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{state_seed});
    auto aliased_terminal = device.buffer<std::uint32_t>(1u);
    if (!aliased || !aliased_terminal) {
      return false;
    }
    auto builder = pipeline(device);
    if (early_stop) {
      builder.windows<maximum, tile>(
          *body, rund::compute::window(*count).until<1u>(7u),
          read(*aliased, *terminal), write_final(*aliased, *aliased_terminal));
    } else {
      builder.windows<maximum, tile>(*body, rund::compute::window(*count),
                                     read(*aliased, *terminal),
                                     write_final(*aliased, *aliased_terminal));
    }
    auto alias_pipeline = std::move(builder).prepare();
    std::array<std::uint32_t, 1u> alias_actual{};
    std::array<std::uint32_t, 1u> alias_terminal_actual{};
    const std::uint32_t expected_state = state_seed[0] + (early_stop ? 1u : 3u);
    return alias_pipeline && alias_pipeline->run() &&
           alias_pipeline->read(*aliased, alias_actual) &&
           alias_pipeline->read(*aliased_terminal, alias_terminal_actual) &&
           alias_actual[0] == expected_state && alias_terminal_actual[0] == 7u;
  };
  // Only the selected final bank must be distinct from the publication
  // target. Reusing the seed Buffer as the target is valid for both an early
  // stop and full completion because that seed bank is no longer selected.
  if (!check_seed_target_alias(true) || !check_seed_target_alias(false)) {
    return 5;
  }
  auto foreign_device = open(Target::cpu(2u));
  auto foreign_output = foreign_device
                            ? foreign_device->buffer<std::uint32_t>(1u)
                            : decltype(foreign_device->buffer<std::uint32_t>(
                                  1u))::fail(foreign_device.reason());
  if (!foreign_output) {
    return 6;
  }
  auto foreign_builder = pipeline(device);
  foreign_builder.windows<maximum, tile>(
      *body, rund::compute::window(*count).until<1u>(7u),
      read(*initial, *terminal), write_final(*foreign_output, *stopped));
  const auto foreign_plan = foreign_builder.plan();
  auto foreign_prepared = std::move(foreign_builder).prepare();
  if (!foreign_plan || foreign_prepared ||
      foreign_prepared.reason() != Reason::BindingDeviceMismatch) {
    return 7;
  }

  auto prior_terminal = device.buffer<std::uint32_t>(1u);
  auto second_terminal = device.buffer<std::uint32_t>(1u);
  if (!prior_terminal || !second_terminal) {
    return 8;
  }
  auto prior_output_builder = pipeline(device);
  prior_output_builder
      .then(*body, read(*initial, *terminal, *count, *count),
            write(*output, *prior_terminal))
      .windows<maximum, tile>(
          *body, rund::compute::window(*count).until<1u>(7u),
          read(*initial, *terminal), write_final(*output, *stopped));
  const auto prior_output_plan = prior_output_builder.plan();
  auto prior_output_prepared = std::move(prior_output_builder).prepare();
  if (!prior_output_plan || prior_output_prepared ||
      prior_output_prepared.reason() != Reason::BindingDuplicate) {
    return 9;
  }

  auto duplicate_publication_builder = pipeline(device);
  duplicate_publication_builder
      .windows<maximum, tile>(
          *body, rund::compute::window(*count).until<1u>(7u),
          read(*initial, *terminal), write_final(*output, *prior_terminal))
      .windows<maximum, tile>(
          *body, rund::compute::window(*count).until<1u>(7u),
          read(*initial, *terminal), write_final(*output, *second_terminal));
  const auto duplicate_publication_plan = duplicate_publication_builder.plan();
  auto duplicate_publication_prepared =
      std::move(duplicate_publication_builder).prepare();
  if (!duplicate_publication_plan || duplicate_publication_prepared ||
      duplicate_publication_prepared.reason() != Reason::BindingDuplicate) {
    return 10;
  }

  const std::array<std::uint32_t, 2u> raw{actual[0], terminal_actual[0]};
  const Fingerprint body_hash = body->fingerprint();
  const Fingerprint pipeline_hash = prepared->fingerprint();
  const std::uint64_t output_hash = Hash(raw.data(), sizeof(raw));
  if (!body_hash || !pipeline_hash || output_hash == 0u) {
    return 11;
  }
  if (identity.body) {
    if (identity.body != body_hash || identity.pipeline != pipeline_hash ||
        identity.output != output_hash) {
      return 12;
    }
  } else {
    identity.body = body_hash;
    identity.pipeline = pipeline_hash;
    identity.output = output_hash;
  }
  return 0;
}

} // namespace rund::node::test_contract::window
