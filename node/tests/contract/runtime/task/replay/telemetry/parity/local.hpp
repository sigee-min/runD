#pragma once

#include <rund/replay.hpp>
#include <rund/session.hpp>
#include <rund/telemetry/event.hpp>
#include <rund/telemetry/sink.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace replay_telemetry_contract {

struct Collector final {
  std::array<rund::telemetry::Event, 16u> values{};
  std::size_t count = 0u;

  void operator()(const rund::telemetry::Event &event);

  [[nodiscard]] const rund::telemetry::Event &
  one(std::size_t before) const noexcept;
};

[[nodiscard]] std::uint64_t SessionId() noexcept;

[[nodiscard]] rund::SessionConfig Config(rund::telemetry::Sink sink,
                                         std::uint64_t id);

void ExpectBasic(const rund::telemetry::Event &event,
                 rund::telemetry::Level level);

struct Identity final {
  rund::replay::Code code = rund::replay::Code::CheckpointInvalid;
  std::uint64_t input = 0u;
  std::uint64_t transcript = 0u;
  std::uint64_t result = 0u;

  [[nodiscard]] friend bool operator==(const Identity &,
                                       const Identity &) = default;
};

struct LevelRun final {
  std::array<rund::telemetry::Event, 6u> events{};
  std::array<Identity, 4u> identities{};
  std::uint64_t producer_calls = 0u;
};

[[nodiscard]] LevelRun RunLevel(rund::telemetry::Level level);
[[nodiscard]] int CheckParity();
[[nodiscard]] int CheckFailures();

} // namespace replay_telemetry_contract
