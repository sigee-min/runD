#pragma once

#include "../model.hpp"

#include <optional>

namespace package_blackbox::replay_detail {

struct Fixture;

struct Restore final {
  Fixture *fixture = nullptr;

  [[nodiscard]] rund::replay::Restore
  operator()(std::span<const std::byte> bytes) const;
};

struct Source final {
  Fixture *fixture = nullptr;

  [[nodiscard]] std::uint64_t operator()(rund::replay::Writer &writer) const;
};

struct Fixture final {
  inline static constexpr rund::replay::Input input{.id = 17u, .schema = 401u};
  inline static constexpr std::uint64_t sequence = 29u;
  inline static constexpr std::uint64_t state_schema = 701u;
  inline static constexpr std::uint64_t continued_sequence = 30u;
  inline static constexpr std::byte continued_expected{0x2d};

  explicit Fixture(rund::Session &session);

  void ContinueSimulation(rund::replay::Context &context);

  rund::Session &session;
  std::vector<std::byte> state;
  bool restored_before_record = false;
  bool continuation_record_ran = false;
  bool continued_value_ok = false;
  std::uint64_t source_sequence = sequence;
  std::byte source_value{0x2a};
  std::uint32_t producers = 0u;
  Restore restore;
  rund::replay::Binding replay_binding;
  Source source;
  rund::replay::Channel commands;
  std::optional<rund::replay::Record> record{};
  std::optional<rund::replay::Record> continued{};
  std::optional<rund::replay::Checkpoint> checkpoint{};
};

[[nodiscard]] int CheckRecord(Fixture &fixture);
[[nodiscard]] int CheckCodec(Fixture &fixture);
[[nodiscard]] int CheckScenario(Fixture &fixture);
[[nodiscard]] int CheckCheckpoint(Fixture &fixture);
[[nodiscard]] int CheckHistory(Fixture &fixture);

} // namespace package_blackbox::replay_detail
