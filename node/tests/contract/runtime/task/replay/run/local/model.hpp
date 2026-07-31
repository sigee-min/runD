#pragma once

#include "../../local.hpp"

#include <rund/replay.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace runtime_task_replay_run {

inline constexpr std::uint64_t kStateSchema = 0x7101u;
inline constexpr rund::replay::Input kInput{.id = 7u, .schema = 101u};
inline constexpr std::uint64_t kSequence = 0u;

void Append(rund::replay::Writer &writer, std::span<const std::byte> bytes);

struct Model final {
  explicit Model(rund::SessionConfig initial);

  rund::SessionConfig config;
  std::array<std::byte, 1u> first_state{std::byte{0x61}};
  std::size_t restore_calls = 0u;

  struct Restore final {
    Model &model;
    rund::replay::Restore operator()(std::span<const std::byte> state) const;
  } restore;

  rund::replay::Binding binding;
  rund::Session session;
  std::array<std::byte, 3u> payload{std::byte{0x10}, std::byte{0x20},
                                    std::byte{0x30}};
  std::size_t producer_calls = 0u;
  std::size_t callback_calls = 0u;
  std::size_t observed_size = 0u;

  struct Source final {
    Model &model;
    std::uint64_t operator()(rund::replay::Writer &writer) const;
  } source;

  rund::replay::Channel commands;

  struct Step final {
    Model &model;
    void operator()(rund::replay::Context &input, rund::Session &active) const;
  } step;

  std::optional<rund::replay::Record> baseline;
};

int Capacity(rund::SessionConfig config);
int Surface(Model &model);
int Scenario(Model &model);
int Lifetime(Model &model);
int History(Model &model);

} // namespace runtime_task_replay_run
