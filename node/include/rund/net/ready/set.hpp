#pragma once

#include <rund/net/ready/many.hpp>

#include <chrono>
#include <cstdint>
#include <span>
#include <type_traits>

namespace rund::net::ready {

struct Set {
  std::uint64_t id = 0u;
  std::uint64_t generation = 0u;
};

static_assert(sizeof(Set) == 16u);
static_assert(std::is_standard_layout_v<Set>);
static_assert(std::is_trivially_copyable_v<Set>);

struct Config {
  std::uint32_t max_members = 1024u;
};

struct Status : net::Status {
  using net::Status::Status;

  Set set{};
};

[[nodiscard]] Status create(Config config = {}) noexcept;
[[nodiscard]] Status destroy(Set set) noexcept;
[[nodiscard]] Status clear(Set set) noexcept;
[[nodiscard]] Status add(Set set, Request request) noexcept;
[[nodiscard]] Status remove(Set set, Request request) noexcept;

namespace many {

[[nodiscard]] Wait wait(Set set, std::span<Event> out,
                        Budget budget = {}) noexcept;
[[nodiscard]] Wait wait(Set set, std::span<Event> out,
                        std::chrono::nanoseconds timeout,
                        Budget budget = {}) noexcept;

} // namespace many
} // namespace rund::net::ready
