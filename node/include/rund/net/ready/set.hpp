#pragma once

#include <rund/net/ready/many.hpp>
#include <rund/net/ready/set/identity.hpp>

#include <chrono>
#include <cstdint>
#include <span>

namespace rund::net::ready {

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
