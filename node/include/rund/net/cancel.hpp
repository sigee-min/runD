#pragma once

#include <rund/net/ready/many.hpp>
#include <rund/net/ready/timed.hpp>
#include <rund/task/cancel.hpp>

#include <chrono>
#include <span>

namespace rund::net::ready::timed {

[[nodiscard]] Wait read(SocketView socket, std::chrono::nanoseconds timeout,
                        task::stop_token token) noexcept;
[[nodiscard]] Wait write(SocketView socket, std::chrono::nanoseconds timeout,
                         task::stop_token token) noexcept;

} // namespace rund::net::ready::timed

namespace rund::net::ready::many {

[[nodiscard]] Wait wait(std::span<const Request> requests,
                        std::span<Event> out,
                        std::chrono::nanoseconds timeout, task::stop_token token,
                        Budget budget = {}) noexcept;

} // namespace rund::net::ready::many
