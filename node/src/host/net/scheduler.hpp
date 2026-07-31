#pragma once

#include <rund/host/event.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/results.hpp>

#include <chrono>

namespace rund::node {
class Scheduler;
}

namespace rund::net {

[[nodiscard]] bool InActiveSchedulerTask() noexcept;
[[nodiscard]] bool RecordHostEvent(::rund::host::Event event) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision
WaitReactor(SocketView socket, short interest) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision
WaitReactor(node::Scheduler &scheduler, SocketView socket,
            short interest) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision WaitReactorTimed(
    SocketView socket, short interest, std::chrono::nanoseconds timeout,
    std::uint64_t stop_scheduler_id = 0u, std::uint64_t stop_source_id = 0u,
    std::uint64_t stop_generation = 0u, std::uint64_t stop_epoch = 0u) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision WaitReactorTimed(
    node::Scheduler &scheduler, SocketView socket, short interest,
    std::chrono::nanoseconds timeout, std::uint64_t stop_scheduler_id = 0u,
    std::uint64_t stop_source_id = 0u, std::uint64_t stop_generation = 0u,
    std::uint64_t stop_epoch = 0u) noexcept;

} // namespace rund::net
