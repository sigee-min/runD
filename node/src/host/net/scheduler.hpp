#pragma once

#include <rund/host/event.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/cancel/identity.hpp>
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
    ::rund::detail::task::StopIdentity stop = {}) noexcept;
[[nodiscard]] ::rund::detail::task::IoDecision WaitReactorTimed(
    node::Scheduler &scheduler, SocketView socket, short interest,
    std::chrono::nanoseconds timeout,
    ::rund::detail::task::StopIdentity stop = {}) noexcept;

} // namespace rund::net
