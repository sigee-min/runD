#pragma once

#include "../local.hpp"

namespace rund::node {

[[nodiscard]] bool ReadyManyParkCreateGroupAndRequests(
    SchedulerState &state, ReadyManyEntry &entry, std::uint64_t group_id,
    std::uint64_t timer_wait_id,
    ::rund::net::ready::Set ready_set) noexcept;

} // namespace rund::node
