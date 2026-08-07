#pragma once

#include "../local.hpp"

namespace rund::node {

[[nodiscard]] ReactorCleanupRequest
ReadyManyParkRollbackRequest(std::uint64_t group_id,
                             ReasonCode reason) noexcept;

[[nodiscard]] bool ReadyManyParkPublishGroup(
    SchedulerState &state, ReadyManyEntry &entry, std::uint64_t group_id,
    std::uint64_t timer_wait_id,
    ::rund::net::ready::Set ready_set) noexcept;

} // namespace rund::node
