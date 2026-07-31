#pragma once

#include "../local.hpp"

namespace rund::node {

[[nodiscard]] bool ReadyManyParkCreateGroupAndRequests(
    SchedulerState &state, ReadyManyEntry &entry, std::uint64_t group_id,
    std::uint64_t timer_wait_id, std::uint64_t stop_source_id,
    std::uint64_t stop_generation, std::uint64_t stop_epoch,
    std::uint64_t ready_set_id, std::uint64_t ready_set_generation) noexcept;

} // namespace rund::node
