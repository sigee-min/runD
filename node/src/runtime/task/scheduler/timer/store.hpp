#pragma once

#include "../state/model/timer.hpp"
#include "../state/storage.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

namespace rund::node {

[[nodiscard]] bool TimerStorePush(
    std::vector<TimerWait>& timers,
    std::vector<TimerWaitIdIndexEntry>& index,
    TimerWait wait) noexcept;

[[nodiscard]] bool TimerStoreCancel(
    std::vector<TimerWait>& timers,
    std::vector<TimerWaitIdIndexEntry>& index,
    std::uint64_t wait_id,
    TimerWait* removed = nullptr) noexcept;

[[nodiscard]] bool TimerStoreContains(
    const std::vector<TimerWait>& timers,
    const std::vector<TimerWaitIdIndexEntry>& index,
    std::uint64_t wait_id) noexcept;

[[nodiscard]] const TimerWait* TimerStoreFind(
    const std::vector<TimerWait>& timers,
    const std::vector<TimerWaitIdIndexEntry>& index,
    std::uint64_t wait_id) noexcept;

[[nodiscard]] bool TimerStorePopDue(
    std::vector<TimerWait>& timers,
    std::vector<TimerWaitIdIndexEntry>& index,
    Clock::time_point now,
    TimerWait* out) noexcept;

[[nodiscard]] int TimerStorePollTimeoutMs(
    const std::vector<TimerWait>& timers,
    Clock::time_point now) noexcept;

}  // namespace rund::node
