#pragma once

#include <cstdint>
#include <vector>

namespace rund::node {

struct TimerWait;
struct TimerWaitIdIndexEntry;

[[nodiscard]] bool ReactorTimeoutReserveTimerStorage(
    std::vector<TimerWait>& timers,
    std::vector<TimerWaitIdIndexEntry>& index) noexcept;
[[nodiscard]] bool ReactorTimeoutCancelTimer(
    std::vector<TimerWait>& timers,
    std::vector<TimerWaitIdIndexEntry>& index,
    std::uint64_t wait_id) noexcept;

}  // namespace rund::node
