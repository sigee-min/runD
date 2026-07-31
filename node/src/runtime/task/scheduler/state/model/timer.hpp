#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace rund::node {

enum class TimerWaitKind : std::uint8_t {
  Sleep,
  ReactorTimeout,
  ReactorManyTimeout,
};

using Clock = std::chrono::steady_clock;

struct TimerDeadline {
  Clock::time_point deadline{};
  std::int64_t deadline_ns = 0;
};

struct TimerWait {
  TimerWaitKind kind = TimerWaitKind::Sleep;
  std::uint64_t task_id = 0u;
  std::uint64_t wait_id = 0u;
  std::uint64_t stop_source_id = 0u;
  std::uint64_t stop_generation = 0u;
  std::uint64_t stop_epoch = 0u;
  Clock::time_point deadline{};
  std::int64_t deadline_ns = 0;
  std::uint64_t sequence = 0u;
  std::uint64_t host_handle_id = 0u;
  int fd = -1;
  short interest = 0;
};

struct TimerWaitIdIndexEntry {
  std::uint64_t wait_id = 0u;
  std::size_t slot = 0u;
};

} // namespace rund::node
