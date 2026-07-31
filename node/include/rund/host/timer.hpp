#pragma once

#include <rund/task/results.hpp>

namespace rund::host::timer {

[[nodiscard]] inline task::SleepOp at(chrono::time_point deadline) noexcept {
  const chrono::time_point now = chrono::logical_clock::now();
  const chrono::nanoseconds duration = deadline - now;
  return task::sleep(duration.count() > 0 ? duration : chrono::nanoseconds{0});
}

} // namespace rund::host::timer
