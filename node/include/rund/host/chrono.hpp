#pragma once

#include <chrono>
#include <cstdint>
#include <limits>

namespace rund::host::chrono {

using nanoseconds = std::chrono::nanoseconds;

struct time_point {
  std::int64_t ns = 0;

  [[nodiscard]] friend constexpr bool operator==(time_point lhs,
                                                 time_point rhs) noexcept {
    return lhs.ns == rhs.ns;
  }
  [[nodiscard]] friend constexpr bool operator<(time_point lhs,
                                                time_point rhs) noexcept {
    return lhs.ns < rhs.ns;
  }
};

struct logical_clock {
  using duration = nanoseconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = rund::host::chrono::time_point;
  static constexpr bool is_steady = true;

  [[nodiscard]] static rund::host::chrono::time_point now() noexcept;
};

[[nodiscard]] constexpr time_point
operator+(const time_point point, const nanoseconds duration) noexcept {
  const std::int64_t delta = duration.count();
  if (delta > 0 &&
      point.ns > std::numeric_limits<std::int64_t>::max() - delta) {
    return time_point{.ns = std::numeric_limits<std::int64_t>::max()};
  }
  if (delta < 0 &&
      point.ns < std::numeric_limits<std::int64_t>::min() - delta) {
    return time_point{.ns = std::numeric_limits<std::int64_t>::min()};
  }
  return time_point{.ns = point.ns + delta};
}

[[nodiscard]] constexpr nanoseconds operator-(const time_point lhs,
                                              const time_point rhs) noexcept {
  if (rhs.ns < 0 &&
      lhs.ns > std::numeric_limits<std::int64_t>::max() + rhs.ns) {
    return nanoseconds{std::numeric_limits<std::int64_t>::max()};
  }
  if (rhs.ns > 0 &&
      lhs.ns < std::numeric_limits<std::int64_t>::min() + rhs.ns) {
    return nanoseconds{std::numeric_limits<std::int64_t>::min()};
  }
  return nanoseconds{lhs.ns - rhs.ns};
}

} // namespace rund::host::chrono
