#include "timeline.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::compute::detail::graph_reduce {
namespace {

using ::rund::detail::counter::Accumulate;

[[nodiscard]] std::uint64_t
intersection(const Interval interval,
             const residency::ExecutionReceipt &compute) noexcept {
  const std::uint64_t first = std::max(interval.started, compute.started_ns);
  const std::uint64_t last = std::min(interval.completed, compute.completed_ns);
  return last > first ? last - first : 0u;
}

} // namespace

std::uint64_t duration(const Interval interval) noexcept {
  return interval.completed > interval.started
             ? interval.completed - interval.started
             : 0u;
}

bool add_timeline(Timeline &timeline, const Interval interval,
                  const std::optional<Timeline::Direction> direction) noexcept {
  const std::uint64_t elapsed = duration(interval);
  if (elapsed == 0u) {
    return true;
  }
  if (timeline.concurrent_count >= timeline.concurrent.size() ||
      (direction.has_value() &&
       timeline.transfer_count >= timeline.transfers.size())) {
    return false;
  }
  timeline.concurrent[timeline.concurrent_count++] = interval;
  if (direction.has_value()) {
    timeline.transfers[timeline.transfer_count] = interval;
    timeline.transfer_directions[timeline.transfer_count++] = *direction;
  }
  return true;
}

bool add_concurrent(Timeline &timeline, const Interval interval) noexcept {
  const std::uint64_t elapsed = duration(interval);
  if (elapsed == 0u) {
    return true;
  }
  if (timeline.concurrent_count >= timeline.concurrent.size()) {
    return false;
  }
  timeline.concurrent[timeline.concurrent_count++] = interval;
  return true;
}

bool add_transfer(Timeline &timeline, const Interval interval,
                  const Timeline::Direction direction) noexcept {
  const std::uint64_t elapsed = duration(interval);
  if (elapsed == 0u) {
    return true;
  }
  if (timeline.transfer_count >= timeline.transfers.size()) {
    return false;
  }
  timeline.transfers[timeline.transfer_count] = interval;
  timeline.transfer_directions[timeline.transfer_count++] = direction;
  return true;
}

bool record_interval(Timeline *const timeline, const Interval interval,
                     const std::optional<Timeline::Direction> direction,
                     ResidencyStats &stats) noexcept {
  if (duration(interval) == 0u) {
    return true;
  }
  if (timeline != nullptr) {
    return add_timeline(*timeline, interval, direction);
  }
  Accumulate(stats.stall_ns, duration(interval));
  return true;
}

void observe_timeline(Timeline &timeline,
                      const residency::ExecutionReceipt &compute,
                      ResidencyStats &stats) noexcept {
  for (std::size_t index = 0u; index < timeline.concurrent_count; ++index) {
    const std::uint64_t elapsed = duration(timeline.concurrent[index]);
    const std::uint64_t hidden =
        intersection(timeline.concurrent[index], compute);
    Accumulate(stats.stall_ns, elapsed - hidden);
  }
  for (std::size_t index = 0u; index < timeline.transfer_count; ++index) {
    const std::uint64_t intersected =
        intersection(timeline.transfers[index], compute);
    Accumulate(stats.overlap_ns, intersected);
    if (timeline.transfer_directions[index] ==
        Timeline::Direction::HostToDevice) {
      Accumulate(stats.h2d_overlap_ns, intersected);
    } else {
      Accumulate(stats.d2h_overlap_ns, intersected);
    }
  }
  timeline = {};
}

} // namespace rund::compute::detail::graph_reduce
