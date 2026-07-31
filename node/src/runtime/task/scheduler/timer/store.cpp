#include "store.hpp"

#include <algorithm>
#include <limits>

namespace rund::node {
namespace {

[[nodiscard]] bool TimerWaitAfter(const TimerWait& lhs,
                                  const TimerWait& rhs) noexcept {
  if (lhs.deadline_ns != rhs.deadline_ns) {
    return lhs.deadline_ns > rhs.deadline_ns;
  }
  if (lhs.sequence != rhs.sequence) {
    return lhs.sequence > rhs.sequence;
  }
  return lhs.task_id > rhs.task_id;
}

[[nodiscard]] auto FindTimerWaitId(
    const std::vector<TimerWaitIdIndexEntry>& index,
    const std::uint64_t wait_id) noexcept {
  return std::lower_bound(
      index.begin(),
      index.end(),
      wait_id,
      [](const TimerWaitIdIndexEntry& entry, const std::uint64_t value) {
        return entry.wait_id < value;
      });
}

[[nodiscard]] bool RebuildTimerIndex(
    const std::vector<TimerWait>& timers,
    std::vector<TimerWaitIdIndexEntry>& index) noexcept {
  try {
    index.clear();
    index.reserve(timers.size());
    for (std::size_t slot = 0u; slot < timers.size(); ++slot) {
      index.push_back(TimerWaitIdIndexEntry{
          .wait_id = timers[slot].wait_id,
          .slot = slot,
      });
    }
    std::sort(index.begin(),
              index.end(),
              [](const TimerWaitIdIndexEntry& lhs,
                 const TimerWaitIdIndexEntry& rhs) {
                return lhs.wait_id < rhs.wait_id;
              });
  } catch (...) {
    index.clear();
    return false;
  }
  return true;
}

}  // namespace

bool TimerStorePush(std::vector<TimerWait>& timers,
                    std::vector<TimerWaitIdIndexEntry>& index,
                    TimerWait wait) noexcept {
  try {
    timers.push_back(wait);
    std::push_heap(timers.begin(), timers.end(), TimerWaitAfter);
  } catch (...) {
    return false;
  }
  if (!RebuildTimerIndex(timers, index)) {
    timers.pop_back();
    std::make_heap(timers.begin(), timers.end(), TimerWaitAfter);
    return false;
  }
  return true;
}

bool TimerStoreCancel(std::vector<TimerWait>& timers,
                      std::vector<TimerWaitIdIndexEntry>& index,
                      const std::uint64_t wait_id,
                      TimerWait* const removed) noexcept {
  const auto found = FindTimerWaitId(index, wait_id);
  if (found == index.end() || found->wait_id != wait_id ||
      found->slot >= timers.size() || timers[found->slot].wait_id != wait_id) {
    return false;
  }
  if (removed != nullptr) {
    *removed = timers[found->slot];
  }
  timers[found->slot] = timers.back();
  timers.pop_back();
  std::make_heap(timers.begin(), timers.end(), TimerWaitAfter);
  return RebuildTimerIndex(timers, index);
}

bool TimerStoreContains(const std::vector<TimerWait>& timers,
                        const std::vector<TimerWaitIdIndexEntry>& index,
                        const std::uint64_t wait_id) noexcept {
  return TimerStoreFind(timers, index, wait_id) != nullptr;
}

const TimerWait* TimerStoreFind(const std::vector<TimerWait>& timers,
                                const std::vector<TimerWaitIdIndexEntry>& index,
                                const std::uint64_t wait_id) noexcept {
  const auto found = FindTimerWaitId(index, wait_id);
  if (found == index.end() || found->wait_id != wait_id ||
      found->slot >= timers.size() || timers[found->slot].wait_id != wait_id) {
    return nullptr;
  }
  return &timers[found->slot];
}

bool TimerStorePopDue(std::vector<TimerWait>& timers,
                      std::vector<TimerWaitIdIndexEntry>& index,
                      const Clock::time_point now,
                      TimerWait* const out) noexcept {
  if (timers.empty() || timers.front().deadline > now) {
    return false;
  }
  std::pop_heap(timers.begin(), timers.end(), TimerWaitAfter);
  if (out != nullptr) {
    *out = timers.back();
  }
  timers.pop_back();
  return RebuildTimerIndex(timers, index);
}

int TimerStorePollTimeoutMs(const std::vector<TimerWait>& timers,
                            const Clock::time_point now) noexcept {
  if (timers.empty()) {
    return -1;
  }
  const TimerWait& next = timers.front();
  if (next.deadline <= now) {
    return 0;
  }
  const std::int64_t timeout_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(next.deadline - now)
          .count();
  const std::int64_t ceil_timeout_ms = (timeout_ns + 999999ll) / 1000000ll;
  return ceil_timeout_ms >
                 static_cast<std::int64_t>(std::numeric_limits<int>::max())
             ? std::numeric_limits<int>::max()
             : static_cast<int>(ceil_timeout_ms);
}

}  // namespace rund::node
