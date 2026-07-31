#include <rund/task/stats/slots.hpp>

#include <rund/counter.hpp>

#include "../state/model/lane.hpp"
#include "../state/storage.hpp"

#include <algorithm>
#include <limits>
#include <thread>

namespace rund::node {

thread_local TaskLane *active_task_lane = nullptr;

namespace {

[[nodiscard]] std::uint32_t
ClampSpinBudget(const std::uint64_t value) noexcept {
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      value, std::numeric_limits<std::uint32_t>::max()));
}

[[nodiscard]] std::uint64_t
ConfiguredTaskWindow(const ::rund::SchedulerConfig &limits) noexcept {
  return std::max<std::uint64_t>(
      1u, std::max<std::uint64_t>(limits.task_capacity,
                                  limits.ready_queue_capacity));
}

} // namespace

[[nodiscard]] bool TakeLaneCompletion(TaskLane &target,
                                      const std::uint64_t sequence) noexcept {
  if (target.completed_job_sequence.load(std::memory_order_acquire) !=
      sequence) {
    return false;
  }
  target.completed_job_sequence.store(0u, std::memory_order_release);
  target.completed_job_signal.store(0u, std::memory_order_release);
  target.completion_wait_requested = false;
  target.completion_signal_wait_requested = false;
  return true;
}

void PublishLaneCompletion(TaskLane &target,
                           const std::uint64_t sequence) noexcept {
  target.completed_job_sequence.store(sequence, std::memory_order_release);
  target.completed_job_signal.store(sequence, std::memory_order_release);
}

std::uint32_t Scheduler::LaneCompletionSpinLoadBudget() const noexcept {
  const std::uint64_t worker_width =
      std::max<std::uint64_t>(1u, state_->lanes.lanes.size());
  const std::uint64_t capacity_budget =
      ConfiguredTaskWindow(state_->resources.limits);
  const std::uint64_t capacity_lane_share =
      (capacity_budget + worker_width - 1u) / worker_width;
  const std::uint64_t capacity_window_budget =
      capacity_budget + capacity_lane_share;
  return ClampSpinBudget(capacity_window_budget);
}

std::uint32_t Scheduler::LaneCompletionSpinYieldStride() const noexcept {
  return ClampSpinBudget(ConfiguredTaskWindow(state_->resources.limits));
}

std::uint32_t Scheduler::LaneHotStandbySpinBudget() const noexcept {
  const std::uint64_t worker_width =
      std::max<std::uint64_t>(1u, state_->lanes.lanes.size());
  const std::uint64_t capacity_budget =
      ConfiguredTaskWindow(state_->resources.limits);
  return ClampSpinBudget(
      std::max<std::uint64_t>(LaneCompletionSpinLoadBudget(),
                              ::rund::detail::counter::SaturatingMultiply(
                                  capacity_budget, worker_width)));
}

std::chrono::microseconds Scheduler::NestedJoinPollInterval() const noexcept {
  return std::chrono::microseconds{
      std::max<std::uint64_t>(1u, state_->lanes.lanes.size())};
}

bool Scheduler::TryConsumeCompletedJobSpin(
    TaskLane &lane, const std::uint64_t sequence,
    const bool release_root_reservation) noexcept {
  if (sequence == 0u) {
    return false;
  }
  const std::uint32_t spin_loads = LaneCompletionSpinLoadBudget();
  const std::uint32_t yield_stride = LaneCompletionSpinYieldStride();
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneCompletionSpinWaits);
  for (std::uint32_t spin = 0u; spin < spin_loads; ++spin) {
    if (lane.completed_job_signal.load(std::memory_order_acquire) == sequence) {
      std::lock_guard<std::mutex> lock(lane.mutex);
      if (TakeLaneCompletion(lane, sequence)) {
        if (release_root_reservation) {
          lane.root_reserved.store(false, std::memory_order_release);
        }
        ++::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::LaneCompletionSpinHits);
        return true;
      }
    }
    if ((spin + 1u) % yield_stride == 0u) {
      std::this_thread::yield();
    }
  }
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneCompletionSpinMisses);
  return false;
}

bool Scheduler::TryConsumeLaneCompletionGroupSignal(
    TaskLane &lane, const std::uint64_t sequence) noexcept {
  if (sequence == 0u) {
    return false;
  }
  const std::uint32_t spin_loads = LaneCompletionSpinLoadBudget();
  const std::uint32_t yield_stride = LaneCompletionSpinYieldStride();
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneCompletionSpinWaits);
  for (std::uint32_t spin = 0u; spin < spin_loads; ++spin) {
    if (lane.completed_job_signal.load(std::memory_order_acquire) == sequence &&
        lane.completed_job_sequence.load(std::memory_order_acquire) ==
            sequence) {
      lane.completed_job_sequence.store(0u, std::memory_order_relaxed);
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneCompletionSpinHits);
      return true;
    }
    if ((spin + 1u) % yield_stride == 0u) {
      std::this_thread::yield();
    }
  }
  ++::rund::detail::task::Stat(
      state_->evidence.metrics,
      ::rund::detail::task::StatSlot::LaneCompletionSpinMisses);
  return false;
}

bool Scheduler::WaitForLaneCompletionSignal(
    TaskLane &lane, const std::uint64_t sequence) noexcept {
  if (sequence == 0u) {
    return false;
  }
  {
    std::unique_lock<std::mutex> lock(lane.mutex);
    if (lane.completed_job_sequence.load(std::memory_order_acquire) ==
        sequence) {
      return TakeLaneCompletion(lane, sequence);
    }
    if (!lane.completion_signal_wait_requested) {
      lane.completion_signal_wait_requested = true;
      ++::rund::detail::task::Stat(
          state_->evidence.metrics,
          ::rund::detail::task::StatSlot::LaneCompletionSingleWakes);
    }
  }
  std::uint64_t observed =
      lane.completed_job_signal.load(std::memory_order_acquire);
  while (observed != sequence) {
    lane.completed_job_signal.wait(observed, std::memory_order_acquire);
    observed = lane.completed_job_signal.load(std::memory_order_acquire);
  }
  std::lock_guard<std::mutex> lock(lane.mutex);
  return TakeLaneCompletion(lane, sequence);
}

} // namespace rund::node
