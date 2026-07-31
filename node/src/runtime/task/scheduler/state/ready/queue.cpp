#include <rund/task/stats/slots.hpp>

#include "../../state/model/join.hpp"
#include "../../state/model/task.hpp"
#include "../../state/model/timer.hpp"
#include "../../state/storage/check.hpp"
#include "../../state/storage.hpp"

#include <cstdlib>

namespace rund::node {

SchedulerReadyState::SchedulerReadyState() noexcept = default;
SchedulerReadyState::~SchedulerReadyState() = default;

namespace {

[[nodiscard]] bool PushGlobal(SchedulerState &state,
                              const std::uint64_t id) noexcept {
  if (!state.ready.ready.push_back(id)) {
    return false;
  }
  ++::rund::detail::task::Stat(
      state.evidence.metrics,
      ::rund::detail::task::StatSlot::GlobalReadyQueuePushes);
  return true;
}

void RecordReadyPush(SchedulerState &state) noexcept {
  ++state.ready.ready_depth;
  ::rund::detail::task::Stat(state.evidence.metrics,
                             ::rund::detail::task::StatSlot::MaxReadyDepth) =
      std::max<std::uint64_t>(
          ::rund::detail::task::Stat(
              state.evidence.metrics,
              ::rund::detail::task::StatSlot::MaxReadyDepth),
          state.ready.ready_depth);
}

[[nodiscard]] bool EnqueueSpawnLocked(SchedulerState &state,
                                      const TaskRecord &record) noexcept {
  if (state.ready.ready_depth >= state.resources.limits.ready_queue_capacity) {
    return false;
  }
  ++::rund::detail::task::Stat(
      state.evidence.metrics,
      ::rund::detail::task::StatSlot::SpawnEpochEnqueueFastPaths);
  if (!PushGlobal(state, record.id)) {
    return false;
  }
  ++::rund::detail::task::Stat(
      state.evidence.metrics,
      ::rund::detail::task::StatSlot::ReadySpawnPushes);
  RecordReadyPush(state);
  return true;
}

void EnqueueProgressLocked(SchedulerState &state,
                           const TaskRecord &record) noexcept {
  if (record.id == 0u || record.state != TaskState::Ready ||
      state.Find(record.id) != &record ||
      state.ready.ready_depth >= state.resources.limits.task_capacity ||
      !PushGlobal(state, record.id)) {
    std::abort();
  }
  ++::rund::detail::task::Stat(
      state.evidence.metrics,
      ::rund::detail::task::StatSlot::ReadyProgressPushes);
  RecordReadyPush(state);
}

} // namespace

std::size_t
SchedulerState::DefaultLaneIndexForTask(const std::uint64_t id,
                                        const std::size_t lane_count) noexcept {
  if (lane_count == 0u || id == 0u) {
    return 0u;
  }
  const std::uint64_t lane_id = plan.installed ? plan.task(id) : id;
  return lane_id == 0u ? 0u
                       : static_cast<std::size_t>((lane_id - 1u) % lane_count);
}

std::size_t
SchedulerState::LaneIndexForTask(const TaskRecord *record,
                                 const std::size_t lane_count) const noexcept {
  if (lane_count == 0u || record == nullptr) {
    return 0u;
  }
  return record->home_lane % lane_count;
}

std::size_t
SchedulerState::LaneIndexForTask(const std::uint64_t id,
                                 const std::size_t lane_count) const noexcept {
  return LaneIndexForTask(Find(id), lane_count);
}

bool SchedulerState::EnqueueSpawn(const TaskRecord &record) noexcept {
  std::lock_guard lock{evidence.mutex};
  RequireSequencer();
  return EnqueueSpawnLocked(*this, record);
}

void SchedulerState::EnqueueProgress(const TaskRecord &record) noexcept {
  std::lock_guard lock{evidence.mutex};
  RequireSequencer();
  EnqueueProgressLocked(*this, record);
}

} // namespace rund::node
