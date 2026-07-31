#include <rund/task/stats/slots.hpp>

#include "../../state/model/task.hpp"
#include "../../state/model/wake.hpp"
#include "../../state/storage.hpp"

#include <atomic>

namespace rund::node {
namespace {

constexpr std::uint8_t kPending = 0u;
constexpr std::uint8_t kParking = 1u;
constexpr std::uint8_t kParked = 2u;

} // namespace

bool Scheduler::ParkExternal(std::atomic<std::uint8_t> &phase,
                             ExternalWake &wake) noexcept {
  EnsureCurrentCommit();
  TaskRecord *const record = state_->Find(CurrentTaskId());
  if (record == nullptr || record->state != TaskState::Running ||
      !record->coroutine_task) {
    return false;
  }
  std::uint8_t expected = kPending;
  if (!phase.compare_exchange_strong(expected, kParking,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
    return false;
  }

  record->state = TaskState::ExternalBlocked;
  record->dynamic_scope_id = CurrentScopeId();
  record->wake_ticket = 0u;
  record->wake_next = nullptr;
  wake = ExternalWake{
      .record = record, .id = record->id, .lane = record->home_lane};
  {
    std::lock_guard lock{state_->batches.direct_mutex};
    ++state_->batches.direct_jobs_in_flight;
    ++state_->batches.task_direct_jobs_in_flight;
  }
  expected = kParking;
  if (!phase.compare_exchange_strong(expected, kParked,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
    {
      std::lock_guard lock{state_->batches.direct_mutex};
      --state_->batches.direct_jobs_in_flight;
      --state_->batches.task_direct_jobs_in_flight;
    }
    state_->batches.direct_cv.notify_all();
    record->state = TaskState::Running;
    wake = {};
    return false;
  }
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::Parked);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::ExternalParks);
  Record(::rund::detail::task::OperationKind::ExternalPark, ReasonCode::Ok,
         record->id);
  ++::rund::detail::task::Stat(state_->evidence.metrics,
                               ::rund::detail::task::StatSlot::CoroutineParks);
  record->coroutine_parked = true;
  return true;
}

} // namespace rund::node
