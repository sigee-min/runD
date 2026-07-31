#include "../../../../reactor/readiness/handle.hpp"
#include "../../state/model/task.hpp"
#include "../../state/storage.hpp"

#include "../generation.hpp"
#include "../state.hpp"

namespace rund::node {

bool Scheduler::ValidateTimedReactorWait(
    const int fd, const std::chrono::nanoseconds timeout,
    const std::uint64_t task_id, const std::uint64_t host_handle_id,
    const std::uint64_t fd_generation, const std::uint64_t stop_scheduler_id,
    const std::uint64_t stop_source_id, const std::uint64_t stop_generation,
    const std::uint64_t stop_epoch, TaskRecord *&record,
    std::uint64_t &wait_host_handle_id,
    ::rund::detail::task::IoDecision &result) noexcept {
  record = state_->Find(task_id);
  if (record == nullptr || record->state != TaskState::Running) {
    result = FailIo(ReasonCode::TaskContextMissing);
    CompletePrimitiveCommit();
    return false;
  }
  if (timeout.count() < 0) {
    result = FailIo(ReasonCode::TimerDurationInvalid);
    CompletePrimitiveCommit();
    return false;
  }
  if (fd < 0) {
    result = FailIo(ReasonCode::IoFdInvalid);
    CompletePrimitiveCommit();
    return false;
  }
  if (stop_source_id != 0u || stop_generation != 0u || stop_epoch != 0u) {
    const task::StopState stop = StopRequestedUnsequenced(
        stop_scheduler_id, stop_source_id, stop_generation, stop_epoch);
    if (!stop) {
      result = FailIo(stop.code());
      CompletePrimitiveCommit();
      return false;
    }
    if (stop.requested()) {
      result = FailIo(ReasonCode::TaskCancelled);
      CompletePrimitiveCommit();
      return false;
    }
  }

  wait_host_handle_id = ReactorHostHandleId(fd, host_handle_id);
  if (fd_generation != 0u) {
    ReasonCode generation_failure = ReasonCode::Ok;
    if (!ReactorGenerationCleanupStaleWaits(*this, ReactorHandleFromPublic(fd),
                                            fd_generation,
                                            &generation_failure)) {
      result = FailIo(generation_failure);
      CompletePrimitiveCommit();
      return false;
    }
  }
  return true;
}

} // namespace rund::node
