#include "../reactor/cleanup/request.hpp"
#include "../reactor/registry.hpp"
#include "../state/model/task.hpp"
#include "local.hpp"

#include <algorithm>

namespace rund::node {
namespace {

[[nodiscard]] bool MatchesStop(
    const ::rund::detail::task::StopSourceIdentity wait,
    const ::rund::detail::task::StopSourceIdentity requested) noexcept {
  return requested.valid() && wait == requested;
}

void SortCanceledWaits(std::vector<ReactorWait>& waits) noexcept {
  std::sort(waits.begin(), waits.end(),
            [](const ReactorWait& left, const ReactorWait& right) {
              if (left.task_id != right.task_id) {
                return left.task_id < right.task_id;
              }
              if (left.wait_id != right.wait_id) {
                return left.wait_id < right.wait_id;
              }
              if (left.fd != right.fd) {
                return left.fd < right.fd;
              }
              return left.interest < right.interest;
            });
}

}  // namespace

task::StopState Scheduler::StopRequested(
    const ::rund::detail::task::StopIdentity identity) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  task::StopState result =
      StopRequestedUnsequenced(identity);
  CompletePrimitiveCommit();
  return result;
}

task::Status Scheduler::RequestStop(
    const ::rund::detail::task::StopIdentity identity) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const auto finish = [this](task::Status result) noexcept {
    CompletePrimitiveCommit();
    return result;
  };

  StopSourceRecord *const source = scheduler_cancel::FindStopSource(
      state_->reactor.stop_sources, identity);
  if (source == nullptr) {
    return finish(task::Status::fail(ReasonCode::TaskInvalid));
  }
  if (source->requested) {
    return finish(task::Status::success());
  }

  std::vector<ReactorWait> &canceled = state_->reactor.canceled_wait_scratch;
  try {
    canceled.clear();
    const std::size_t count = ReactorRegistrySize(state_->reactor.reactor);
    canceled.reserve(count);
    for (std::size_t index = 0u; index < count; ++index) {
      const ReactorWait &wait =
          ReactorRegistryWaitAt(state_->reactor.reactor, index);
      if (MatchesStop(wait.stop, identity.source())) {
        canceled.push_back(wait);
      }
    }
  } catch (...) {
    canceled.clear();
    return finish(task::Status::fail(ReasonCode::ReactorWaitCapacityExceeded));
  }
  SortCanceledWaits(canceled);
  source->requested = true;

  bool cleanup_ok = true;
  for (const ReactorWait &wait : canceled) {
    TaskRecord *const record = state_->Find(wait.task_id);
    const std::uint64_t group_id =
        record == nullptr ? 0u : record->wait_source_id;
    if (!ReactorCleanupWait(
            *this, ReactorCleanupRequest{
                       .wait_id = wait.wait_id,
                       .group_id = group_id,
                       .reason = ReasonCode::TaskCancelled,
                       .timeout_cleanup = ReactorTimeoutCleanupPolicy::Required,
                       .remove_ready_backlog = true,
                       .cleanup_siblings = true})) {
      cleanup_ok = false;
    }
  }

  return finish(cleanup_ok ? task::Status::success()
                           : task::Status::fail(ReasonCode::IoPollFailed));
}

} // namespace rund::node
