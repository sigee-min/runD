#include "local.hpp"

#include "../../../../../host/net/socket/access.hpp"

namespace rund::node {

ReadyManyEntry ReadyManyAccess::PrepareEntry(
    Scheduler &scheduler, const std::span<const ReactorManyRequest> requests,
    const std::span<::rund::net::ready::Event> out,
    const ::rund::net::ready::many::Budget budget,
    const ::rund::detail::task::StopIdentity stop_identity) noexcept {
  SchedulerState &state = *scheduler.state_;
  ReadyManyEntry entry{};
  (void)scheduler.TrapLaneOwnedSegmentPrimitive(
      ::rund::detail::task::OperationKind::IoPark);
  scheduler.EnsureCurrentCommit();
  entry.task_id = scheduler.CurrentTaskId();
  entry.record = state.Find(entry.task_id);
  if (entry.record == nullptr || entry.record->state != TaskState::Running) {
    entry.code = ReasonCode::TaskContextMissing;
    scheduler.CompletePrimitiveCommit();
    return entry;
  }
  if (!entry.record->coroutine_task) {
    scheduler.SetLeafFailure(*entry.record,
                             ReasonCode::TaskLeafPrimitiveForbidden);
    entry.code = ReasonCode::TaskLeafPrimitiveForbidden;
    scheduler.CompletePrimitiveCommit();
    return entry;
  }
  if (!stop_identity.empty()) {
    const task::StopState stop =
        scheduler.StopRequestedUnsequenced(stop_identity);
    if (!stop) {
      entry.code = stop.code();
      scheduler.CompletePrimitiveCommit();
      return entry;
    }
    if (stop.requested()) {
      entry.code = ReasonCode::TaskCancelled;
      scheduler.CompletePrimitiveCommit();
      return entry;
    }
  }
  entry.stop = stop_identity.source();

  entry.output_limit = OutputLimit(out, budget);
  RecordReactorReadyManyRequest(state.evidence.metrics);
  if (requests.size() > state.resources.limits.reactor_wait_capacity ||
      requests.size() > std::numeric_limits<std::uint32_t>::max()) {
    entry.code = ReasonCode::ReactorWaitCapacityExceeded;
    scheduler.CompletePrimitiveCommit();
    return entry;
  }
  const ReasonCode validation = ReactorManyValidateRequests(
      requests, state.reactor.reactor_many_index_scratch,
      &state.reactor.reactor_many_validation_comparisons);
  if (validation != ReasonCode::Ok) {
    entry.code = validation;
    scheduler.CompletePrimitiveCommit();
    return entry;
  }

  entry.requests = requests;
  for (const ReactorManyRequest &request : entry.requests) {
    const std::uint64_t generation =
        ::rund::net::detail::SocketAccess::generation(request.socket);
    if (generation == 0u) {
      continue;
    }
    ReasonCode generation_failure = ReasonCode::Ok;
    if (!ReactorGenerationCleanupStaleWaits(scheduler, request.fd, generation,
                                            &generation_failure)) {
      entry.code = generation_failure;
      scheduler.CompletePrimitiveCommit();
      return entry;
    }
  }

  entry.code = ReasonCode::Ok;
  return entry;
}

} // namespace rund::node
