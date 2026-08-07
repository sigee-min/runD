#include "../../../../reactor/diagnostics.hpp"
#include "../../../../reactor/readiness/handle.hpp"
#include "../../../../reactor/readiness/mask.hpp"
#include "../../state/storage.hpp"
#include "../apply/policy.hpp"
#include "../backend.hpp"
#include "../change/queue.hpp"
#include "../cleanup/operations.hpp"
#include "../cleanup/request.hpp"
#include "../generation.hpp"
#include "../lease.hpp"
#include "../record.hpp"
#include "../registry.hpp"
#include "../scratch.hpp"
#include "../stats.hpp"
#include "batch.hpp"

#include <rund/task/observation.hpp>

#include <vector>

namespace rund::node {

bool Scheduler::DrainReactorReadyBatch(
    const std::vector<ReactorReady> &ordered,
    const ReactorInvalidChangeToken invalid_change) noexcept {
  ReactorRuntime &reactor = state_->reactor.reactor;
  ReactorLeaseScope leases{state_->reactor.reactor_socket_lease_scratch};
  if (!leases.acquire(ordered, [&reactor](const ReactorReady &ready) noexcept {
        const ReactorWait *const wait =
            ReactorRegistryFindWait(reactor, ready.wait_id);
        if (wait == nullptr) {
          return ReactorLeaseSource{};
        }
        if (!wait->socket && wait->fd_generation == 0u) {
          return ReactorLeaseSource::host();
        }
        return ReactorLeaseSource::socket(wait->socket);
      })) {
    bool invalidated = false;
    return ReactorGenerationCleanupInvalidWaits(*this, &invalidated) &&
           invalidated;
  }
  RecordReactorReadyEvents(state_->evidence.metrics, ordered.size());
  RecordReactorReadyBatch(ordered.size());

  if (!ReactorScratchPrepareHostEvents(
          state_->reactor.reactor_host_event_scratch, ordered.size())) {
    return false;
  }

  ReactorDrainBatch batch = ReactorBuildDrainBatch(reactor, ordered);
  if (batch.ok && invalid_change.valid() &&
      (ReactorRegistryFirstWait(reactor, invalid_change.handle()) !=
           kNoReactorSlot ||
       !ReactorChangeQueueAcknowledgeInvalid(reactor, invalid_change))) {
    batch.ok = false;
  }
  ReactorApplyResult remove_applied = ReactorApplyResult::failed();
  if (batch.ok) {
    ReactorApplyPolicyRecordFlush(reactor, true);
    remove_applied =
        ReactorBackendApplyChanges(reactor, state_->evidence.metrics);
  }
  const bool registration_cleanup_ok =
      ReactorApplyAllowsLogicalProgress(remove_applied);
  if (batch.ready == nullptr || batch.removed_waits == nullptr ||
      batch.ready->size() != batch.removed_waits->size()) {
    return false;
  }

  bool changed = false;
  const std::vector<ReactorReady> &batch_ready = *batch.ready;
  const std::vector<ReactorWait> &batch_removed_waits = *batch.removed_waits;
  std::vector<ReasonCode> &ready_codes =
      state_->reactor.reactor_ready_code_scratch;
  ready_codes.clear();
  if (ready_codes.capacity() < batch_ready.size())
    return false;

  for (std::size_t index = 0u; index < batch_ready.size(); ++index) {
    const ReactorReady &ready = batch_ready[index];
    const ReactorWait &wait = batch_removed_waits[index];
    ReasonCode ready_code = ReasonCode::Ok;
    task::ObservationKind observation_kind = task::ObservationKind::IoReady;
    if (!batch.ok || !registration_cleanup_ok ||
        ready.disposition == ReactorReadyDisposition::PollFailed) {
      ready_code = ReasonCode::IoPollFailed;
      observation_kind = task::ObservationKind::IoPollFailed;
    } else if (ready.disposition == ReactorReadyDisposition::Invalid ||
               (wait.socket && !::rund::net::IsCurrentSocket(wait.socket))) {
      ready_code = ReasonCode::IoFdInvalid;
      observation_kind = task::ObservationKind::IoInvalid;
    }
    if (!ReactorCleanupRemovedWait(*this, ReactorRemovedWaitCleanupRequest{
                                              .wait = wait,
                                              .reason = ready_code,
                                              .cancel_timeout_timer = true,
                                              .remove_ready_backlog = true,
                                              .cleanup_siblings = true,
                                              .wake_owner = false,
                                              .events = ready.events,
                                              .store_event = true})) {
      ready_code = ReasonCode::IoPollFailed;
      observation_kind = task::ObservationKind::IoPollFailed;
      RecordReactorTimeoutCleanupFailure(state_->evidence.metrics);
    }
    ready_codes.push_back(ready_code);

    RecordReactorObservation(observation_kind, ready_code, wait.task_id,
                             wait.wait_id, ReactorHandleForPublic(wait.fd),
                             ReactorInterestBits(wait.interest),
                             ReactorEventBits(ready.events));
    state_->reactor.reactor_host_event_scratch.push_back(
        MakeReactorHostEvent(ready_code, wait.task_id, wait.host_handle_id));
  }

  if (!RecordHostEvents(state_->reactor.reactor_host_event_scratch)) {
    changed = true;
  }

  for (std::size_t index = 0u; index < batch_ready.size(); ++index) {
    const ReactorReady &ready = batch_ready[index];
    const ReactorWait &wait = batch_removed_waits[index];
    const ReasonCode ready_code = ready_codes[index];
    if (ReactorCleanupRemovedWait(*this, ReactorRemovedWaitCleanupRequest{
                                             .wait = wait,
                                             .reason = ready_code,
                                             .cancel_timeout_timer = false,
                                             .remove_ready_backlog = false,
                                             .cleanup_siblings = false,
                                             .wake_owner = true,
                                             .events = ready.events})) {
      changed = true;
      continue;
    }
    TaskRecord *const record = state_->Find(wait.task_id);
    if (record == nullptr || record->state != TaskState::IoBlocked) {
      changed = true;
      continue;
    }
    reactor_cancel_cleanup::WakeTask(
        *this, *record, wait.wait_id, wait.fd, wait.interest, ready.events, 0,
        ready_code, true);
    changed = true;
  }
  reactor.ready.clear();
  return changed;
}

} // namespace rund::node
