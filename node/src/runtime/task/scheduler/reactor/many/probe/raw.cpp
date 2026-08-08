#include "raw.hpp"

#include "../../../../../reactor/platform.hpp"
#include "../../../../../reactor/readiness/handle.hpp"
#include "../../../../../reactor/readiness/mask.hpp"
#include "../../../state/storage.hpp"
#include "../events.hpp"

#include <cerrno>

namespace rund::node {

ReactorManyProbeResult
ReactorProbeManyReady(Scheduler &scheduler, ReactorPlatform &platform,
                      const std::uint64_t task_id, const std::uint32_t limit,
                      const std::span<const ReactorManyRequest> requests,
                      std::vector<BatchIoPollRequest> &poll_requests,
                      std::vector<BatchIoReady> &ready_results,
                      ReactorManyGroup &group,
                      std::vector<ReactorManyEventSlot> &event_slots) noexcept {
  ReactorManyEventSlotsReset(event_slots, group.request_count);
  if (event_slots.size() != group.request_count) {
    return ReactorManyProbeResult{
        .code = ReasonCode::ReactorWaitCapacityExceeded,
    };
  }

  const BatchIoProbeResult probed = ReactorProbeManyReadyNow(
      platform, requests, poll_requests, ready_results);
  switch (probed.disposition()) {
  case BatchIoProbeDisposition::BackendUnavailable:
    return ReactorManyProbeResult{
        .code = ReasonCode::ReactorBackendUnavailable,
    };
  case BatchIoProbeDisposition::Failed: {
    const ReactorManyRequest *const first =
        requests.empty() ? nullptr : &requests.front();
    const ReactorHandle handle =
        first == nullptr ? kInvalidReactorHandle : first->fd;
    const ReactorInterest interest =
        first == nullptr ? ReactorInterest::None : first->interest;
    const std::uint64_t wait_id = first == nullptr ? 0u : first->wait_id;
    const std::uint64_t host_handle_id =
        first == nullptr ? 0u : first->socket.id();
    scheduler.RecordReactorObservation(task::ObservationKind::IoPollFailed,
                                       ReasonCode::IoPollFailed, task_id,
                                       wait_id, ReactorHandleForPublic(handle),
                                       ReactorInterestBits(interest), 0);
    if (!scheduler.RecordReactorHostEvent(ReasonCode::IoPollFailed, task_id,
                                          host_handle_id)) {
      return ReactorManyProbeResult{
          .code = ReasonCode::HostReplayEventMismatch,
      };
    }
    return ReactorManyProbeResult{
        .code = ReasonCode::IoPollFailed,
    };
  }
  case BatchIoProbeDisposition::Success:
    break;
  }

  ReactorManyProbeResult result{};
  for (const BatchIoReady &ready : ready_results) {
    const ReactorManyRequest *const request =
        ReactorManyProbeRequestForReady(requests, ready);
    if (request == nullptr) {
      continue;
    }
    if (!ready.invalid &&
        !ReactorEventsMatch(ready.events, request->interest)) {
      continue;
    }
    const ReasonCode ready_code =
        ready.invalid ? ReasonCode::IoFdInvalid : ReasonCode::Ok;
    scheduler.RecordReactorObservation(
        ready.invalid ? task::ObservationKind::IoInvalid
                      : task::ObservationKind::IoReady,
        ready_code, task_id, request->wait_id,
        ReactorHandleForPublic(request->fd),
        ReactorInterestBits(request->interest), ReactorEventBits(ready.events));
    if (!scheduler.RecordReactorHostEvent(ready_code, task_id,
                                          request->socket.id())) {
      return ReactorManyProbeResult{
          .code = ReasonCode::HostReplayEventMismatch,
          .total_ready = result.total_ready,
      };
    }
    if (result.total_ready < limit) {
      ReactorManyEventSlotsAppend(group, *request, ready.events, ready_code,
                                  event_slots);
    }
    ++result.total_ready;
    if (ready.invalid) {
      result.code = ReasonCode::IoFdInvalid;
      return result;
    }
  }
  return result;
}

BatchIoProbeResult
ReactorProbeManyReadyNow(ReactorPlatform &platform,
                         const std::span<const ReactorManyRequest> requests,
                         std::vector<BatchIoPollRequest> &poll_requests,
                         std::vector<BatchIoReady> &ready) noexcept {
  poll_requests.clear();
  try {
    poll_requests.reserve(requests.size());
    for (const ReactorManyRequest &request : requests) {
      poll_requests.push_back(BatchIoPollRequest{
          .index = request.slot,
          .handle = request.fd,
          .interest = request.interest,
      });
    }
  } catch (...) {
    poll_requests.clear();
    ready.clear();
    return BatchIoProbeResult::failed(ENOMEM);
  }

  return ProbeReactorPlatformNow(platform, poll_requests.data(),
                                 poll_requests.size(), ready);
}

const ReactorManyRequest *ReactorManyProbeRequestForReady(
    const std::span<const ReactorManyRequest> requests,
    const BatchIoReady &ready) noexcept {
  const std::size_t direct_index = ready.index;
  if (direct_index < requests.size() &&
      requests[direct_index].slot == ready.index) {
    return &requests[direct_index];
  }
  return nullptr;
}

} // namespace rund::node
