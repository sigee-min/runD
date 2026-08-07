#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "local.hpp"

#include "../../../../../../host/net/socket/access.hpp"
#include "../../../../../reactor/readiness/handle.hpp"
#include "../../../../../reactor/readiness/mask.hpp"

namespace rund::node {

bool ReadyManyParkCreateGroupAndRequests(
    SchedulerState &state, ReadyManyEntry &entry, const std::uint64_t group_id,
    const std::uint64_t timer_wait_id, const std::uint64_t stop_source_id,
    const std::uint64_t stop_generation, const std::uint64_t stop_epoch,
    const ::rund::net::ready::Set ready_set) noexcept {
  const std::size_t request_count = entry.requests.size();
  const std::size_t first_request = state.reactor.reactor_many_requests.size();
  if (request_count == 0u ||
      request_count > std::numeric_limits<std::uint32_t>::max() ||
      first_request > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::size_t request_capacity =
      state.reactor.reactor_many_requests.capacity();
  const std::size_t slot_capacity =
      state.reactor.reactor_many_event_slots.capacity();
  const std::size_t group_capacity =
      state.reactor.reactor_many_groups.capacity();
  try {
    state.reactor.reactor_many_requests.insert(
        state.reactor.reactor_many_requests.end(), entry.requests.begin(),
        entry.requests.end());
    std::span<ReactorManyRequest> stored =
        std::span<ReactorManyRequest>{state.reactor.reactor_many_requests}
            .subspan(first_request, request_count);
    for (std::size_t slot = 0u; slot < stored.size(); ++slot) {
      stored[slot].group_id = group_id;
      stored[slot].wait_id = state.identity.next_wait_id++;
      stored[slot].slot = static_cast<std::uint32_t>(slot);
    }
    if (!ReactorManyEventSlotsInsertGroup(
            state.reactor.reactor_many_event_slots,
            static_cast<std::uint32_t>(first_request),
            static_cast<std::uint32_t>(request_count))) {
      state.reactor.reactor_many_requests.resize(first_request);
      return false;
    }
    state.reactor.reactor_many_groups.push_back(ReactorManyGroup{
        .group_id = group_id,
        .task_id = entry.record->id,
        .ready_set = ready_set,
        .timer_wait_id = timer_wait_id,
        .stop_source_id = stop_source_id,
        .stop_generation = stop_generation,
        .stop_epoch = stop_epoch,
        .first_request = static_cast<std::uint32_t>(first_request),
        .request_count = static_cast<std::uint32_t>(request_count),
        .max_events = entry.output_limit,
    });
  } catch (...) {
    state.reactor.reactor_many_requests.resize(first_request);
    ReactorManyEventSlotsEraseGroup(state.reactor.reactor_many_event_slots,
                                    static_cast<std::uint32_t>(first_request),
                                    static_cast<std::uint32_t>(request_count));
    return false;
  }
  state.reactor.reactor_many_storage_growths +=
      state.reactor.reactor_many_requests.capacity() != request_capacity ? 1u
                                                                         : 0u;
  state.reactor.reactor_many_storage_growths +=
      state.reactor.reactor_many_event_slots.capacity() != slot_capacity ? 1u
                                                                         : 0u;
  state.reactor.reactor_many_storage_growths +=
      state.reactor.reactor_many_groups.capacity() != group_capacity ? 1u : 0u;
  state.reactor.reactor_many_request_copies += request_count;
  return true;
}

bool ReadyManyAccess::ParkRegisterWaits(
    Scheduler &scheduler, ReadyManyEntry &entry,
    const std::uint64_t stop_source_id, const std::uint64_t stop_generation,
    const std::uint64_t stop_epoch) noexcept {
  SchedulerState &state = *scheduler.state_;
  const ReactorManyGroup *const group =
      ReactorManyFindGroup(state.reactor.reactor_many_groups, entry.group_id);
  if (group == nullptr) {
    return false;
  }
  const std::span<const ReactorManyRequest> requests =
      ReactorManyRequests(state.reactor.reactor_many_requests, *group);
  if (requests.size() != group->request_count) {
    return false;
  }
  for (const ReactorManyRequest &request : requests) {
    const ReactorWait wait{
        .socket = request.socket,
        .task_id = entry.record->id,
        .wait_id = request.wait_id,
        .host_handle_id = ::rund::net::detail::SocketAccess::id(
            ReactorHandleForPublic(request.fd)),
        .fd_generation =
            ::rund::net::detail::SocketAccess::generation(request.socket),
        .stop_source_id = stop_source_id,
        .stop_generation = stop_generation,
        .stop_epoch = stop_epoch,
        .fd = request.fd,
        .interest = request.interest,
    };
    if (!ReactorRegistryAddWait(state.reactor.reactor, wait) ||
        !ReactorRegistryCollectChangesForWaitAdd(state.reactor.reactor, wait)) {
      return false;
    }
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(
            state.evidence.metrics,
            ::rund::detail::task::StatSlot::ReactorWaits),
        1u);
    RecordReactorWaitRegistered(state.evidence.metrics);
    scheduler.Record(::rund::detail::task::OperationKind::IoPark,
                     ReasonCode::Ok, entry.record->id, 0u, request.wait_id, 0u,
                     request.fd, ReactorInterestBits(request.interest), 0);
  }
  return true;
}

} // namespace rund::node
