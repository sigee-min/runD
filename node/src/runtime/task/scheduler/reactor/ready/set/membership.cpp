#include "operations.hpp"

#include "../../../../../../host/net/socket/access.hpp"
#include "../../../../../reactor/readiness/handle.hpp"

#include <limits>

namespace rund::node {

::rund::net::ready::Status
Scheduler::AddReadyInterest(const ::rund::net::ready::Set handle,
                            const ::rund::net::ready::Request request) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const auto finish = [this](const ReasonCode code,
                             const ::rund::net::ready::Set set = {}) noexcept {
    ::rund::net::ready::Status result = ReadySetStatus(code, set);
    CompletePrimitiveCommit();
    return result;
  };

  ReactorReadySet *const set =
      ReactorReadySetFind(state_->reactor.reactor_ready_sets, handle);
  if (set == nullptr) {
    return finish(ReasonCode::TaskInvalid);
  }
  const ReactorInterest interest = ::rund::net::ReactorInterestFor(request.interest);
  if (interest == ReactorInterest::None) {
    return finish(ReasonCode::TaskInvalid);
  }
  if (ReactorReadySetHasDuplicate(
          *set, request.socket, interest)) {
    return finish(ReasonCode::TaskInvalid);
  }
  if (set->members.size() >= set->max_members) {
    return finish(ReasonCode::ReactorWaitCapacityExceeded);
  }
  if (set->next_member_index == std::numeric_limits<std::uint32_t>::max()) {
    return finish(ReasonCode::ReactorWaitCapacityExceeded);
  }
  const std::uint32_t index = set->next_member_index++;
  const std::size_t member_capacity = set->members.capacity();
  try {
    set->members.push_back(ReactorReadySetMember{
        .socket = request.socket,
        .fd = ReactorHandleFromPublic(
            ::rund::net::detail::SocketAccess::native(request.socket)),
        .index = index,
        .interest = interest,
    });
  } catch (...) {
    return finish(ReasonCode::ReactorWaitCapacityExceeded);
  }
  state_->reactor.reactor_ready_set_storage_growths +=
      set->members.capacity() != member_capacity ? 1u : 0u;
  RecordReactorReadySetMemberAdded(state_->evidence.metrics);
  return finish(ReasonCode::Ok,
                ::rund::net::ready::Set{.id = set->id, .generation = set->generation});
}

::rund::net::ready::Status
Scheduler::RemoveReadyInterest(const ::rund::net::ready::Set handle,
                               const ::rund::net::ready::Request request) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const auto finish = [this](const ReasonCode code,
                             const ::rund::net::ready::Set set = {}) noexcept {
    ::rund::net::ready::Status result = ReadySetStatus(code, set);
    CompletePrimitiveCommit();
    return result;
  };

  ReactorReadySet *const set =
      ReactorReadySetFind(state_->reactor.reactor_ready_sets, handle);
  if (set == nullptr) {
    return finish(ReasonCode::TaskInvalid);
  }
  const ReactorInterest interest = ::rund::net::ReactorInterestFor(request.interest);
  for (auto member = set->members.begin(); member != set->members.end();
       ++member) {
    if (member->socket == request.socket &&
        member->interest == interest) {
      if (!ReadySetMemberIsCurrent(*member)) {
        RecordReactorReadySetInvalidations(state_->evidence.metrics, 1u);
        return finish(ReasonCode::IoFdInvalid);
      }
      set->members.erase(member);
      RecordReactorReadySetMemberRemoved(state_->evidence.metrics);
      return finish(
          ReasonCode::Ok,
          ::rund::net::ready::Set{.id = set->id, .generation = set->generation});
    }
  }
  return finish(ReasonCode::TaskInvalid);
}

} // namespace rund::node
