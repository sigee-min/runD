#include "operations.hpp"

#include "../../../../../../host/net/registry/socket.hpp"
#include "../../../../../reactor/readiness/handle.hpp"

namespace rund::node {

::rund::net::ready::Status ReadySetStatus(const ReasonCode code,
                                   const ::rund::net::ready::Set set) noexcept {
  ::rund::net::ready::Status result{code};
  result.set = set;
  return result;
}

::rund::net::ready::many::Result ReadySetWaitStatus(const ReasonCode code,
                                        const std::uint32_t events,
                                        const bool budget_exhausted) noexcept {
  ::rund::net::ready::many::Result result{code};
  result.events = events;
  result.budget_exhausted = budget_exhausted;
  return result;
}

bool ReadySetMemberIsCurrent(const ReactorReadySetMember &member) noexcept {
  return member.fd != kInvalidReactorHandle &&
         ::rund::net::IsCurrentSocket(member.socket);
}

} // namespace rund::node
