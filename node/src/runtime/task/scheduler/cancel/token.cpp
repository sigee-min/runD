#include "access.hpp"

#include "../access.hpp"

namespace rund::node {

::rund::detail::task::StopIdentity
scheduler_access::StopTokenIdentity(const task::stop_token token) noexcept {
  return ::rund::detail::task::StopAccess::Identity(token);
}

} // namespace rund::node
