#include "../../../host/io/validation.hpp"
#include "../../platform/io.hpp"
#include "../../reactor/readiness/mask.hpp"
#include "../scheduler/io.hpp"

#include <rund/host/io.hpp>

namespace rund::host::io {

task::IoOp readable(const FdView fd) noexcept {
  const detail::FdIdentity identity = detail::Project(fd);
  if (!identity.live()) {
    return ::rund::detail::task::OpAccess::io(
        task::Status::fail(ReasonCode::IoFdInvalid), 0, false, identity.native,
        ::rund::node::ReactorInterestBits(::rund::node::ReactorInterest::Read),
        identity.host_id, 0u);
  }
  return ::rund::node::scheduler_io::WaitAdmittedFd(
      identity,
      ::rund::node::ReactorInterestBits(::rund::node::ReactorInterest::Read));
}

task::IoOp writable(const FdView fd) noexcept {
  const detail::FdIdentity identity = detail::Project(fd);
  if (!identity.live()) {
    return ::rund::detail::task::OpAccess::io(
        task::Status::fail(ReasonCode::IoFdInvalid), 0, false, identity.native,
        ::rund::node::ReactorInterestBits(::rund::node::ReactorInterest::Write),
        identity.host_id, 0u);
  }
  return ::rund::node::scheduler_io::WaitAdmittedFd(
      identity,
      ::rund::node::ReactorInterestBits(::rund::node::ReactorInterest::Write));
}

} // namespace rund::host::io
