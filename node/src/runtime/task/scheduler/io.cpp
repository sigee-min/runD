#include "io.hpp"
#include "state/storage.hpp"

#include "../../../host/io/access.hpp"

namespace rund::node::scheduler_io {

task::IoOp WaitAdmittedFd(const ::rund::host::io::detail::FdIdentity fd,
                          const short interest) noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::detail::task::OpAccess::io(
        task::Status::fail(ReasonCode::NodeRuntimeMissing), 0, false, fd.native,
        interest, fd.host_id, 0u);
  }
  if (scheduler->CurrentTaskIsCoroutine()) {
    return ::rund::detail::task::OpAccess::io(
        task::Status::success(), 0, true, fd.native, interest, fd.host_id, 0u);
  }
  const ::rund::detail::task::IoDecision decision =
      scheduler->WaitReactor(fd.native, interest, fd.host_id);
  return ::rund::detail::task::OpAccess::io(decision.status, decision.revents,
                                            false, fd.native, interest,
                                            fd.host_id, 0u);
}

} // namespace rund::node::scheduler_io
