#include "local.hpp"

namespace rund::node {

::rund::net::ready::many::Result MakeReadyManyCancelledResult() noexcept {
  return ::rund::net::ready::many::Result{ReasonCode::TaskCancelled};
}

void ResetReadyManyResumeTask(TaskRecord &record) noexcept {
  ResetReadyManyResumeWaitState(record);
  record.io_revents = 0;
}

void ResetReadyManyResumeWaitState(TaskRecord &record) noexcept {
  record.wait_id = 0u;
  record.wait_source_id = 0u;
  record.io_result = ReasonCode::Ok;
}

} // namespace rund::node
