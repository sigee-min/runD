#include "operations.hpp"

namespace rund::node {

bool ReactorCleanupWait(Scheduler& scheduler,
                        ReactorCleanupRequest request) noexcept {
  if (request.group_id != 0u) {
    return reactor_cancel_cleanup::CleanupGroup(scheduler, request);
  }
  if (request.wait_id == 0u) {
    return true;
  }
  return reactor_cancel_cleanup::CleanupSingleWait(scheduler, request);
}

}  // namespace rund::node
