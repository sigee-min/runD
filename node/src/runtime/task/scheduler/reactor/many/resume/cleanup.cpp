#include "local.hpp"

namespace rund::node {

bool CleanupReadyManyResume(Scheduler &scheduler, const std::uint64_t group_id,
                            const ReasonCode ready_code) noexcept {
  return ReactorCleanupWait(scheduler,
                            ReactorCleanupRequest{.wait_id = 0u,
                                                  .group_id = group_id,
                                                  .reason = ready_code,
                                                  .cancel_timeout_timer = false,
                                                  .remove_ready_backlog = true,
                                                  .cleanup_siblings = true,
                                                  .erase_group = true,
                                                  .wake_owner = false});
}

} // namespace rund::node
