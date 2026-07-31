#pragma once

#include "../model/context.hpp"
#include "../storage.hpp"

#include <cstdlib>

namespace rund::node {

inline void SchedulerState::RequireSequencer() const noexcept {
  if (active_scheduler_context != nullptr &&
      active_scheduler_context->scheduler == owner &&
      !active_scheduler_context->commit_acquired) {
    std::abort();
  }
}

} // namespace rund::node
