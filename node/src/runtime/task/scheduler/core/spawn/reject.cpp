#include <rund/task/stats/slots.hpp>

#include "../../state/storage.hpp"

#include <rund/task/coroutine.hpp>

namespace rund::node {

void Scheduler::DestroyRejectedSpawnPayload(
    const ::rund::detail::task::CoroutineStart coroutine) noexcept {
  if (coroutine.frame != nullptr && coroutine.ops != nullptr &&
      coroutine.ops->destroy != nullptr) {
    coroutine.ops->destroy(coroutine.frame);
    ++::rund::detail::task::Stat(
        state_->evidence.metrics,
        ::rund::detail::task::StatSlot::CoroutineFrameDestroys);
  }
}

} // namespace rund::node
