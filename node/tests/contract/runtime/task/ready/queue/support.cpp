#include "local.hpp"

namespace ready_queue_detail {

rund::task::Task<void> HoldIndex(
    rund::task::channel<std::uint32_t>* const gate) {
  (void)co_await gate->recv();
}

rund::task::Task<void> CompleteIndex(
    std::atomic<std::uint64_t>* const completed) {
  completed->fetch_add(1u, std::memory_order_relaxed);
  co_return;
}


}  // namespace ready_queue_detail
