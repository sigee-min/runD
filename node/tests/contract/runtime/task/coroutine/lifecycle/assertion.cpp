#include <rund/task/status.hpp>
#include "../local.hpp"

namespace rund::node::test_contract::coroutine {

int AssertCoroutineAwaitSuccess(const rund::Session::Result &report,
                                const rund::task::Status &join,
                                const bool handle_valid,
                                const std::atomic<std::uint32_t> &after_await,
                                const std::uint64_t expected_spawned,
                                const std::uint64_t expected_completed) {
  TEST_ASSERT(report.ok());
  TEST_ASSERT(handle_valid);
  TEST_ASSERT(join.ok());
  TEST_ASSERT(join.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(after_await.load(std::memory_order_acquire) == 1u);
  TEST_ASSERT(report.tasks().spawned() == expected_spawned);
  TEST_ASSERT(report.tasks().completed() == expected_completed);
  TEST_ASSERT(report.tasks().failed() == 0u);
  return 0;
}

} // namespace rund::node::test_contract::coroutine
