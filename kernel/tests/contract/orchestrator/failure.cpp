#include "contract/orchestrator/cases.hpp"
#include "test/assert.hpp"

#include <kernel/dispatch/orchestrator.hpp>

#include <atomic>
#include <string_view>

int RunOrchestratorFailureSignalContract() {
  rund::kernel::FailureSignal signal{};
  rund::kernel::ResetFailureSignal(signal);
  TEST_ASSERT(!rund::kernel::HasFailure(signal));
  rund::kernel::MarkFailure(signal, "first_failure");
  rund::kernel::MarkFailure(signal, "second_failure");
  TEST_ASSERT(rund::kernel::HasFailure(signal));
  TEST_ASSERT(signal.reason.load(std::memory_order_acquire) == std::string_view{"first_failure"});
  return 0;
}
