#include <kernel/dispatch/orchestrator.hpp>

namespace rund::kernel {

void ResetFailureSignal(FailureSignal& signal) {
  signal.failed.store(false, std::memory_order_relaxed);
  signal.reason.store("pass", std::memory_order_relaxed);
}

void MarkFailure(FailureSignal& signal, const char* const reason) {
  const char* expected_reason = "pass";
  const char* const failure_reason = reason != nullptr ? reason : "dispatch_failed";
  signal.reason.compare_exchange_strong(expected_reason,
                                        failure_reason,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire);
  signal.failed.store(true, std::memory_order_release);
}

bool HasFailure(const FailureSignal& signal) {
  return signal.failed.load(std::memory_order_acquire);
}

} // namespace rund::kernel
