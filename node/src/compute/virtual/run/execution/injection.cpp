#include "../execution.hpp"

#include <atomic>

namespace rund::compute::detail {
namespace {

std::atomic<bool> close_failure_once{};

} // namespace

void inject_virtual_execution_close_failure_once() noexcept {
  close_failure_once.store(true, std::memory_order_release);
}

bool consume_virtual_execution_close_failure_once() noexcept {
  return close_failure_once.exchange(false, std::memory_order_acq_rel);
}

} // namespace rund::compute::detail
