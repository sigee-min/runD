#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

std::atomic_bool begin_failure_once{};

} // namespace

bool consume_persistent_pipeline_begin_failure_once() noexcept {
  return begin_failure_once.exchange(false, std::memory_order_acq_rel);
}

void inject_persistent_pipeline_begin_failure_once() noexcept {
  begin_failure_once.store(true, std::memory_order_release);
}

} // namespace rund::compute::detail::sliding_product_detail
