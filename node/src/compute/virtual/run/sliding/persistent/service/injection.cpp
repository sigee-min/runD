#include "internal.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

std::atomic_bool unknown_service_once{};

} // namespace

void inject_persistent_service_unknown_once() noexcept {
  unknown_service_once.store(true, std::memory_order_release);
}

bool consume_persistent_service_unknown_injection() noexcept {
  return unknown_service_once.exchange(false, std::memory_order_acq_rel);
}

} // namespace rund::compute::detail::sliding_product_detail
