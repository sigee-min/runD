#include "internal.hpp"

#include <utility>

namespace rund::compute::detail::accel_backend::service_free_direct {

void complete(void *const raw,
              node::accel::detail::ServiceFreeDirectFinal &&final) noexcept {
  auto *const run = static_cast<Run *>(raw);
  if (run == nullptr) {
    return;
  }
  run->final = std::move(final);
  ++run->callback_count;
  run->done.store(true, std::memory_order_release);
  run->done.notify_one();
}

} // namespace rund::compute::detail::accel_backend::service_free_direct
