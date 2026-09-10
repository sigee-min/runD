#include "internal.hpp"

#include "../registry.hpp"

#include <atomic>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency {

std::uint64_t next_cpu_book_domain() noexcept {
  static std::atomic<std::uint64_t> next{1u};
  std::uint64_t value = next.load(std::memory_order_relaxed);
  while (value != 0u && value != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(value, value + 1u, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return value;
    }
  }
  return 0u;
}

std::uint64_t next_cpu_authority_owner() noexcept {
  static std::atomic<std::uint64_t> next{1u};
  std::uint64_t value = next.load(std::memory_order_relaxed);
  while (value != 0u && value != std::numeric_limits<std::uint64_t>::max()) {
    if (next.compare_exchange_weak(value, value + 1u, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return value;
    }
  }
  return 0u;
}

registry_model::PendingCpu::~PendingCpu() noexcept {
  if (active && owner != nullptr &&
      owner->cpu_graph_state_.pending_cpu == key) {
    owner->cpu_graph_state_.pending_cpu = {};
  }
}

void registry_model::PendingCpu::publish() noexcept {
  if (active && owner != nullptr &&
      owner->cpu_graph_state_.pending_cpu == key) {
    owner->cpu_graph_state_.pending_cpu = {};
  }
  active = false;
}

} // namespace rund::compute::detail::residency

namespace rund::compute::detail::residency {

Authority::Authority() noexcept
    : credentials_{.owner_id = next_cpu_authority_owner()} {}

} // namespace rund::compute::detail::residency
