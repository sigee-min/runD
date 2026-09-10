#include "host_ring.hpp"

#include "../device/residency/pool.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool extend_ring(const std::uint64_t page_count,
                               const std::uint64_t maximum_capacity,
                               const std::uint64_t page_bytes,
                               std::uint64_t &remaining,
                               std::uint64_t &capacity) noexcept {
  if (page_bytes == 0u || capacity > page_count ||
      capacity > maximum_capacity) {
    return false;
  }
  const std::uint64_t limit = std::min(page_count, maximum_capacity);
  const std::uint64_t added =
      std::min(limit - capacity, remaining / page_bytes);
  std::uint64_t consumed = 0u;
  if (!kernel::checked::mul(added, page_bytes, consumed) ||
      consumed > remaining) {
    return false;
  }
  capacity += added;
  remaining -= consumed;
  return true;
}

} // namespace

bool project_virtual_host_ring_capacities(
    const std::uint64_t page_count, const std::uint64_t execution_capacity,
    const std::uint64_t input_page_bytes, const std::uint64_t output_page_bytes,
    const std::uint64_t host_budget, const std::uint64_t maximum_capacity,
    VirtualHostRingCapacities &result) noexcept {
  result = {};
  std::uint64_t pair_bytes = 0u;
  if (page_count == 0u || execution_capacity == 0u || input_page_bytes == 0u ||
      output_page_bytes == 0u || maximum_capacity < execution_capacity ||
      !kernel::checked::add(input_page_bytes, output_page_bytes, pair_bytes) ||
      pair_bytes == 0u) {
    return false;
  }
  const std::uint64_t bank_budget = host_budget / residency::Pool::BankCount;
  const std::uint64_t balanced = execution_capacity;
  if (balanced > page_count || balanced > maximum_capacity ||
      balanced > bank_budget / pair_bytes) {
    return false;
  }
  std::uint64_t used = 0u;
  if (!kernel::checked::mul(balanced, pair_bytes, used) || used > bank_budget) {
    return false;
  }
  std::uint64_t remaining = bank_budget - used;
  result.input = balanced;
  result.output = balanced;
  if (!extend_ring(page_count, maximum_capacity, input_page_bytes, remaining,
                   result.input) ||
      !extend_ring(page_count, maximum_capacity, output_page_bytes, remaining,
                   result.output)) {
    result = {};
    return false;
  }
  std::uint64_t input_bytes = 0u;
  std::uint64_t output_bytes = 0u;
  std::uint64_t bank_bytes = 0u;
  if (!kernel::checked::mul(result.input, input_page_bytes, input_bytes) ||
      !kernel::checked::mul(result.output, output_page_bytes, output_bytes) ||
      !kernel::checked::add(input_bytes, output_bytes, bank_bytes) ||
      !kernel::checked::mul(bank_bytes, residency::Pool::BankCount,
                            result.storage_bytes) ||
      result.storage_bytes > host_budget) {
    result = {};
    return false;
  }
  return true;
}

} // namespace rund::compute::detail
