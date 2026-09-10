#pragma once

#include <cstdint>

namespace rund::compute::detail {

struct VirtualHostRingCapacities final {
  std::uint64_t input{};
  std::uint64_t output{};
  std::uint64_t storage_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const VirtualHostRingCapacities &) const noexcept = default;
};

// Projects the two-bank accelerator Host budget into independently tagged
// input-cache and output-staging depths. Both rings first receive the complete
// executable depth. Any unpaired remainder extends the reusable input cache
// before the transient output ring.
[[nodiscard]] bool project_virtual_host_ring_capacities(
    std::uint64_t page_count, std::uint64_t execution_capacity,
    std::uint64_t input_page_bytes, std::uint64_t output_page_bytes,
    std::uint64_t host_budget, std::uint64_t maximum_capacity,
    VirtualHostRingCapacities &result) noexcept;

} // namespace rund::compute::detail
