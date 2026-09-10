#include "../internal.hpp"

#include "../../../../device/residency/pool.hpp"
#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::compute::detail {

bool virtual_valid_regions(
    const residency::Pool &pool,
    const std::array<residency::FrameRegion, residency::Pool::BankCount>
        &regions,
    const residency::FrameTier tier, const residency::FrameRole role,
    const std::uint32_t capacity) noexcept {
  const std::uint32_t authority_capacity = pool.authority().frame_capacity();
  for (const residency::FrameRegion region : regions) {
    if (region.tier != tier || region.role != role ||
        region.count != capacity || region.first > authority_capacity ||
        region.count > authority_capacity - region.first) {
      return false;
    }
  }
  return true;
}

residency::FrameRegion
virtual_graph_host_input_region(const VirtualRunProjection &run,
                                const std::size_t input,
                                const std::uint32_t bank) noexcept {
  std::uint64_t bank_capacity = 0u;
  std::uint64_t bank_offset = 0u;
  std::uint64_t input_offset = 0u;
  std::uint64_t first = 0u;
  if (input >= run.host_input_count || bank >= residency::Pool::BankCount ||
      run.host_frame_capacity == 0u ||
      !kernel::checked::mul(run.host_frame_capacity, run.host_input_count,
                            bank_capacity) ||
      !kernel::checked::mul(bank_capacity, bank, bank_offset) ||
      !kernel::checked::mul(run.host_frame_capacity, input, input_offset) ||
      !kernel::checked::add(run.first_host_input_frame, bank_offset, first) ||
      !kernel::checked::add(first, input_offset, first) ||
      first > std::numeric_limits<std::uint32_t>::max()) {
    return {};
  }
  return residency::FrameRegion{
      .tier = residency::FrameTier::Host,
      .role = residency::FrameRole::Input,
      .first = static_cast<std::uint32_t>(first),
      .count = run.host_frame_capacity,
  };
}

} // namespace rund::compute::detail
