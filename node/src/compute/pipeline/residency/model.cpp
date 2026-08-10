#include "model.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail::residency {

bool LinearPlan::wave(const std::uint64_t index, PageRun &run) const noexcept {
  std::uint64_t first_page = 0u;
  if (index >= wave_count() || slot_capacity_ == 0u ||
      !kernel::checked::mul(index, slot_capacity_, first_page) ||
      first_page >= page_count_) {
    return false;
  }
  run = PageRun{
      .first_page = first_page,
      .page_count = std::min(slot_capacity_, page_count_ - first_page),
  };
  return true;
}

ResidencyPlan::ResidencyPlan(const std::uint64_t page_bytes,
                             const LinearPlan linear,
                             const Identity identity) noexcept
    : page_bytes_(page_bytes), linear_(linear), identity_(identity) {}

} // namespace rund::compute::detail::residency
