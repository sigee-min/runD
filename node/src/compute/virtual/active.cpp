#include "active.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail {

bool project_virtual_active(const residency::LinearPlan &capacity,
                            const std::uint64_t active_count,
                            const std::uint64_t capacity_count,
                            const std::uint64_t input_page_bytes,
                            const std::uint64_t output_page_bytes,
                            const std::uint64_t input_element_bytes,
                            const std::uint64_t output_element_bytes,
                            VirtualActiveProjection &projection) noexcept {
  projection = {};
  if (active_count > capacity_count || capacity.slot_capacity() == 0u ||
      input_element_bytes == 0u || output_element_bytes == 0u ||
      input_page_bytes == 0u || output_page_bytes == 0u ||
      input_page_bytes % input_element_bytes != 0u ||
      output_page_bytes % output_element_bytes != 0u) {
    return false;
  }
  const std::uint64_t page_elements = input_page_bytes / input_element_bytes;
  const std::uint64_t output_page_elements =
      output_page_bytes / output_element_bytes;
  if (page_elements == 0u || page_elements != output_page_elements) {
    return false;
  }
  const std::uint64_t full_pages =
      capacity_count / page_elements +
      static_cast<std::uint64_t>(capacity_count % page_elements != 0u);
  const std::uint64_t active_pages =
      active_count / page_elements +
      static_cast<std::uint64_t>(active_count % page_elements != 0u);
  if (capacity.page_count() != full_pages ||
      !kernel::checked::mul(active_count, input_element_bytes,
                            projection.input_bytes) ||
      !kernel::checked::mul(active_count, output_element_bytes,
                            projection.output_bytes)) {
    projection = {};
    return false;
  }
  projection.linear =
      residency::LinearPlan{active_pages, capacity.slot_capacity()};
  projection.active_count = active_count;
  projection.active_slots_peak =
      std::min(active_pages, capacity.slot_capacity());
  return true;
}

} // namespace rund::compute::detail
