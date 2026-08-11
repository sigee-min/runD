#include "active.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>

namespace rund::compute::detail {

bool project_virtual_active(const residency::StreamPlan &capacity,
                            const std::uint64_t active_count,
                            const std::uint64_t capacity_count,
                            const std::uint64_t input_payload_elements,
                            const std::uint64_t active_output_count,
                            const std::uint64_t input_element_bytes,
                            const std::uint64_t output_element_bytes,
                            VirtualActiveProjection &projection) noexcept {
  projection = {};
  if (active_count > capacity_count || capacity.frame_capacity() == 0u ||
      input_element_bytes == 0u || output_element_bytes == 0u ||
      input_payload_elements == 0u) {
    return false;
  }
  const std::uint64_t full_pages =
      capacity_count / input_payload_elements +
      static_cast<std::uint64_t>(capacity_count % input_payload_elements != 0u);
  const std::uint64_t active_pages =
      active_count / input_payload_elements +
      static_cast<std::uint64_t>(active_count % input_payload_elements != 0u);
  if (capacity.page_count() != full_pages ||
      !kernel::checked::mul(active_count, input_element_bytes,
                            projection.input_bytes) ||
      !kernel::checked::mul(active_output_count, output_element_bytes,
                            projection.output_bytes)) {
    projection = {};
    return false;
  }
  projection.stream =
      residency::StreamPlan{active_pages, capacity.frame_capacity()};
  projection.active_count = active_count;
  projection.resident_frames_peak =
      std::min(active_pages, capacity.frame_capacity());
  return true;
}

} // namespace rund::compute::detail
