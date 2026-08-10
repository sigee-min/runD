#pragma once

#include "../pipeline/residency/model.hpp"

#include <cstdint>

namespace rund::compute::detail {

struct VirtualActiveProjection final {
  residency::LinearPlan linear{};
  std::uint64_t active_count{};
  std::uint64_t active_slots_peak{};
  std::uint64_t input_bytes{};
  std::uint64_t output_bytes{};
};

[[nodiscard]] bool project_virtual_active(
    const residency::LinearPlan &capacity, std::uint64_t active_count,
    std::uint64_t capacity_count, std::uint64_t input_page_bytes,
    std::uint64_t output_page_bytes, std::uint64_t input_element_bytes,
    std::uint64_t output_element_bytes,
    VirtualActiveProjection &projection) noexcept;

} // namespace rund::compute::detail
