#pragma once

#include "../pipeline/residency/model.hpp"

#include <cstdint>

namespace rund::compute::detail {

struct VirtualActiveProjection final {
  residency::StreamPlan stream{};
  residency::TiledGraphInvocation graph{};
  std::uint64_t active_count{};
  std::uint64_t resident_frames_peak{};
  std::uint64_t input_bytes{};
  std::uint64_t output_bytes{};
};

[[nodiscard]] bool project_virtual_graph_active(
    const residency::TiledGraphPlan &capacity, std::uint64_t active_count,
    std::uint64_t capacity_count, std::uint64_t input_payload_elements,
    std::uint64_t input_element_bytes, std::uint64_t output_element_bytes,
    VirtualActiveProjection &projection) noexcept;

[[nodiscard]] bool project_virtual_active(
    const residency::StreamPlan &capacity, std::uint64_t active_count,
    std::uint64_t capacity_count, std::uint64_t input_payload_elements,
    std::uint64_t active_output_count, std::uint64_t input_element_bytes,
    std::uint64_t output_element_bytes, bool transient_page_outputs,
    VirtualActiveProjection &projection) noexcept;

} // namespace rund::compute::detail
