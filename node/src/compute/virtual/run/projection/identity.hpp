#pragma once

#include "../../../pipeline/residency/model.hpp"

#include <cstdint>

namespace rund::compute::detail {

[[nodiscard]] residency::Identity input_materialization_identity(
    Type type, FixedFormat format, std::uint64_t page_bytes,
    std::uint64_t payload_bytes, std::uint64_t prefix_bytes,
    std::uint64_t frame_elements, std::uint32_t boundary) noexcept;

[[nodiscard]] residency::Identity
output_materialization_identity(Type type, FixedFormat format,
                                std::uint64_t page_bytes,
                                std::uint64_t frame_elements) noexcept;

[[nodiscard]] residency::Identity
transient_materialization_identity(residency::Identity graph,
                                   std::uint32_t resource,
                                   std::uint64_t domain) noexcept;

} // namespace rund::compute::detail
