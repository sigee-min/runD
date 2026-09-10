#pragma once

#include "model.hpp"

#include <cstdint>
#include <vector>

namespace rund::compute::detail::residency {

[[nodiscard]] Identity
IdentifyResidencyPlan(std::uint64_t page_bytes, std::uint64_t page_count,
                      std::uint64_t frame_capacity, DirtyRange dirty,
                      std::uint64_t dirty_bytes,
                      std::uint64_t prefetch_distance) noexcept;

[[nodiscard]] Identity
IdentifyResidencyPlan(std::uint64_t page_count, std::uint64_t frame_capacity,
                      std::uint64_t prefetch_distance,
                      std::span<const TiledGraphResource> resources,
                      std::span<const TiledGraphPhysicalClass> physical_classes,
                      std::span<const TiledGraphStage> stages,
                      std::uint64_t graph_fingerprint_hi = 0u,
                      std::uint64_t graph_fingerprint_lo = 0u) noexcept;

} // namespace rund::compute::detail::residency
