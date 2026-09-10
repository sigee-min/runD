#pragma once

#include "../transaction.hpp"

#include <array>

namespace rund::compute::detail::transaction_detail {

[[nodiscard]] std::uint64_t
next_backing_version(std::uint64_t version) noexcept;

[[nodiscard]] bool transaction_regions(const VirtualRunTransaction &,
                                       std::array<residency::FrameRegion, 4u> &,
                                       std::size_t &) noexcept;

[[nodiscard]] bool valid_captured_region(residency::FrameRegion,
                                         residency::FrameTier) noexcept;

[[nodiscard]] bool disjoint_captured_regions(
    const std::array<residency::FrameRegion, 2u> &) noexcept;

[[nodiscard]] residency::Authority *
scan_authority(const VirtualPipelineState &) noexcept;

[[nodiscard]] bool valid_output_lease(const VirtualRunTransaction &,
                                      const residency::Authority *) noexcept;

[[nodiscard]] Status
validate_transaction_base(const VirtualPipelineState &,
                          const VirtualRunTransaction &) noexcept;

void clear_transaction(VirtualRunTransaction &) noexcept;

} // namespace rund::compute::detail::transaction_detail
