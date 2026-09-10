#pragma once

#include "../transaction_owner.hpp"

namespace rund::compute::detail::residency::transaction_detail {

[[nodiscard]] bool
virtual_key(const CacheKey &, std::uint64_t backing, std::uint64_t version,
            std::uint64_t materialization_hi, std::uint64_t materialization_lo,
            std::uint64_t page_count, std::uint64_t boundary_page,
            std::uint64_t boundary_extent) noexcept;
[[nodiscard]] bool valid_regions(const std::vector<Authority::Frame> &,
                                 std::span<const FrameRegion>) noexcept;
[[nodiscard]] bool
capture_rows(const std::vector<Authority::Frame> &, std::uint64_t backing,
             std::uint64_t version, std::uint64_t materialization_hi,
             std::uint64_t materialization_lo, std::uint64_t page_count,
             std::uint64_t boundary_page, std::uint64_t boundary_extent,
             std::span<const FrameRegion>,
             std::span<const VirtualTransactionLease::Row>,
             std::uint64_t owner_generation, bool require_rows,
             VirtualTransactionLease &) noexcept;
void consume(VirtualTransactionLease &) noexcept;

} // namespace rund::compute::detail::residency::transaction_detail
