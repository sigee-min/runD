#pragma once

#include "../registry.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

// Stateless direct owner for the bounded VirtualBacking output-row
// transaction. The Authority remains the sole owner of the gate, frame table,
// lease slots, and identity counters; this facet owns only this protocol's
// algorithms and borrows that physical state.
class VirtualTransactionOwner final {
public:
  explicit VirtualTransactionOwner(Authority &) noexcept;

  VirtualTransactionOwner(const VirtualTransactionOwner &) noexcept = default;
  VirtualTransactionOwner &operator=(const VirtualTransactionOwner &) = delete;

  // Identity proof used by the Virtual run coordinator without exposing the
  // Authority object as a second transaction API.
  [[nodiscard]] bool owns(const Authority *candidate) const noexcept {
    return candidate == &authority_;
  }

  [[nodiscard]] bool
  tag_virtual_transaction_output(std::uint64_t lease_token, std::uint32_t frame,
                                 CacheKey key, FrameTier tier,
                                 std::uint64_t owner_generation) const noexcept;

  [[nodiscard]] bool prepare_virtual_transaction_lease(
      std::uint64_t backing, std::uint64_t version,
      std::uint64_t materialization_hi, std::uint64_t materialization_lo,
      std::uint64_t page_count, std::uint64_t boundary_page,
      std::uint64_t boundary_extent, std::span<const FrameRegion> regions,
      std::span<const VirtualTransactionLease::Row> journal,
      std::uint64_t owner_generation, std::size_t expected_rows,
      bool require_rows, VirtualTransactionLease &) const noexcept;

  [[nodiscard]] bool cleanup_virtual_transaction_rows(
      std::uint64_t backing, std::uint64_t version,
      std::uint64_t materialization_hi, std::uint64_t materialization_lo,
      std::uint64_t page_count, std::uint64_t boundary_page,
      std::uint64_t boundary_extent, std::span<const FrameRegion> regions,
      std::uint64_t owner_generation) const noexcept;

  void commit_virtual_transaction_lease(
      VirtualTransactionLease &&,
      std::uint64_t committed_version) const noexcept;
  void abort_virtual_transaction_lease(VirtualTransactionLease &&,
                                       bool unknown) const noexcept;

private:
  Authority &authority_;
};

} // namespace rund::compute::detail::residency
