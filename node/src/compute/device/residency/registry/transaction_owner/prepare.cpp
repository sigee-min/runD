#include "local.hpp"

#include "../lease_state.hpp"

#include <limits>
#include <mutex>
#include <utility>

namespace rund::compute::detail::residency {

bool VirtualTransactionOwner::prepare_virtual_transaction_lease(
    const std::uint64_t backing, const std::uint64_t version,
    const std::uint64_t materialization_hi,
    const std::uint64_t materialization_lo, const std::uint64_t page_count,
    const std::uint64_t boundary_page, const std::uint64_t boundary_extent,
    const std::span<const FrameRegion> regions,
    const std::span<const VirtualTransactionLease::Row> journal,
    const std::uint64_t owner_generation, const std::size_t expected_rows,
    const bool require_rows, VirtualTransactionLease &lease) const noexcept {
  lease = VirtualTransactionLease{};
  if (backing == 0u || version == 0u || page_count == 0u ||
      (boundary_page != std::numeric_limits<std::uint64_t>::max() &&
       boundary_page >= page_count) ||
      (require_rows && (expected_rows == 0u ||
                        expected_rows > VirtualTransactionLease::Capacity))) {
    return false;
  }
  std::unique_lock lock{authority_.gate_};
  if (authority_.view_commit_quarantined_locked()) {
    return false;
  }
  const bool captured =
      authority_.execution_state_.slot.token == 0u &&
      !any_active(authority_.cycle_state_.epochs) &&
      !active(authority_.cycle_state_.writeback) &&
      transaction_detail::capture_rows(
          authority_.frames_, backing, version, materialization_hi,
          materialization_lo, page_count, boundary_page, boundary_extent,
          regions, journal, owner_generation, require_rows, lease);
  if (!captured || (require_rows && lease.row_count != expected_rows)) {
    lease.row_count = 0u;
    return false;
  }
  lease.authority = &authority_;
  lease.active = true;
  lease.gate = std::move(lock);
  return true;
}

} // namespace rund::compute::detail::residency
