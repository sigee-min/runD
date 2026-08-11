#pragma once

#include "projection.hpp"

#include "../../device/residency.hpp"
#include "../../device/residency_prefetch.hpp"

#include <rund/compute/stats.hpp>

namespace rund::compute::detail {

struct VirtualSupplyResult final {
  Status status{Status::success()};
  std::uint64_t fetched_pages{};
  std::uint64_t backing_bytes{};
  std::uint64_t late_pages{};
};

[[nodiscard]] Status
validate_virtual_recovery(const VirtualBacking &input,
                          const VirtualBacking &output,
                          std::uint64_t required_output_bytes) noexcept;

[[nodiscard]] VirtualSupplyResult
read_virtual_epoch(VirtualBacking &backing, const VirtualEpochProjection &epoch,
                   const VirtualRunProjection &run, residency::EpochLease lease,
                   const residency::PrefetchReceipt &prefetched,
                   ResidencyStats &stats) noexcept;

[[nodiscard]] Status schedule_virtual_prefetch(
    VirtualBacking &backing, const VirtualEpochProjection &epoch,
    const VirtualRunProjection &run, const residency::Authority &authority,
    residency::Prefetcher &prefetcher, bool &pending) noexcept;

void clear_virtual_recovery(VirtualBacking &backing) noexcept;

} // namespace rund::compute::detail
