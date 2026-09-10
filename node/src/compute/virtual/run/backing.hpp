#pragma once

#include "projection.hpp"

#include "../../device/residency/prefetch.hpp"
#include "../../device/residency/registry.hpp"

#include <rund/compute/stats.hpp>

#include <span>

namespace rund::compute::detail {

namespace residency {
class Pool;
}

struct VirtualSupplyResult final {
  Status status{Status::success()};
  std::uint64_t fetched_pages{};
  std::uint64_t backing_bytes{};
  std::uint64_t late_pages{};
};

struct VirtualInputMaterialization final {
  std::uint64_t page{};
  std::byte *frame{};
  bool read{};
  bool finalize{};
};

// One authenticated predecessor footprint is sufficient for a monotonic page
// stream. The recurrent controller advances this cell only after the exact
// Input materialization succeeds, so storage remains O(1) in epoch count.
struct VirtualInputReuseSeed final {
  std::uint64_t page{ResidencyStats::no_failed_page};
  const std::byte *frame{};
};

struct VirtualInputMaterializationResult final {
  Status status{Status::success()};
  std::uint64_t read_pages{};
  std::uint64_t backing_bytes{};
};

[[nodiscard]] VirtualInputMaterializationResult materialize_virtual_input(
    VirtualBacking &backing, const VirtualRunProjection &run,
    std::span<const VirtualInputMaterialization> pages,
    const VirtualInputReuseSeed *prior = nullptr) noexcept;

[[nodiscard]] Status
validate_virtual_recovery(const VirtualBacking &input,
                          const VirtualBacking &output,
                          std::uint64_t required_output_bytes) noexcept;

[[nodiscard]] Status
validate_virtual_recovery(std::span<VirtualBacking *const> inputs,
                          const VirtualBacking &output,
                          std::uint64_t required_output_bytes) noexcept;

[[nodiscard]] Status
validate_virtual_run_recovery(const VirtualRunProjection &,
                              std::span<VirtualBacking *const>,
                              const VirtualBacking &) noexcept;

[[nodiscard]] VirtualSupplyResult
read_virtual_epoch(VirtualBacking &backing, const VirtualEpochProjection &epoch,
                   const VirtualRunProjection &run, residency::EpochLease lease,
                   const residency::PrefetchReceipt &prefetched,
                   ResidencyStats &stats,
                   VirtualInputReuseSeed *reuse = nullptr) noexcept;

[[nodiscard]] Status schedule_virtual_prefetch(
    VirtualBacking &backing, const VirtualEpochProjection &epoch,
    const VirtualRunProjection &run, residency::Pool &pool,
    residency::Prefetcher &prefetcher, bool speculative, bool &pending,
    bool &cleanup_failed, std::span<std::byte> coherent_input = {},
    bool coherent_deferred = false) noexcept;

void clear_virtual_recovery(VirtualBacking &backing) noexcept;

} // namespace rund::compute::detail
