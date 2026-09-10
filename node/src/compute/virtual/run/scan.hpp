#pragma once

#include "projection.hpp"

#include "../../device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail {

namespace residency {
struct PrefetchReceipt;
}

struct VirtualScan final {
  std::array<std::byte, sizeof(std::uint64_t)> carry{};
  std::array<std::array<std::byte, sizeof(std::uint64_t)>, PipelineLeafCapacity>
      injected{};
  std::array<std::array<std::byte, sizeof(std::uint64_t)>, PipelineLeafCapacity>
      original{};
  std::array<std::array<std::byte, sizeof(std::uint64_t)>, PipelineLeafCapacity>
      input_last{};
  std::array<std::uint64_t, PipelineLeafCapacity> input_pages{};
  Type type{Type::I32};
  std::uint64_t failed_page{std::numeric_limits<std::uint64_t>::max()};
  std::size_t input_count{};
  bool inclusive{};
};

[[nodiscard]] Status begin_virtual_scan(const VirtualRunProjection &run,
                                        VirtualScan &scan) noexcept;

[[nodiscard]] Status capture_virtual_scan_input(
    const VirtualEpochProjection &epoch, const VirtualRunProjection &run,
    residency::EpochLease input_lease,
    const residency::PrefetchReceipt *prefetched, VirtualScan &scan) noexcept;

[[nodiscard]] Status prepare_virtual_scan_uniform(
    const VirtualEpochProjection &epoch, const VirtualRunProjection &run,
    residency::EpochLease input_lease, residency::EpochLease output_lease,
    VirtualScan &scan) noexcept;

[[nodiscard]] Status upload_virtual_scan_uniform(
    PipelineState &pipeline, const VirtualRunProjection &run,
    residency::EpochLease lease, const VirtualScan &scan, Stats &stats,
    bool restore) noexcept;

} // namespace rund::compute::detail
