#pragma once

#include "projection.hpp"

#include "../../device/residency.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct VirtualScan final {
  std::array<std::byte, sizeof(std::uint64_t)> carry{};
  Type type{Type::I32};
  bool inclusive{};
};

[[nodiscard]] Status begin_virtual_scan(const VirtualRunProjection &run,
                                        VirtualScan &scan) noexcept;

[[nodiscard]] Status prepare_virtual_scan_page(const VirtualRunProjection &run,
                                               residency::EpochLease lease,
                                               VirtualScan &scan) noexcept;

[[nodiscard]] Status complete_virtual_scan_page(
    VirtualBacking &input, const VirtualEpochProjection &epoch,
    const VirtualRunProjection &run, residency::EpochLease lease,
    VirtualScan &scan, ResidencyStats &stats) noexcept;

} // namespace rund::compute::detail
