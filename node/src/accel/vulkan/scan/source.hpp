#pragma once

#include "stage.hpp"

#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/scan/model.hpp>

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {

inline constexpr std::uint32_t kVulkanScanWidth = 128u;

[[nodiscard]] std::string VulkanScanSource(rund::kernel::ScanElement element,
                                           rund::kernel::ComputeDomain domain,
                                           VulkanScanStage stage,
                                           bool inclusive);
[[nodiscard]] bool VulkanScanSourceBytes(
    rund::kernel::ScanElement element, rund::kernel::ComputeDomain domain,
    VulkanScanStage stage, bool inclusive, std::uint64_t &bytes) noexcept;

} // namespace rund::node::accel::detail
