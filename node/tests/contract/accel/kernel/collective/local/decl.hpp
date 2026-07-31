#pragma once

#include <accel/device.hpp>

#include "sort.hpp"

namespace node_accel_contract::collective {

[[nodiscard]] bool GraphContract();
[[nodiscard]] bool RequiredMetalScan(const rund::AccelDevice &pick);
[[nodiscard]] bool RequiredVulkanScan(const rund::AccelDevice &pick);
[[nodiscard]] bool BackendRunsScanSortCompact(const rund::AccelDevice &pick);
[[nodiscard]] bool
AvailableBackendsRunSameSort(const rund::AccelDevice &metal,
                             const rund::AccelDevice &vulkan);

} // namespace node_accel_contract::collective
