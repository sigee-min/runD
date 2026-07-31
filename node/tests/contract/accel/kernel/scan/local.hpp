#pragma once

#include <accel/device.hpp>

#include "../primitive/local.hpp"

namespace node_accel_contract {

[[nodiscard]] bool BackendRunsInclusiveScan(const rund::AccelDevice &pick);
[[nodiscard]] bool RequiredMetalRunsInclusiveScan();
[[nodiscard]] bool RequiredVulkanRunsInclusiveScan();
[[nodiscard]] bool
ScanThenMapPreservesInternalRoundtrip(const rund::AccelDevice &pick);

} // namespace node_accel_contract
