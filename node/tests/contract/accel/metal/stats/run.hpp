#pragma once

#include <accel/device.hpp>

namespace node_accel_contract {

[[nodiscard]] bool
MetalRepeatedStagedRunsReportWarmRuntimeStats(const rund::AccelDevice &pick);

} // namespace node_accel_contract
